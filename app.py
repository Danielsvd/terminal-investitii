import streamlit as st
import logging
logger = logging.getLogger(__name__)
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import feedparser
from data.news import filter_company_news, future_note, holdings_news, mark_future, news_card_html, safe_link, plain_text
from data import fred_calendar as fred_cal
from data import sheet_view
from data import universes
from ui import glosar
from ui.card_help import FUND_HELP, fund_help, BVB_HELP, PILLAR_HELP, PILLAR_TEST_NOTE, RISK_HELP, OPT_HELP, PORTF_HELP, MARKET_HELP
from ui import theme as ui_theme
from streamlit.delta_generator import DeltaGenerator as _DeltaGenerator
# Cardurile HTML vechi au culori din paleta GitHub dark: la afișare trec pe tokenurile temei (ui/theme.py).
ui_theme.install(st, _DeltaGenerator)
# Culorile din tabele (canvas, fără var()): după modul temei, cu rezervă lizibilă în ambele moduri.
try:
    _THEME_TYPE = st.context.theme.type
except (AttributeError, RuntimeError):
    _THEME_TYPE = None
DF_UP, DF_DOWN, DF_MUTED = ui_theme.df_colors(_THEME_TYPE)
CORR_SCALE, CORR_TEXT = ui_theme.corr_heatmap_colors(_THEME_TYPE)   # matricile de corelație, după mod
sign_color = lambda v: ui_theme.sign_css(v, DF_UP, DF_DOWN)  # celule text „+x” verde, „-x” roșu
import os
import time
from datetime import datetime, timedelta
import socket
import numpy as np
import re
import html
import requests
import asyncio
import httpx
import pandas_datareader.data as web
import gspread
from google.oauth2.service_account import Credentials
import concurrent.futures
from collections import deque
from threading import Lock

# Module proprii: funcții de calcul pure, acoperite de teste (vezi tests/)
from analytics.technical import atr_trailing_stop, rsi_wilder, macd as macd_lines, price_changes
from analytics.macro import yoy_pct, real_rate, market_of, monthly_spread, same_day_spread, tbill_discount_to_bey
from analytics import macro as macro_calc
from analytics import markets as markets_calc
from analytics import rates as rates_calc
from analytics import stress as stress_calc
from analytics import briefing as briefing_calc
from analytics import day_notes
from analytics import portfolio_day as pday_calc
from analytics import scanner as scanner_calc
from analytics.earnings import earnings_table
from analytics import watchlist as watchlist_calc
from analytics import growth
from analytics.portfolio import value_positions, portfolio_curve as build_portfolio_curve, positions_detail
from analytics import portfolio as pf_calc
from analytics import fundamentals as fund
from analytics import ledger as ledger_calc
from analytics import simulation
from analytics import options as opt_calc
from data import portfolio_sheet
from data import iv_history
from data import broker_import
from analytics import ideas as ideas_calc
from analytics import bvb_fundamental as bvb_calc
from analytics import pilon2 as p2_calc
from data import pilon2 as pilon2_data
from analytics import sec_13f, sec_view, insiders, schedule13
from data import sec_edgar, sec_funds, cusip_map, sec_insiders, sec_ownership
from data import sec_xbrl
from analytics import xbrl_facts
from data.ucits import ucits_for, ucits_symbols, ucits_currency
from analytics import events as events_calc
from data import entities as entities_data
import math
import zipfile
from analytics.risk import beta_benchmark, beta_weekly, beta_corr_weekly, jensen_alpha, performance_risk, common_day_correlation
from analytics.risk import CORR_HIGH, CORR_MID, CORR_BANDS, correlation_band
from analytics.audit import get_sector_benchmarks, generate_advanced_audit_v2
from analytics.scoring import technical_subscore, technical_components, moment_label, rsi_description, explain_moment
from analytics.decision import sentiment_band, news_window
from analytics import global_fundamental as gfund
from analytics.bvb_fundamental import current_indicators, cross_section, verdict_banner, score_rank, explain_fundamental, piotroski_value_text, final_verdict, GN_LIMIT, TEST_NOTE
from data.company_files import load_company, is_bank_template, ticker_from_title, period_label, FIELD_LABELS as COMPANY_FIELD_LABELS
from data.company_files import annual_eps_growth, annual_pair, latest_ratio
from analytics.peers import PEERS, METRICS as PEER_METRICS, peer_region, peer_list, peer_medians, versus_median, peer_delta_color
from analytics.peers import BVB_REGION, bvb_sector, bvb_sector_peers, bvb_regional_peers, peer_symbols, sheet_peer_row, format_peer_value, split_financial_peers
from data.bvb_sheet import parse_bvb_sheet, bvb_symbol, reprice as reprice_bvb, unmapped_info_keys
from data import bnr as bnr_data
from data.helpers import format_ro, score_ro, num, close_frame, field_frame, slice_window, now_ro, struct_time_utc_to_ro, smart_to_float, parse_ecb_csv, parse_ecb_series, parse_eurostat_monthly, positive_or_none, price_ro, entry_target_view, scale_number, watchlist_status, target_zone, NEAR_TARGET_PCT, ro, ro_spec, md_bold_html, text_frame, na_safe_frame

# =============================================================================
# ARHITECTURĂ #5: RATE LIMITER YAHOO FINANCE
# Previne eroarea 429 (Too Many Requests) prin limitarea la 5 req/secundă
# =============================================================================
class _RateLimiter:
    def __init__(self, max_calls=5, period=1.0):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
        self._lock = Lock()

    def wait_if_needed(self):
        with self._lock:
            now = time.time()
            while self.calls and now - self.calls[0] > self.period:
                self.calls.popleft()
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            self.calls.append(time.time())

_yf_limiter = _RateLimiter(max_calls=5, period=1.0)

async def fetch_ticker_price_async(client, ticker):
    """Cere prețul unui singur ticker în mod asincron."""
    # Folosim API-ul intern de chart al Yahoo pentru viteză maximă și date 'light'
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?range=1d&interval=1d"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    # La eșec întoarce None, nu 0: o poziție evaluată la 0 ar apărea ca pierdere de 100%.
    try:
        response = await client.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            price = data['chart']['result'][0]['meta']['regularMarketPrice']
            return ticker, float(price)
        logger.warning(f"preț live {ticker}: HTTP {response.status_code}")
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as e:
        logger.warning(f"preț live {ticker} indisponibil: {e}")
    return ticker, None

async def get_all_portfolio_prices(tickers):
    """Lansează toate cererile simultan."""
    async with httpx.AsyncClient() as client:
        tasks = [fetch_ticker_price_async(client, t) for t in tickers]
        results = await asyncio.gather(*tasks)
        return dict(results)

@st.cache_data(ttl=300, show_spinner=False)
def _live_prices_cached(tickers):
    """Prețurile live pentru un tuplu de simboluri, în cache 5 minute (inclusiv golurile None; get_fast_live_prices
    le recere separat la fiecare apel). Eroarea generală (event loop, httpx) iese ca excepție și nu intră în cache."""
    return asyncio.run(get_all_portfolio_prices(list(tickers)))

# Wrapper pentru a putea rula cod asincron în interiorul Streamlit (care e sincron)
def get_fast_live_prices(tickers):
    """Prețuri curente: dict simbol -> preț, sau None pentru simbolurile care nu au putut fi citite.

    Prețurile citite stau 5 minute în cache (`_live_prices_cached`): înainte, fiecare clic din Portofoliu cerea din nou
    prețul fiecărei poziții (o dată pentru registru și încă o dată în fiecare tab de monedă). Simbolurile rămase fără
    preț se recer la fiecare apel, fără cache, ca un refuz temporar Yahoo (429) să nu țină 5 minute."""
    if not tickers: return {}
    key = tuple(dict.fromkeys(tickers))   # ordinea păstrată, fără dubluri
    try:
        prices = dict(_live_prices_cached(key))
        missing = [t for t in key if prices.get(t) is None]
        if missing:
            retry = asyncio.run(get_all_portfolio_prices(missing))
            prices.update({t: v for t, v in retry.items() if v is not None})
        return prices
    except (RuntimeError, httpx.HTTPError) as e:
        logger.warning(f"prețuri live indisponibile: {e}")
        return {t: None for t in key}

# --- 0. CONFIGURARE GLOBALĂ ---
st.set_page_config(page_title="Terminal Investiții PRO", page_icon="📈", layout="wide")
socket.setdefaulttimeout(15) # Mărit timeout-ul pentru conexiuni lente

# =============================================================================
# ARHITECTURĂ #5: SINGLETON GOOGLE SHEETS
# ÎNAINTE: connect_to_gsheets() era apelată la FIECARE rerun (~50 auth/min)
# ACUM: O singură autentificare per sesiune prin @st.cache_resource
# =============================================================================
@st.cache_resource(show_spinner=False)
def _get_gsheets_client():
    """Singleton client gspread — creat O SINGURĂ DATĂ per sesiune."""
    scope = ["https://www.googleapis.com/auth/spreadsheets",
             "https://www.googleapis.com/auth/drive"]
    try:
        if "gcp_service_account" not in st.secrets:
            st.error("⚠️ Nu s-au găsit credențialele în Secrets!")
            return None
        creds = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]), scopes=scope
        )
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"Eroare autentificare Google: {e}")
        return None

@st.cache_resource(show_spinner=False)
def _get_spreadsheet():
    """Singleton spreadsheet — deschis O SINGURĂ DATĂ per sesiune."""
    client = _get_gsheets_client()
    if client:
        try:
            return client.open("portofoliu_db")
        except Exception as e:
            st.error(f"Nu s-a putut deschide 'portofoliu_db': {e}")
    return None

def connect_to_gsheets(sheet_name=None):
    """
    Compatibilitate completă cu codul existent.
    Dacă sheet_name=None → returnează Sheet1 (portofoliu principal)
    Dacă sheet_name="watchlist" → returnează tab-ul watchlist
    ZERO autentificări noi — folosește singleton-ul din cache.
    """
    spreadsheet = _get_spreadsheet()
    if not spreadsheet:
        return None
    try:
        if sheet_name is None:
            return spreadsheet.sheet1
        return spreadsheet.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        return None
    except Exception as e:
        st.error(f"Eroare acces worksheet '{sheet_name}': {e}")
        return None

# Tokenurile de culoare ale temei (ui/theme.py, DESIGN.md): --tk-* cu light-dark(), urmează modul ales în meniu.
st.markdown(f"<style>{ui_theme.root_css()}</style>", unsafe_allow_html=True)

# --- CSS MODERNIZAT (UI PREMIUM) ---
st.markdown("""
    <style>
    /* Stil general aplicație */
    /* Fundalul paginii vine din tema (.streamlit/config.toml), ca să meargă și modul luminos. */
    
    /* Carduri Principale */
    .fin-card, .news-card {
        background-color: #161B22;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #30363D;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px;
        transition: transform 0.2s;
    }
    .fin-card:hover, .news-card:hover { border-color: #58A6FF; }

    /* Stilizare Metrici (KPIs) */
    div[data-testid="stMetric"] {
        background-color: var(--tk-surface);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #30363D;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * { font-size: 14px; color: var(--tk-muted); }
    div[data-testid="stMetricValue"] { font-size: 24px; font-weight: 600; color: var(--tk-text); font-variant-numeric: tabular-nums; }

    /* Stilizare Știri */
    .news-card { border-left: 5px solid #238636; }
    .news-title {
        font-size: 18px; font-weight: 600; color: #58A6FF !important;
        text-decoration: none; margin-bottom: 8px; display: block;
    }
    .news-meta {
        font-size: 12px; color: #8B949E; margin-bottom: 10px;
        border-bottom: 1px solid #30363D; padding-bottom: 5px;
    }

    /* Bara Progres Analiști */
    .analyst-bar-container {
        width: 100%; background-color: #30363D; height: 12px;
        border-radius: 6px; position: relative; margin-top: 10px; margin-bottom: 5px;
    }
    .analyst-bar-gradient {
        width: 100%; height: 100%; border-radius: 6px;
        background: linear-gradient(90deg, #238636 0%, #d29922 50%, #da3633 100%); opacity: 0.8;
    }
    .analyst-marker {
        position: absolute; top: -4px; width: 4px; height: 20px;
        background-color: #FFFFFF; border: 1px solid #000;
        box-shadow: 0 0 5px rgba(255,255,255,0.8); z-index: 10; transform: translateX(-50%);
    }
    .analyst-labels {
        display: flex; justify-content: space-between; font-size: 12px; color: #8B949E; margin-top: 5px;
    }

    /* Sentiment Tags */
    .impact-poz { color: #3FB950; font-weight: bold; background: rgba(63, 185, 80, 0.1); padding: 2px 6px; border-radius: 4px; }
    .impact-neg { color: #F85149; font-weight: bold; background: rgba(248, 81, 73, 0.1); padding: 2px 6px; border-radius: 4px; }
    .impact-neu { color: #8B949E; font-weight: bold; background: rgba(139, 148, 158, 0.1); padding: 2px 6px; border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)
# Suprafețe „premium” (ui/theme.premium_css, DESIGN.md §4): după CSS-ul vechi, ca să-l înlocuiască.
st.markdown(f"<style>{ui_theme.premium_css()}</style>", unsafe_allow_html=True)

# --- 1. CONFIGURARE AGREGATOR ---
RSS_CONFIG = {
    "Feeds": [
        "https://www.zf.ro/rss",                    
        "https://www.biziday.ro/feed/",             
        "https://www.economica.net/rss",            
        "https://www.bursa.ro/_rss/?t=pcaps",      
        "https://www.profit.ro/rss",                
        "https://www.startupcafe.ro/rss",           
        "https://financialintelligence.ro/feed/",   
        "https://www.wall-street.ro/rss/business",
        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=19854910", # CNBC Asia-Pacific
        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=19832390", # CNBC Asia News
        "http://feeds.bbci.co.uk/news/world/asia/rss.xml", # BBC Asia
        "https://www.scmp.com/rss/91/feed", # South China Morning Post (Excelent pt China/HK)
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC,EURUSD=X,GC=F,CL=F&region=US&lang=en-US", 
        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
        "http://feeds.marketwatch.com/marketwatch/topstories",
        "https://www.investing.com/rss/news.rss"    
    ],
    "Categorii": {
        "General": [], 
        "Tehnologie": ["tehnologie", "tech", "it", "ai", "software", "hardware", "digital", "cyber", "apple", "microsoft", "google", "nvidia", "oracle", "amazon", "adobe", "asml", "tsm", "palantir", "qualcomm", "micron", "amd", "meta", "broadcom", "intel", "innodata", "crypto", "blockchain", "semiconductori", "startup"],
        "Energie": ["energie", "petrol", "gaze", "oil", "wti", 'VG', "energy", "curent", "hidroelectrica", "omv", "romgaz", "nuclearelectrica", "electrica", "simtel", "transelectrica", "transgaz", "regenerabil", "eolian", "solar", "fotovoltaic", "exxon", "chevron", "devon", "lng", "oklo", "shell", "vistra", "nuscale"],
        "Financiar": ["banca", "bank", "credit", "bursa", "finante", "fonduri", "asigurari", "bvb", "fiscal", "profit", "taxe", "buget", "wall street", "jpm", "unicredit", "ubs", "goldman", "dobanda", "monetar", "Banca Transilvania", "BRD", "BAC", "WFC", "AXP", "JP", "Visa", "BNP", "GS", "Mastercard", "investitii"],
        "Farma": ["farma", "pharma", "sanatate", "medicament", "spital", "medical", "pfizer", "nvo", "sanofi", "eli lilly", "novartis", "biogen", "medicover", "medlife", "regina maria", "BIO", "Antibiotice", "biotech"],
        "Militar": ["militar", "aparare", "defense", "armata", "razboi", "nato", "arme", "securitate", "geopolitic", "taiwan", "ucraina", "rusia", "lmt", "raytheon", "bae", "Leonardo", "Boeing", "rheinmetall", "Thales", "Vinci", "Red Cat", "drone"],
        "Imobiliare": ["imobiliare", "real estate", "apartament", "garsoniera", "casa", "vila", "locuinta", "teren", "birou", "birouri", "santier", "dezvoltator", "rezidential", "chirie", "chirii", "ipotecar", "reit", "mall", "spatii comerciale", "impact", "one united"],
        "Auto": ["auto", "masini", "ev", "electric", "dacia", "ford", "tesla", "volkswagen", "bmw", "mercedes", "byd", "xpeng", "nio", "toyota", "audi", "ferrari", "inmatriculari", "autostrada"],
        "Asia": ["asia", "china", "japonia", "tokyo", "beijing", "shanghai", "hong kong", "taiwan", "india", "seul", "coreea", "nikkei", "yen", "yuan", "rupee", "boj", "evergrande", "alibaba", "tencent", "tsmc", "nifty", "hang seng"],
        "Aur/Metale": ["aur", "gold", "argint", "silver", "metal", "cupru", "precious", "aluminiu", "otel", "minereu", "rio tinto", "bhp", "METC", "glencore", "mp materials"],
        "Macro/Joburi": ["inflatie", "cpi", "pce", "fed", "bce", "robor", "ircc", "bnr", "somaj", "jobs", "angajari", "salarii", "pib", "tarif", "gdp", "pmi", "recesiune", "dobanzi", "economie"]
    }
}

# --- FUNCȚII UTILITARE ---
def parse_date(entry):
    """Data publicării unei știri, în ora României.

    feedparser întoarce `published_parsed` în UTC. Serverul rulează tot în UTC,
    deci fără conversie orele apăreau cu 2-3 ore în urmă față de România.
    """
    try:
        if getattr(entry, 'published_parsed', None):
            return struct_time_utc_to_ro(entry.published_parsed)
        if getattr(entry, 'updated_parsed', None):
            return struct_time_utc_to_ro(entry.updated_parsed)
    except (TypeError, ValueError, OverflowError) as e:
        logger.warning(f"dată de știre neinterpretabilă: {e}")
    return now_ro()

# --- FUNCȚIE NOUĂ DE PARSARE INTELIGENTĂ (SENIOR FIX) ---
# smart_to_float este acum în data/helpers.py (aceeași logică, cu teste în tests/test_helpers.py)
def plot_ro(fig, **kwargs):
    """st.plotly_chart cu separatori românești: punct la mii, virgulă la zecimale („1.234,5”).
    Se aplică axelor, etichetelor și casetelor de hover; nu schimbă datele graficului.
    Culorile scrise pentru fundal închis (fundal #0E1117, grilă #30363D, galben, cyan) trec pe valori
    vizibile în ambele teme (ui/theme.theme_figure_dict); cu modul temei cunoscut, liniile vechi (portocaliu,
    violet, gri deschis, chihlimbar) primesc nuanța modului (PLOT_TOKENS). Restul stilului vine din tema Streamlit."""
    fig.update_layout(separators=",.")
    fig = go.Figure(ui_theme.theme_figure_dict(fig.to_dict(), up=DF_UP, mode=_THEME_TYPE))
    return st.plotly_chart(fig, **kwargs)

def format_large_currency(val):
    """Formatează numerele mari (Trilioane, Miliarde) pentru afișare string."""
    try:
        if isinstance(val, str):
            val = smart_to_float(val)
        
        if val is None or val == 0: return "-"
        if val >= 1e12: return f"{ro(val/1e12):.2f} T USD"
        if val >= 1e9: return f"{ro(val/1e9):.2f} B USD"
        if val >= 1e6: return f"{ro(val/1e6):.2f} M USD"
        return f"{ro(val):,.2f} USD"
    except Exception as exc:  # înainte: except: gol
        logger.warning("format_large_currency: %s", exc)
        return str(val)
    
def format_num(val, is_pct=False, small_decimals=None):
    """Formatare afișare (folosește smart_to_float intern). Cu `small_decimals`, valorile sub 1 în modul
    se afișează cu atâtea zecimale (EPS de 0,0446 lei nu devine 0,04)."""
    if val is None: return "N/A"
    # Asigurăm conversia dacă vine string
    if isinstance(val, str):
        val = smart_to_float(val)
    if pd.isna(val): return "N/A"
        
    if is_pct: return f"{ro(val * 100):.2f}%"
    if small_decimals is not None and 0 < abs(val) < 1:
        return f"{ro(val):.{small_decimals}f}"
    return scale_number(val)

def format_amount(val):
    """Sumă din situațiile financiare, scalată (mld / mil), cu semn. None sau NaN -> 'N/A'."""
    if val is None or pd.isna(val):
        return "N/A"
    sign = "-" if val < 0 else ""
    a = abs(val)
    if a >= 1e12: return f"{sign}{ro(a/1e12):,.2f} T"
    if a >= 1e9: return f"{sign}{ro(a/1e9):,.2f} mld"
    if a >= 1e6: return f"{sign}{ro(a/1e6):,.2f} mil"
    return f"{sign}{ro(a):,.2f}"

def get_macro_interpretation(ticker_data, vix_close=None):
    """Ghidul macro al zilei: semnalele din analytics/macro.py (daily_cross_asset_signals) pe variațiile ultimei ședințe
    a fiecărei serii. Un simbol fără date nu mai contează ca variație 0: regula lui se sare. Randamentul pe 10 ani
    se compară în puncte procentuale, nu în procente din randament."""
    def close_of(t):
        try:
            if isinstance(ticker_data.columns, pd.MultiIndex) and t in ticker_data.columns.levels[0]:
                return ticker_data[t]['Close']
        except (KeyError, AttributeError) as exc:
            logger.warning("get_macro_interpretation %s: %s", t, exc)
        return None

    def pct_of(t):
        c = macro_calc.last_change(close_of(t))
        return None if c is None else c["pct"]

    pct = {"usd": pct_of('DX-Y.NYB'), "gold": pct_of('GC=F'), "oil": pct_of('CL=F'), "copper": pct_of('HG=F')}
    tnx = macro_calc.last_change(close_of('^TNX'))
    vix = macro_calc.last_change(vix_close)
    if all(v is None for v in pct.values()) and tnx is None:
        return ["⚠️ Date insuficiente pentru procesarea corelațiilor macro."]
    texts = {
        "aur_refugiu": "🟡 **AUR:** Refugiu activ. Dolarul slăbește (sub −0,3% azi), iar aurul crește (peste +0,3%).",
        "petrol_dolar": "🛢️ **PETROL:** Presiune valutară. Dolarul puternic (peste +0,3%) scumpește barilul pentru importatori; petrolul scade.",
        "cupru_expansiune": "🏗️ **CUPRU:** Creștere de peste 1% azi: metalele industriale semnalează cerere industrială.",
        "dobanzi_cresc": "📈 **DOBÂNZI:** Randamentul SUA pe 10 ani a crescut cu peste 10 puncte de bază azi: presiune pe acțiunile de creștere "
                         "(valoarea prezentă a profiturilor viitoare scade) și pe companiile mici îndatorate; marje mai bune la bănci.",
        "dobanzi_scad": "🟢 **DOBÂNZI:** Randamentul SUA pe 10 ani a scăzut cu peste 10 puncte de bază azi: costul capitalului scade "
                        "(favorabil tehnologiei și imobiliarelor).",
        "fuga_siguranta": "🚨 **FUGĂ SPRE SIGURANȚĂ:** Aurul (peste +1%), dolarul (peste +0,5%) și VIX (peste +10%) cresc simultan azi.",
    }
    sig = macro_calc.daily_cross_asset_signals(pct, None if tnx is None else tnx["abs"], None if vix is None else vix["pct"])
    out = [texts[k] for k in sig]
    missing = [n for n, v in (("dolar", pct["usd"]), ("aur", pct["gold"]), ("petrol", pct["oil"]), ("cupru", pct["copper"]),
                              ("randament 10 ani", tnx), ("VIX", vix)) if v is None]
    if not out:
        out = ["⚖️ **ECHILIBRU:** Nicio regulă a ghidului nu s-a declanșat azi."]
    out.append("ℹ️ Semnale pe variația unei singure ședințe, praguri orientative, netestate."
               + (f" Fără date azi pentru: {', '.join(missing)} (regulile lor nu s-au evaluat)." if missing else ""))
    return out

def get_watchlist_target(symbol):
    """Extrage prețul țintă din foaia 'watchlist' pentru simbolul analizat."""
    try:
        df_wl = load_watchlist() # Folosește funcția ta existentă de încărcare
        if not df_wl.empty and 'Symbol' in df_wl.columns:
            match = df_wl[df_wl['Symbol'] == symbol]
            if not match.empty:
                return positive_or_none(match.iloc[0]['TargetPrice'])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        logger.warning(f"preț țintă din watchlist pentru {symbol}: {exc}")
    return None

@st.cache_data(ttl=900, show_spinner=False)
def get_bvb_peer_prices(sheet_symbols):
    """Prețul curent al comparabililor BVB (simboluri din foaie, ca tuplu): {simbol: preț sau None}.
    Foaia BVB se actualizează manual, deci multiplii ei pot fi la un preț vechi. În cache 15 minute."""
    prices = get_fast_live_prices([f"{sym}.RO" for sym in sheet_symbols])
    return {sym: prices.get(f"{sym}.RO") for sym in sheet_symbols}

# =============================================================================
# FIȘIERELE „ANALIZĂ COMPANII" (Google Drive, doar citire) -> SCORUL FUNDAMENTAL BVB
# Folderul e partajat de Daniel cu contul de serviciu al aplicației, ca Viewer.
# Fiecare fișier se recitește doar când i se schimbă data modificării (cache pe id + dată).
# =============================================================================
COMPANY_FOLDER_NAME = "Analiză companii"
COMPANY_RANGES = ["'Date Q'!A1:Z60", "'Date Anuale'!A1:Z60"]


@st.cache_data(ttl=3600, show_spinner=False)
def list_company_files():
    """(listă de fișiere {id, name, modified}, mesaj de eroare sau None). Un singur apel Drive."""
    client = _get_gsheets_client()
    if not client:
        return [], "Fără conexiune la Google."
    try:
        query = (f"name = '{COMPANY_FOLDER_NAME}' and mimeType = 'application/vnd.google-apps.folder' "
                 "and trashed = false")
        resp = client.http_client.request("get", "https://www.googleapis.com/drive/v3/files",
                                          params={"q": query, "fields": "files(id,name)", "pageSize": 10})
        folders = resp.json().get("files", [])
        if not folders:
            return [], (f"Folderul „{COMPANY_FOLDER_NAME}” nu este partajat cu contul de serviciu al aplicației "
                        "(Drive → click dreapta pe folder → Share → adresa client_email, drept Viewer).")
        files = client.list_spreadsheet_files(folder_id=folders[0]["id"])
        return [{"id": f["id"], "name": f["name"], "modified": f.get("modifiedTime")} for f in files], None
    except (gspread.exceptions.APIError, requests.exceptions.RequestException, ValueError, KeyError) as e:
        logger.warning(f"lista fișierelor de analiză: {e}")
        return [], f"Lista fișierelor din „{COMPANY_FOLDER_NAME}” nu a putut fi citită: {e}"


@st.cache_data(ttl=30 * 24 * 3600, show_spinner=False, max_entries=300)
def read_company_file(file_id, modified):
    """Valorile foilor „Date Q” și „Date Anuale” (nevizualizate, adică numere, nu text formatat).
    Cheia de cache include data modificării: un fișier editat se recitește automat.
    O eroare nu se pune în cache (excepția trece mai departe)."""
    client = _get_gsheets_client()
    for attempt in (1, 2):
        try:
            res = client.http_client.values_batch_get(file_id, list(COMPANY_RANGES),
                                                      params={"valueRenderOption": "UNFORMATTED_VALUE"})
            ranges = res.get("valueRanges", [])
            return [r.get("values", []) for r in ranges] + [[]] * (2 - len(ranges))
        except gspread.exceptions.APIError as e:
            if attempt == 1 and "429" in str(e):
                time.sleep(10)          # limita Google: 60 de citiri pe minut
                continue
            raise


@st.cache_data(ttl=3600, show_spinner=False)
def build_bvb_dataset(file_keys):
    """Toate fișierele, parsate și validate. `file_keys` = tuplu (id, nume, data modificării).
    Întoarce {companii: {simbol: tabel}, probleme: {simbol: [...]}, fisiere: {simbol: nume}, sarite, erori}."""
    out = {"companii": {}, "probleme": {}, "fisiere": {}, "sarite": [], "erori": []}
    for file_id, name, modified in file_keys:
        tk = ticker_from_title(name)
        if tk is None:
            out["sarite"].append(name)
            continue
        try:
            q_values, a_values = read_company_file(file_id, modified)
        except (gspread.exceptions.APIError, requests.exceptions.RequestException, ValueError) as e:
            out["erori"].append(f"{name}: {e}")
            continue
        if is_bank_template(q_values):
            out["sarite"].append(f"{name} (bancă: alt șablon)")
            continue
        df, issues, missing = load_company(q_values, a_values)
        if df.empty:
            out["sarite"].append(f"{name} (fără perioade cu date)")
            continue
        out["companii"][tk], out["probleme"][tk], out["fisiere"][tk] = df, issues, name
    return out


@st.cache_data(ttl=900, show_spinner=False)
def get_bvb_file_prices(symbols):
    """Prețul curent al emitenților din fișiere (tuplu de simboluri BVB): {simbol: preț sau None}."""
    prices = get_fast_live_prices([f"{sym}.RO" for sym in symbols])
    return {sym: positive_or_none(prices.get(f"{sym}.RO")) for sym in symbols}


def bvb_file_market_cap(symbol, price):
    """Capitalizarea unui emitent BVB din fișierele „Analiză companii”: prețul dat × acțiunile din ultimul
    raport (aceeași regulă ca în secțiunea „Decizie BVB”, inclusiv verificarea de split). None dacă lipsește."""
    if price is None:
        return None
    files, err = list_company_files()
    if err:
        return None
    data = build_bvb_dataset(tuple((f["id"], f["name"], f["modified"]) for f in files))
    df = data["companii"].get(str(symbol).upper().replace(".RO", ""))
    if df is None:
        return None
    mcap = current_indicators(df, price)["mcap"]
    return None if pd.isna(mcap) else float(mcap)



def _bvb_fmt(value, kind):
    if value is None or pd.isna(value):
        return "N/A"
    if kind == "pct":
        return f"{ro(value * 100):.1f}%"
    if kind == "x":
        return f"{ro(value):.2f}"
    return f"{ro(value):.1f}"


def compact_html(fragment):
    """HTML pentru st.markdown fără linii goale și fără indentare. În Markdown, o linie indentată
    cu 4+ spații după o linie goală devine bloc de cod: o listă goală (de ex. nimic „la mijloc”)
    făcea ca restul casetei să apară ca text HTML brut."""
    return "".join(line.strip() for line in fragment.splitlines() if line.strip())


XTB_BULLET_CSS = """<style>
.xtb-bullet-container {background-color:#161B22; padding:25px; border-radius:12px; border:1px solid #30363D; box-shadow:0 4px 6px rgba(0,0,0,0.1);}
.xtb-row {display:flex; align-items:flex-start; margin-bottom:20px;}
.xtb-icon {margin-right:15px; font-size:20px; margin-top:-2px;}
.xtb-text {font-size:15px; color:#C9D1D9; line-height:1.5; font-weight:400;}
.xtb-footer {margin-top:10px; font-size:12px; color:#8B949E; border-top:1px solid #30363D; padding-top:15px; display:flex; justify-content:space-between; align-items:center;}
</style>"""


def render_bullet_box(bullets, footer_text):
    """Caseta cu bulleturi (stil XTB) într-un singur st.markdown.

    Înainte, deschiderea <div>, fiecare rând și închiderea erau trimise în apeluri st.markdown separate:
    Streamlit închide singur fiecare fragment, deci caseta apărea goală, cu rândurile dedesubt.
    Textul bulleturilor e generat de aplicație; titlurile de știri din el sunt deja escapate."""
    rows = "".join(
        f'<div class="xtb-row"><div class="xtb-icon" style="color:{b["color"]};">{b["icon"]}</div>'
        f'<div class="xtb-text">{b["text"]}</div></div>' for b in bullets)
    footer = (f'<div class="xtb-footer"><span>{footer_text}</span>'
              '<span style="color:#A371F7; font-weight:bold;">⚡ Quant-AI Engine</span></div>')
    st.markdown(compact_html(XTB_BULLET_CSS + f'<div class="xtb-bullet-container">{rows}{footer}</div>'),
                unsafe_allow_html=True)


def render_bvb_decision(real_sym, hist):
    """Secțiunea „Decizie BVB în doi pași” pentru simbolurile .RO care au fișier în „Analiză companii”."""
    sym = str(real_sym).upper()
    if not sym.endswith(".RO"):
        return
    st.subheader("🇷🇴 Decizie BVB în doi pași (testată pe istoric)")
    files, err = list_company_files()
    c_reload, _ = st.columns([1, 3])
    if c_reload.button("🔄 Recitește fișierele de analiză", key="bvb_files_reload"):
        list_company_files.clear()
        build_bvb_dataset.clear()
        st.rerun()
    if err:
        st.info(err)
        return
    with st.spinner("Citesc fișierele din „Analiză companii” (prima dată durează până la ~30 de secunde)..."):
        data = build_bvb_dataset(tuple((f["id"], f["name"], f["modified"]) for f in files))
    tk = sym[:-3]
    if tk not in data["companii"]:
        st.caption(f"{tk} nu are fișier în „{COMPANY_FOLDER_NAME}” (sau e bancă, cu alt șablon). "
                   "Secțiunea apare doar pentru emitenții din fișiere.")
        return

    prices = get_bvb_file_prices(tuple(sorted(data["companii"])))
    price = prices.get(tk)
    if price is None and hist is not None and not hist.empty:
        price = positive_or_none(hist["Close"].iloc[-1])
    indicators = {t: current_indicators(df, price if t == tk else prices.get(t))
                  for t, df in data["companii"].items()}
    cs = cross_section(indicators)
    me, row = indicators[tk], cs.loc[tk]
    close = hist["Close"] if hist is not None and not hist.empty else pd.Series(dtype=float)
    moment = technical_subscore(close)
    comp = technical_components(close, hist["RSI"].iloc[-1] if "RSI" in getattr(hist, "columns", []) else None)
    m_label, m_color, m_note = moment_label(moment, True)
    tier = row["treapta"]
    rank, n_ranked = score_rank(cs["scor"], tk)
    vb = verdict_banner(tier, row["scor"], rank, n_ranked, moment)

    st.markdown(f"""
        <div style="background:#161B22; border:1px solid {vb['border']}88; border-radius:12px; padding:14px; margin:8px 0 14px 0;">
            <div style="display:flex; flex-wrap:wrap; gap:28px; align-items:baseline;">
                <div><span style="color:#8B949E; font-size:12px; text-transform:uppercase;">Fundamental</span><br>
                    <span style="color:{vb['fund_color']}; font-weight:bold; font-size:20px;">{html.escape(vb['fund_label'])}</span>
                    <span style="color:#C9D1D9; font-size:15px;">&nbsp;{html.escape(vb['fund_detail'])}</span></div>
                <div><span style="color:#8B949E; font-size:12px; text-transform:uppercase;">Moment</span><br>
                    <span style="color:{vb['moment_color']}; font-weight:bold; font-size:20px;">{html.escape(vb['moment_label'])}</span>
                    <span style="color:#C9D1D9; font-size:15px;">&nbsp;{html.escape(vb['moment_detail'])}</span></div>
            </div>
            <div style="color:#C9D1D9; font-size:13px; margin-top:8px;">{html.escape(vb['text'])}</div>
        </div>""", unsafe_allow_html=True)

    # Explicația pe înțeles: fiecare pas, cu punctele tari și slabe (texte construite din cifrele de mai jos)
    strong, middle, weak, notes = explain_fundamental(me, cs, tk)
    moment_lines = explain_moment(close, comp["raw"]["rsi"])

    def _li(items, color):
        return "".join(f'<li style="color:{color}; margin-bottom:3px;">{html.escape(x)}</li>' for x in items)

    e1, e2 = st.columns(2)
    with e1:
        st.markdown(compact_html(f"""
            <div style="background:#161B22; border:1px solid #30363D; border-radius:10px; padding:12px 14px; height:100%;">
                <div style="color:#C9D1D9; font-weight:bold; margin-bottom:6px;">Fundamental: compania stă {html.escape(vb['fund_label'].lower())} ({html.escape(vb['fund_detail'])})</div>
                <ul style="margin:0; padding-left:18px; font-size:13px;">
                    {_li(["Puncte tari: " + s for s in strong], "#3FB950")}
                    {_li(["La mijloc: " + s for s in middle], "#D29922")}
                    {_li(["Puncte slabe: " + s for s in weak], "#F85149")}
                    {_li(notes, "#8B949E")}
                </ul>
            </div>"""), unsafe_allow_html=True)
    with e2:
        st.markdown(compact_html(f"""
            <div style="background:#161B22; border:1px solid #30363D; border-radius:10px; padding:12px 14px; height:100%;">
                <div style="color:#C9D1D9; font-weight:bold; margin-bottom:6px;">Momentul: {html.escape(vb['moment_label'].lower())} ({html.escape(vb['moment_detail'])})</div>
                <ul style="margin:0; padding-left:18px; font-size:13px;">
                    {_li(moment_lines, "#C9D1D9")}
                </ul>
            </div>"""), unsafe_allow_html=True)
    st.write("")

    n_cmp = int(row["n_comparate"]) if pd.notna(row["n_comparate"]) else 0
    tier_txt = {2: "FAVORABIL", 1: "NEUTRU", 0: "NEFAVORABIL"}.get(int(tier), "") if pd.notna(tier) else "fără treaptă"
    pct = lambda k: (f"percentila {ro(row[k + '_pct'] * 100):.0f}" if pd.notna(row.get(k + "_pct")) else "N/A")

    st.markdown("**Pasul 1 · Fundamental** (relativ la celelalte companii BVB din fișierele tale)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Scor fundamental", "N/A" if pd.isna(row["scor"]) else f"{ro(row['scor']):.0f}/100",
              f"{tier_txt} · din {n_cmp} companii", delta_color="off", delta_arrow="off", help=BVB_HELP["scor"])
    c2.metric("P/E TTM", _bvb_fmt(me["PE"], "x"), pct("EP"), delta_color="off", delta_arrow="off", help=BVB_HELP["pe"])
    c3.metric("P/BV", _bvb_fmt(me["PBV"], "x"), pct("BP"), delta_color="off", delta_arrow="off", help=BVB_HELP["pbv"])
    c4.metric("Randament dividend", _bvb_fmt(me["DY"], "pct"), pct("DY"), delta_color="off", delta_arrow="off", help=BVB_HELP["dy"])
    c5.metric("Piotroski simplificat", "N/A" if pd.isna(me["PIO"]) else f"{ro(me['PIO']):.1f}/6", pct("PIO"),
              delta_color="off", delta_arrow="off", help=BVB_HELP["pio"])
    gn = me["GN"]
    gn_txt = "N/A" if pd.isna(gn) else ("≤ 22,5 ✓" if gn <= GN_LIMIT else "> 22,5")

    st.markdown("**Pasul 2 · Momentul** (tehnic, doar pentru intrare)")
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("Moment", "N/A" if moment is None else f"{score_ro(moment)}/100", m_label, delta_color="off", delta_arrow="off", help=BVB_HELP["moment"])
    trend = comp["trend"]
    t2.metric("Trend SMA 50/200", "N/A" if trend is None else f"{ro(trend * 3):.0f} din 3" if len(close.dropna()) >= 200 else f"{ro(trend):.0f} din 1",
              help=BVB_HELP["trend"])
    mom_raw = comp["raw"]["momentum_12_1"]
    t3.metric("Momentum 12-1 luni", _bvb_fmt(mom_raw, "pct"), help=BVB_HELP["momentum"])
    t4.metric("RSI 14", rsi_description(comp["raw"]["rsi"]), help=BVB_HELP["rsi"])
    t5.metric("GN (P/E × P/BV)", _bvb_fmt(gn, "x"), gn_txt, delta_color="off", delta_arrow="off", help=BVB_HELP["gn"])

    with st.expander("Informații fără putere de predicție în test (ROE, marjă, îndatorare)"):
        i1, i2, i3 = st.columns(3)
        i1.metric("ROE TTM", _bvb_fmt(me["ROE"], "pct"), help=BVB_HELP["roe"])
        i2.metric("Marjă netă TTM", _bvb_fmt(me["marja"], "pct"), help=BVB_HELP["marja"])
        i3.metric("Datorii totale / capital propriu", _bvb_fmt(me["indatorare"], "x"), help=BVB_HELP["indatorare"])

    with st.expander("Piotroski simplificat: criteriile"):
        if me["piotroski"]:
            rows_p = []
            for c in me["piotroski"]:
                rows_p.append({"Criteriu": c["criteriu"], "Valori (an trecut → acum)": piotroski_value_text(c),
                               "Rezultat": {True: "✓", False: "✗", None: "N/A"}[c["trece"]]})
            st.dataframe(pd.DataFrame(rows_p), hide_index=True, width='stretch')
            st.caption(f"Raportul {me['raport_flux']} față de același raport de acum un an (TTM), din fișierul tău. "
                       "N/A la emisiune = numărul de acțiuni a crescut de peste 1,65× (split sau acțiuni gratuite). "
                       "Diferă de cardul „Piotroski F-Score” de mai jos (din Yahoo): acela compară ultimele două exerciții "
                       "anuale, pe 9 criterii, și ia doar datoria financiară pe termen lung, nu toate datoriile pe termen lung.")
        else:
            st.caption("Fără raport comparabil de acum un an.")

    with st.expander(f"Clasamentul BVB ({n_cmp} companii din fișiere)"):
        table = pd.DataFrame({
            "Simbol": cs.index,
            "Scor": cs["scor"].round(0),
            "Treaptă": cs["treapta"].map({2.0: "FAVORABIL", 1.0: "NEUTRU", 0.0: "NEFAVORABIL"}).fillna("N/A"),
            "P/E": [indicators[t]["PE"] for t in cs.index],
            "P/BV": [indicators[t]["PBV"] for t in cs.index],
            "Dividend %": [indicators[t]["DY"] * 100 if pd.notna(indicators[t]["DY"]) else np.nan for t in cs.index],
            "Piotroski /6": cs["PIO"],
            "Ultimul raport": [indicators[t]["raport_bilant"] for t in cs.index],
        }).sort_values("Scor", ascending=False, na_position="last")
        st.dataframe(table.style.format({"Scor": ro_spec(".0f"), "P/E": ro_spec(",.2f"), "P/BV": ro_spec(",.2f"),
                                         "Dividend %": ro_spec(".2f"), "Piotroski /6": ro_spec(".1f")}),
                     hide_index=True, width='stretch')
        no_price = [t for t, p in prices.items() if p is None]
        if no_price:
            st.caption("Fără preț Yahoo acum (multiplii lor sunt N/A): " + ", ".join(sorted(no_price)) + ".")

    issues = data["probleme"].get(tk, [])
    with st.expander(f"Calitatea datelor din fișierul {data['fisiere'][tk]} ({len(issues)} semnalări)"):
        if issues:
            st.dataframe(pd.DataFrame([{
                "Perioadă": period_label(i["year"], i["months"]),
                "Câmp": COMPANY_FIELD_LABELS.get(i["field"], i["field"]),
                "Problemă": i["message"],
                "Efect": "exclus din calcul" if i["excluded"] else "doar semnalat"} for i in issues]),
                hide_index=True, width='stretch')
        else:
            st.caption("Nicio problemă găsită de verificările automate.")
        if data["erori"]:
            st.caption("Fișiere necitite: " + "; ".join(data["erori"]))

    for w in me["avertismente"]:
        st.warning(w)
    st.caption(f"Sursa: fișierul „{data['fisiere'][tk]}”, bilanț {me['raport_bilant']}, profit TTM la "
               f"{me['raport_flux'] or 'N/A'}, dividend {me['raport_anual'] or 'N/A'}; preț "
               f"{_bvb_fmt(price, 'x')} lei (Yahoo, întârziat). {TEST_NOTE}")
    _, _, history_text = final_verdict(tier, moment)
    return {"is_bvb": True, "fund_label": vb["fund_label"], "fund_color": vb["fund_color"],
            "fund_score": None if pd.isna(row["scor"]) else float(row["scor"]), "fund_detail": vb["fund_detail"],
            "fund_basis": "testat pe BVB", "moment": moment, "moment_label": vb["moment_label"],
            "moment_color": vb["moment_color"], "moment_detail": vb["moment_detail"],
            "moment_basis": "prag testat pe BVB", "pillars": [], "history_text": history_text, "financial": False,
            # Randamentul dividendului din fișier (dividend total pe ultimul an / capitalizarea la prețul curent),
            # pentru cardul „Dividend (Randament)”: Yahoo nu îl trimite la BVB.
            "dy": None if pd.isna(me["DY"]) else float(me["DY"]), "dy_year": me.get("raport_anual"), "price": price,
            # Creșterea EPS pe 3 ani din foaia anuală a fișierului, pentru PEG (istoric)
            "eps_growth": annual_eps_growth(data["companii"][tk]),
            # Pentru criteriile lui Daniel din „Verdict final”: CA și profit pe ultimii doi ani, datorii / active
            "rev_pair": annual_pair(data["companii"][tk], "rev"), "ni_pair": annual_pair(data["companii"][tk], "ni"),
            "debt_ratio": latest_ratio(data["companii"][tk], "liab", "assets")}
    st.markdown("---")


NON_COMPANY_QUOTE_TYPES = ("ETF", "MUTUALFUND", "INDEX", "CRYPTOCURRENCY", "CURRENCY", "FUTURE")


def global_fundamental_inputs(real_sym, info, hist, risk_stats):
    """Intrările scorului fundamental SUA/UE (analytics/global_fundamental.py), din situațiile
    financiare și din aceleași funcții ca secțiunile de mai jos din pagină (DCF, Altman, Piotroski).
    Ipoteze fixe, independente de cursoarele calculatorului: g terminal 2%, WACC calculat."""
    sym = str(real_sym).upper()
    fin = get_financial_statements(real_sym)
    t_curr = info.get('currency') or 'USD'
    price = num(info, 'currentPrice') or num(info, 'previousClose')
    if price is None and hist is not None and not hist.empty:
        price = positive_or_none(hist['Close'].iloc[-1])
    rf_val, rf_label = get_risk_free_for_currency(t_curr)
    beta = risk_stats.get("beta")
    financial = fund.is_financial_issuer(info.get('sector'), sym)
    inp = {"financial": financial, "price": price, "pbv": num(info, 'priceToBook'),
           "rf": rf_val, "rf_label": rf_label, "beta": beta, "beta_label": risk_stats.get("beta_label"),
           "statement_errors": fin.get("errors", [])}
    if financial:
        inp.update(roe=num(info, 'returnOnEquity'), ke=fund.cost_of_equity(rf_val, beta))
        return inp

    dcf_in = fund.dcf_inputs(fin.get("income"), fin.get("balance"), fin.get("cashflow"),
                             fin.get("q_cashflow"), fin.get("q_balance"))
    dcf_na = fund.dcf_not_applicable_reason(info.get('sector'), info.get('industry'), sym)
    fin_curr = info.get('financialCurrency')
    if dcf_na is None and not info.get('sector') and not fin_curr:
        dcf_na = "Yahoo nu a trimis sectorul și moneda de raportare: nu se poate exclude o bancă sau un ADR"
    if dcf_na is None and fin_curr and fin_curr != t_curr:
        dcf_na = f"situațiile sunt în {fin_curr}, acțiunea în {t_curr}; conversia valutară nu e implementată"
    wacc_res = fund.wacc(rf_val, beta, num(info, 'marketCap'), dcf_in["total_debt"],
                         dcf_in["interest_expense"], dcf_in["tax_rate"])
    fcf, fcf_basis = dcf_in["fcf"], dcf_in["fcf_basis"]
    if (fund.fcf_distortion_warning(dcf_in["cfo"], dcf_in["capex"], dcf_in["fcf"], dcf_in["fcf_average"])
            and dcf_in["fcf_average"] is not None):
        # FCF-ul curent e deformat (vârf de investiții, an atipic): DCF-ul invers pornește de la medie.
        fcf, fcf_basis = dcf_in["fcf_average"], f"media ultimilor {dcf_in['fcf_average_years']} ani fiscali"
    lev = fund.leverage_ratios(fin.get("income"), fin.get("balance"), fin.get("q_income"), fin.get("q_balance"))
    # Randamentul AAA pentru criteriul lui Graham pe randament (E/P ≥ 2 × AAA): același Y ca la Graham revizuit
    # (USD: FRED AAA; EUR: proxy Bund + marja AAA din SUA; alte valute: N/A).
    aaa, aaa_label = get_graham_yield(t_curr)
    inp.update(dcf_na=dcf_na, wacc=wacc_res["wacc"] if wacc_res else None, fcf=fcf,
               fcf_basis=fcf_basis.split(" (")[0] if fcf_basis else None,
               net_debt=dcf_in["net_debt"], shares=dcf_in["shares"], pe=num(info, 'trailingPE'),
               eps=num(info, 'trailingEps'), aaa=aaa, aaa_label=aaa_label,
               piotroski=fund.piotroski(fin.get("income"), fin.get("balance"), fin.get("cashflow")),
               altman=calculate_altman_z(info, sym, fin), ebitda=lev["ebitda"], coverage=lev["interest_coverage"])
    # Evaluarea din 14.11: creșterea cerută de preț față de cea livrată (cifra de afaceri); PEG pe creșterea EPS
    rev_g = fund.statement_cagr(fin.get("income"), fund.REVENUE_ROWS)
    eps_g = fund.statement_cagr(fin.get("income"), fund.EPS_ROWS)
    inp.update(rev_growth=(rev_g or {}).get("growth"), rev_growth_years=(rev_g or {}).get("years"),
               eps_growth=(eps_g or {}).get("growth"), eps_growth_years=(eps_g or {}).get("years"))
    return inp


def render_global_decision(real_sym, info, hist, risk_stats):
    """Secțiunea „Scor fundamental” pentru acțiunile din afara BVB: scorul companiei și, separat,
    momentul (fără verdict: testul pe SUA nu a găsit putere de predicție)."""
    sym = str(real_sym).upper()
    if sym.endswith(".RO") or (info.get('quoteType') or "").upper() in NON_COMPANY_QUOTE_TYPES:
        return
    st.subheader("🌍 Scor fundamental (descriptiv, fără putere de predicție în test)")
    inp = global_fundamental_inputs(real_sym, info, hist, risk_stats)
    res = gfund.fundamental_score(inp)
    close = hist["Close"] if hist is not None and not hist.empty else pd.Series(dtype=float)
    moment = technical_subscore(close)
    comp = technical_components(close, hist["RSI"].iloc[-1] if "RSI" in getattr(hist, "columns", []) else None)
    m_label, m_color, m_note = moment_label(moment, False)
    n_ok = sum(1 for p in res["pillars"] if p["score"] is not None)
    fund_detail = ("N/A" if res["score"] is None
                   else f"{score_ro(res['score'])}/100 · {n_ok} din {len(res['pillars'])} piloni cu date · descriptiv")
    moment_detail = "N/A" if moment is None else f"tehnic {score_ro(moment)}/100, descriptiv"

    st.markdown(f"""
        <div style="background:#161B22; border:1px solid {res['color']}88; border-radius:12px; padding:14px; margin:8px 0 14px 0;">
            <div style="display:flex; flex-wrap:wrap; gap:28px; align-items:baseline;">
                <div><span style="color:#8B949E; font-size:12px; text-transform:uppercase;">Fundamental</span><br>
                    <span style="color:{res['color']}; font-weight:bold; font-size:20px;">{html.escape(res['label'])}</span>
                    <span style="color:#C9D1D9; font-size:15px;">&nbsp;{html.escape(fund_detail)}</span></div>
                <div><span style="color:#8B949E; font-size:12px; text-transform:uppercase;">Moment</span><br>
                    <span style="color:{m_color}; font-weight:bold; font-size:20px;">{html.escape(m_label)}</span>
                    <span style="color:#C9D1D9; font-size:15px;">&nbsp;{html.escape(moment_detail)}</span></div>
            </div>
            <div style="color:#C9D1D9; font-size:13px; margin-top:8px;">{html.escape(res['note'])}</div>
        </div>""", unsafe_allow_html=True)

    strong, middle, weak, missing = gfund.explain(res)
    info_lines = gfund.info_notes(res)
    moment_lines = explain_moment(close, comp["raw"]["rsi"]) + [m_note]

    def _li(items, color):
        return "".join(f'<li style="color:{color}; margin-bottom:3px;">{html.escape(x)}</li>' for x in items)

    e1, e2 = st.columns(2)
    with e1:
        st.markdown(compact_html(f"""
            <div style="background:#161B22; border:1px solid #30363D; border-radius:10px; padding:12px 14px; height:100%;">
                <div style="color:#C9D1D9; font-weight:bold; margin-bottom:6px;">Fundamental: compania stă {html.escape(res['label'].lower())} ({html.escape(fund_detail)})</div>
                <ul style="margin:0; padding-left:18px; font-size:13px;">
                    {_li(["Puncte tari: " + s for s in strong], "#3FB950")}
                    {_li(["La mijloc: " + s for s in middle], "#D29922")}
                    {_li(["Puncte slabe: " + s for s in weak], "#F85149")}
                    {_li(["Fără date: " + s for s in missing], "#8B949E")}
                    {_li(["Informativ: " + s for s in info_lines], "#8B949E")}
                </ul>
            </div>"""), unsafe_allow_html=True)
    with e2:
        st.markdown(compact_html(f"""
            <div style="background:#161B22; border:1px solid #30363D; border-radius:10px; padding:12px 14px; height:100%;">
                <div style="color:#C9D1D9; font-weight:bold; margin-bottom:6px;">Momentul: {html.escape(m_label.lower())} ({html.escape(moment_detail)})</div>
                <ul style="margin:0; padding-left:18px; font-size:13px;">
                    {_li(moment_lines, "#C9D1D9")}
                </ul>
            </div>"""), unsafe_allow_html=True)
    st.write("")

    cols = st.columns(len(res["pillars"]))
    for col, p in zip(cols, res["pillars"]):
        col.metric(p["label"], "N/A" if p["score"] is None else f"{ro(p['score'] * 100):.0f}/100",
                   help=PILLAR_HELP[p["key"]] + PILLAR_TEST_NOTE)

    with st.expander("Componentele scorului fundamental"):
        st.dataframe(pd.DataFrame([{
            "Pilon": p["label"], "Componentă": c["label"], "Valoare": c["value"],
            "Notă": ("informativ" if c.get("info") else "N/A" if c["score"] is None else f"{ro(c['score'] * 100):.0f}/100"),
            "Explicație": c["text"]} for p in res["pillars"] for c in p["components"]]),
            hide_index=True, width='stretch')
        if res["financial"]:
            ke = inp.get("ke")
            st.caption("Bancă / asigurător / emitent financiar: DCF, Altman și Piotroski nu se aplică (datoria e materie primă, "
                       "fluxul de numerar reflectă depozitele și creditele). ke = rf + beta × 5% = "
                       + ("N/A" if ke is None else f"{ro(ke * 100):.1f}%") + f"; rf: {inp['rf_label']}; beta: {inp['beta_label']}.")
        else:
            st.caption("Ipoteze fixe ale scorului, independente de cursoarele calculatorului de mai jos: g terminal 2%, "
                       "WACC calculat, FCF TTM (sau media anilor fiscali când FCF-ul curent e deformat). "
                       f"rf: {inp['rf_label']}; beta: {inp['beta_label']}. Detaliile (DCF cu ipotezele tale, Altman, "
                       "Piotroski) sunt în secțiunile de mai jos.")
        if inp.get("statement_errors"):
            st.caption("Situații financiare necitite de la Yahoo: " + ", ".join(inp["statement_errors"]) + ".")
    st.caption("„Verdict final” de mai jos reia acest scor și momentul, fără să le recalculeze.")
    st.markdown("---")
    return {"is_bvb": False, "fund_label": res["label"], "fund_color": res["color"], "fund_score": res["score"],
            "fund_detail": fund_detail, "fund_basis": "descriptiv, netestat", "moment": moment,
            "moment_label": m_label, "moment_color": m_color, "moment_detail": moment_detail,
            "moment_basis": "descriptiv, fără verdict",
            "pillars": [(p["label"], None if p["score"] is None else p["score"] * 100) for p in res["pillars"]],
            "history_text": None, "financial": bool(res["financial"])}


def _read_peer_rows(symbols):
    """Citește din Yahoo indicatorii unor comparabili: (rânduri, simboluri fără date).
    Valorile lipsă rămân None (afișate N/A)."""
    rows, failed = [], []
    for p_sym in symbols:
        _yf_limiter.wait_if_needed()
        try:
            inf = yf.Ticker(p_sym).info or {}
        except Exception as e:
            logger.warning(f"comparabil {p_sym} indisponibil: {e}")
            failed.append(p_sym)
            continue
        row = {"Simbol": p_sym, "Capitalizare": num(inf, 'marketCap'), "Monedă": inf.get('currency') or "",
               "Industrie": inf.get('industry') or None}
        for key, label, mult in PEER_METRICS:
            value = num(inf, key)
            row[label] = None if value is None else value * mult
        if all(row[label] is None for _, label, _ in PEER_METRICS):
            failed.append(p_sym)       # simbol delistat, redenumit sau refuzat de Yahoo
            continue
        rows.append(row)
    return rows, failed

@st.cache_data(ttl=21600, show_spinner=False)
def get_peer_rows(region, sector):
    """Indicatorii comparabililor dintr-o regiune și un sector (listele din analytics/peers.py).

    Întoarce (rânduri, simboluri fără date). În cache 6 ore, pe (regiune, sector): fiecare
    comparabil costă o cerere la endpoint-ul Yahoo cel mai limitat, iar înainte tabelul era
    recitit la fiecare interacțiune cu pagina.
    """
    return _read_peer_rows(peer_symbols(region, sector))

@st.cache_data(ttl=21600, show_spinner=False)
def get_bvb_yahoo_peer_rows(yahoo_symbols):
    """Comparabilii BVB care nu au indicatori în foaia BVB (tuplu de simboluri Yahoo, ex. „BRD.RO"),
    citiți din Yahoo. Datele Yahoo pentru BVB sunt mai rare decât cele din foaie; sursa se afișează."""
    return _read_peer_rows(list(yahoo_symbols))

def calculate_atr_trailing_stop(df, window=14, multiplier=2.5):
    """
    Calculează pragul de Stop-Loss bazat pe volatilitatea istorică (ATR).
    Un multiplicator de 2.5 este standardul pentru investitori 'swing'.
    """
    # Logica este în analytics/technical.py (testată în tests/test_technical.py):
    # ATR cu netezire Wilder; stopul urcă odată cu prețul și se resetează când
    # închiderea ajunge la sau sub el. Întoarce o copie cu coloanele
    # ATR, ATR_Stop și ATR_Stop_Hit, sau None dacă istoricul e prea scurt.
    return atr_trailing_stop(df, window=window, multiplier=multiplier)

# --- FUNCȚII ȘTIRI ---
@st.cache_data(ttl=600, show_spinner=False)
def fetch_news_data():
    all_news = []
    for url in RSS_CONFIG["Feeds"]:
        try:
            feed = feedparser.parse(url)
            if not feed.entries: continue
            for entry in feed.entries[:15]:
                dt = parse_date(entry)
                all_news.append({
                    "title": entry.title,
                    "link": entry.link,
                    "summary": getattr(entry, "summary", ""),
                    "source": feed.feed.get("title", "Sursă Externă"),
                    "date_obj": dt,
                    "date_str": dt.strftime("%Y-%m-%d %H:%M")
                })
        except Exception as exc:  # înainte: except: gol
            logger.debug("fetch_news_data: %s", exc)
            continue
    # Cea mai nouă prima; știrile cu data din sursă în viitor (Ziarul Bursa: data ediției) marcate și puse la coadă.
    return mark_future(all_news, now_ro().replace(tzinfo=None))

def filter_news(all_news, category):
    keywords = RSS_CONFIG["Categorii"].get(category, [])
    
    # --- LOGICA NOUĂ PENTRU GENERAL ---
    # Tab-ul General va arăta acum TOATE știrile, ordonate cronologic.
    # Este mai util ca punct de plecare (Landing Page).
    if category == "General":
        return all_news

    filtered = []
    for item in all_news:
        text_full = (item['title'] + " " + item['summary']).lower()
        
        match_found = False
        for k in keywords:
            k = k.lower()
            # Dacă cuvântul cheie este scurt (sub 4 litere), căutăm doar cuvânt întreg
            # pentru a evita potriviri greșite (ex: "it" în "venit")
            if len(k) <= 3:
                pattern = rf"\b{re.escape(k)}\b"
                if re.search(pattern, text_full):
                    match_found = True
                    break
            else:
                # Pentru cuvinte lungi, căutăm doar rădăcina (ex: "imobilia" prinde și "imobiliar")
                if k in text_full:
                    match_found = True
                    break
        
        if match_found:
            filtered.append(item)
            
    return filtered

class _NoRssNews(Exception):
    """Fluxul RSS nu a dat nicio știre. Excepțiile nu intră în cache: o listă goală (Yahoo limitează uneori) se reîncearcă."""


@st.cache_data(ttl=900, show_spinner=False)
def _yahoo_rss_news(symbol):
    """Primele 30 de știri din fluxul Yahoo RSS al simbolului, nefiltrate (cache 15 min). Înainte fluxul se descărca
    la fiecare rerulare a paginii (orice clic pe un widget). Flux gol → _NoRssNews; eroare → excepția (fără cache)."""
    rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
    feed = feedparser.parse(rss_url)
    if not feed.entries:
        raise _NoRssNews(symbol)
    news_list = []
    for entry in feed.entries[:30]:
        dt = parse_date(entry)
        news_list.append({
            "title": entry.title,
            "link": entry.link,
            "summary": getattr(entry, "summary", ""),
            "publisher": "Yahoo Finance",
            "date_str": dt.strftime("%Y-%m-%d %H:%M")
        })
    return news_list


def get_company_news_rss(symbol, names=None, limit=7):
    """Știrile Yahoo (RSS) pe simbol. Cu `names` (numele companiei din `info`), se păstrează doar
    știrile care menționează emitentul (data/news.py): fluxul pe simbol aduce des știri despre alte
    companii. Fără `names` (indici: ^GSPC, ^IXIC) lista rămâne nefiltrată."""
    try:
        news_list = [dict(n) for n in _yahoo_rss_news(symbol)]   # copie: lista din cache nu se modifică
    except _NoRssNews:
        logger.info("get_company_news_rss %s: flux fără știri", symbol)
        return []
    except Exception as exc:  # rețea / format RSS
        logger.warning("get_company_news_rss: %s", exc)
        return []
    if names is not None:
        news_list = filter_company_news(news_list, symbol, names)
    return news_list[:limit]

# --- FUNCȚII ANALIZĂ (Professional Update) ---
@st.cache_data(ttl=3600)
def get_macro_data_visuals():
    tickers = {
        # --- Indicatori Macro (Dobânzi, Valute, Mărfuri) ---
        'US 10Y Yield 🇺🇸': '^TNX', 
        'Dolar Index 💲': 'DX-Y.NYB', 
        'Petrol WTI 🛢️': 'CL=F', 
        'Aur 🥇': 'GC=F',
        'Argint 🥇': 'SI=F',
        'Copper': 'HG=F',
        'EUR/USD 🇪🇺': 'EURUSD=X',
        'EUR/RON 🇪🇺': 'EURRON=X',
        'USD/RON 🇺🇸': 'USDRON=X',
        
        # --- Indici Bursieri Majori (NOU) ---
        'Bursa RO (BET) 🇷🇴': 'TVBETETF.RO',
        'S&P 500 (US) 🇺🇸': '^GSPC',
        'Nasdaq 100 (Tech) 💻': '^NDX',
        'Dow Jones 30 🏭': '^DJI',
        'DAX 40 (Germania) 🇩🇪': '^GDAXI'
    }
    # Descărcăm 5 ani (5y)
    data = yf.download(list(tickers.values()), period="5y", group_by='ticker', progress=False, auto_adjust=True)
    return tickers, data

@st.cache_data(ttl=3600, show_spinner=False)
def get_financial_statements(symbol):
    """Situațiile financiare anuale și trimestriale din yfinance, pentru analytics/fundamentals.py.

    Întoarce un dict cu cheile income, balance, cashflow, q_income, q_balance, q_cashflow
    (DataFrame sau None) și `errors`: lista situațiilor care nu au putut fi citite.
    Nu afișează nimic: interfața decide ce mesaj arată.
    """
    out = {"errors": []}
    t = yf.Ticker(symbol)
    for key, attr in (("income", "financials"), ("balance", "balance_sheet"), ("cashflow", "cashflow"),
                      ("q_income", "quarterly_financials"), ("q_balance", "quarterly_balance_sheet"),
                      ("q_cashflow", "quarterly_cashflow")):
        df = None
        try:
            raw = getattr(t, attr)
            if isinstance(raw, pd.DataFrame) and not raw.empty:
                df = raw
        except Exception as e:
            logger.warning(f"{attr} indisponibil pentru {symbol}: {e}")
        out[key] = df
        if df is None:
            out["errors"].append(attr)
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def get_benchmark_close(symbol):
    """Închiderile ajustate pe 5 ani ale unui benchmark (pentru beta). Series goală la eșec."""
    try:
        data = yf.Ticker(symbol).history(period="5y", auto_adjust=True)
        if data is not None and not data.empty and 'Close' in data:
            return data['Close'].dropna()
    except Exception as e:
        logger.warning(f"benchmark {symbol} indisponibil: {e}")
    return pd.Series(dtype=float)


STATEMENT_RATIO_LABELS = {
    "trailingPE": "P/E", "priceToBook": "P/BV", "trailingEps": "EPS", "bookValue": "valoare contabilă/acțiune",
    "returnOnEquity": "ROE", "returnOnAssets": "ROA", "profitMargins": "marjă netă",
    "operatingMargins": "marjă operațională", "debtToEquity": "datorii/capital", "currentRatio": "current ratio",
    "quickRatio": "quick ratio", "totalRevenue": "venituri", "netIncomeToCommon": "profit net",
    "operatingCashflow": "flux din exploatare", "totalDebt": "datorie totală", "totalCash": "numerar",
}


class SheetUnavailable(Exception):
    """Foaia cerută nu există sau nu e accesibilă (mesaj fără date din foaie)."""


@st.cache_data(ttl=900, show_spinner=False)
def load_sheet_values(sheet_name):
    """`get_all_values()` al unei foi din `portofoliu_db`, pentru tabul „Import Date”. Doar citire.
    Eroarea aruncă excepția, deci nu intră în cache."""
    ws = connect_to_gsheets(sheet_name)
    if not ws:
        raise SheetUnavailable(f"foaia {sheet_name} nu există sau nu e accesibilă")
    return ws.get_all_values()


def read_sheet_values(sheet_name):
    """(valori, None) sau (None, motivul) — pentru afișare cu st.error, fără excepție brută."""
    try:
        return load_sheet_values(sheet_name), None
    except SheetUnavailable as exc:
        return None, str(exc)
    except (gspread.exceptions.GSpreadException, requests.RequestException) as exc:
        logger.warning("citire foaie %s: %s", sheet_name, exc)
        return None, f"citirea foii {sheet_name} a eșuat ({type(exc).__name__})"


@st.cache_data(ttl=900, show_spinner=False)
def load_bvb_fundamentals():
    """Indicatorii din foaia `BVB` a `portofoliu_db`, pe simbol (vezi data/bvb_sheet.py).
    Dict gol dacă foaia nu poate fi citită. Doar citire: foaia nu e modificată niciodată de aici.
    Cheia `_unmapped` (nu e simbol) ține indicatorii mapați care nu au fost găsiți în foaie,
    de regulă după o redenumire de rând."""
    try:
        ws = connect_to_gsheets("BVB")
        if not ws:
            return {}
        values = ws.get_all_values()
        parsed = parse_bvb_sheet(values)
        parsed["_unmapped"] = unmapped_info_keys(values)
        return parsed
    except Exception as e:
        logger.warning(f"foaia BVB nu a putut fi citită: {e}")
        return {}


def apply_bvb_sheet(info, symbol, hist=None):
    """Pentru simbolurile .RO prezente în foaia `BVB`, indicatorii din foaie înlocuiesc valorile
    Yahoo (rare și nesigure la BVB). Ce a fost preluat ajunge în `info['_from_bvb_sheet']`.
    P/E și P/BV se recalculează la prețul curent din EPS-ul și multiplii din foaie; cheile
    recalculate ajung în `info['_bvb_repriced']`."""
    info["_from_bvb_sheet"], info["_bvb_period"], info["_bvb_indicators"] = [], None, []
    info["_bvb_repriced"] = []
    sheet_symbol = bvb_symbol(symbol)
    if not sheet_symbol:
        return info
    entry = load_bvb_fundamentals().get(sheet_symbol)
    if not entry:
        return info
    for key, value in entry["info"].items():
        info[key] = value
        info["_from_bvb_sheet"].append(key)
    info["_bvb_period"], info["_bvb_indicators"] = entry["period"], entry["indicators"]
    price = num(info, 'currentPrice') or num(info, 'previousClose')
    if price is None and hist is not None and not hist.empty and pd.notna(hist['Close'].iloc[-1]):
        price = float(hist['Close'].iloc[-1])
    for key, value in reprice_bvb(entry["info"], price).items():
        info[key] = value
        info["_bvb_repriced"].append(key)
    if info["_from_bvb_sheet"]:
        info["_fundamentals_available"] = True
    return info


def enrich_info_from_statements(info, symbol, hist):
    """Completează în `info` indicatorii pe care Yahoo nu i-a trimis, calculați din situațiile
    financiare (analytics/fundamentals.py). Valorile primite de la Yahoo nu sunt suprascrise.

    Lista celor completați ajunge în `info['_from_statements']`, ca interfața să spună ce e
    calculat aici. P/E, P/BV, EPS și valoarea contabilă pe acțiune se calculează doar când
    moneda situațiilor e cunoscută și egală cu cea de tranzacționare (sau la BVB, unde e RON).
    """
    info["_from_statements"] = []
    if all(num(info, key) is not None for key in STATEMENT_RATIO_LABELS):
        return info
    fin = get_financial_statements(symbol)
    price = num(info, 'currentPrice') or num(info, 'previousClose')
    if price is None and hist is not None and not hist.empty and pd.notna(hist['Close'].iloc[-1]):
        price = float(hist['Close'].iloc[-1])
    fin_curr, trade_curr = info.get('financialCurrency'), info.get('currency')
    per_share_ok = (fin_curr == trade_curr) if (fin_curr and trade_curr) else str(symbol).upper().endswith(".RO")
    ratios = fund.ratios_from_statements(fin.get("income"), fin.get("balance"), fin.get("cashflow"),
                                         fin.get("q_income"), fin.get("q_balance"), fin.get("q_cashflow"),
                                         price=price, per_share_ok=per_share_ok)
    for key, value in ratios.items():
        if value is not None and num(info, key) is None:
            info[key] = value
            info["_from_statements"].append(key)
    return info


def resolve_beta_alpha(symbol, info, hist):
    """Beta și alpha ale unei acțiuni, cu sursa fiecăruia. Un singur loc: DCF, audit și scoruri
    folosesc aceleași valori.

    Beta: la BVB cel din Yahoo e calculat față de un indice nepotrivit (iese mult prea mic),
    deci se calculează față de BET (prin TVBETETF.RO, proxy), pe randamente săptămânale.
    La celelalte piețe rămâne beta Yahoo; dacă lipsește, se calculează față de indicele pieței.
    Alpha: când beta e calculat aici, alpha folosește același benchmark și rata fără risc a
    valutei. Cu beta din Yahoo: acțiunile în USD se compară cu SPY (fereastră calendaristică de un an,
    rata fără risc din ^TNX, fără valoare de rezervă); cele în EUR
    se compară cu DAX (.DE) sau Euro Stoxx 50, cu beta calculat față de același indice; alte valute: N/A.
    Întoarce un dict: beta, beta_label, alpha, alpha_label. Valorile lipsă sunt None.
    """
    currency = info.get('currency') or 'USD'
    yahoo_beta = num(info, 'beta')
    out = {"beta": yahoo_beta, "beta_label": "Yahoo" if yahoo_beta is not None else "indisponibil",
           "alpha": None, "alpha_label": "indisponibil"}
    is_bvb = str(symbol).upper().endswith(".RO")

    if is_bvb or yahoo_beta is None:
        bench_sym, bench_name = beta_benchmark(symbol, currency)
        bench_close = get_benchmark_close(bench_sym) if bench_sym else pd.Series(dtype=float)
        own = beta_weekly(hist['Close'], bench_close) if bench_sym else None
        # Un beta ≤ 0 ar da un cost al capitalului sub rata fără risc: se respinge.
        if own is not None and own["beta"] > 0:
            out["beta"] = own["beta"]
            out["beta_label"] = (f"calculat față de {bench_name}, {own['n']} randamente săptămânale, "
                                 f"{own['start']:%m.%Y}–{own['end']:%m.%Y}")
            rf_val, rf_label = get_risk_free_for_currency(currency)
            alpha = jensen_alpha(hist['Close'], bench_close, own["beta"], rf_val)
            if alpha is not None:
                out["alpha"] = alpha["alpha"]
                out["alpha_label"] = (f"față de {bench_name}, {alpha['start']:%d.%m.%Y}–{alpha['end']:%d.%m.%Y}; "
                                      f"rată fără risc {ro(rf_val * 100):.2f}% [{rf_label}]")
            elif rf_val is None:
                out["alpha_label"] = f"indisponibil: {rf_label}"
            else:
                out["alpha_label"] = "indisponibil: sub un an de ședințe comune cu benchmarkul"
            return out
        if is_bvb:
            # Beta Yahoo pentru BVB e nesigur: fără calcul propriu, mai bine N/A decât o cifră greșită.
            out["beta"], out["beta_label"] = None, f"indisponibil: date insuficiente pentru calculul față de {bench_name}"
            out["alpha_label"] = "indisponibil: lipsește beta"
            return out

    if out["beta"] is None:
        out["alpha_label"] = "indisponibil: lipsește beta"
        return out
    cur = currency.upper()
    if cur == "EUR":
        # Zona euro: alpha față de indicele pieței (DAX pentru .DE, Euro Stoxx 50 în rest), în EUR, cu rata
        # fără risc în EUR. Jensen cere beta față de ACELAȘI indice, deci se calculează aici (săptămânal);
        # beta Yahoo rămâne cel folosit în DCF și WACC.
        bench_sym, bench_name = beta_benchmark(symbol, cur)
        if bench_sym == "^STOXX50E":
            bench_name += " (indice de preț, fără dividende)"
        bench_close = get_benchmark_close(bench_sym)
        own = beta_weekly(hist['Close'], bench_close)
        rf_val, rf_label = get_risk_free_for_currency(cur)
        if own is None or own["beta"] <= 0:
            out["alpha_label"] = f"indisponibil: beta față de {bench_name} nu se poate calcula (sub 52 de săptămâni comune)"
        elif rf_val is None:
            out["alpha_label"] = f"indisponibil: {rf_label}"
        else:
            alpha = jensen_alpha(hist['Close'], bench_close, own["beta"], rf_val)
            if alpha is None:
                out["alpha_label"] = f"indisponibil: sub un an de ședințe comune cu {bench_name}"
            else:
                out["alpha"] = alpha["alpha"]
                out["alpha_label"] = (f"față de {bench_name}, {alpha['start']:%d.%m.%Y}–{alpha['end']:%d.%m.%Y}; "
                                      f"beta {ro(own['beta']):.2f} față de același indice ({own['n']} randamente săptămânale; "
                                      f"beta Yahoo din DCF rămâne separat); rată fără risc {ro(rf_val * 100):.2f}% [{rf_label}]")
        return out
    if cur != "USD":
        out["alpha_label"] = f"indisponibil: nu există încă un indice de referință în {cur} (doar SUA, zona euro, BVB)"
        return out
    # USD: beta Yahoo (față de S&P 500), benchmark SPY. Fereastra pe date calendaristice (jensen_alpha),
    # nu pe număr de rânduri; rata fără risc citită din ^TNX, fără valoarea de rezervă de 4% (lipsă → N/A).
    rf_val, rf_label = get_risk_free_for_currency("USD")
    if rf_val is None:
        out["alpha_label"] = f"indisponibil: {rf_label}"
        return out
    alpha = jensen_alpha(hist['Close'], get_benchmark_close("SPY"), out["beta"], rf_val)
    if alpha is None:
        out["alpha_label"] = "indisponibil: sub un an de ședințe comune cu S&P 500 (SPY)"
        return out
    out["alpha"] = alpha["alpha"]
    out["alpha_label"] = (f"față de S&P 500 (SPY), {alpha['start']:%d.%m.%Y}–{alpha['end']:%d.%m.%Y}; "
                          f"beta Yahoo {ro(out['beta']):.2f}; rată fără risc {ro(rf_val * 100):.2f}% [{rf_label}]")
    return out


@st.cache_data(ttl=86400, show_spinner=False)
def get_aaa_yield():
    """Randamentul obligațiunilor corporative AAA din SUA (Moody's, FRED `AAA`, lunar), în procente:
    (valoare, descriere) sau (None, motiv). Fără valoare de rezervă."""
    try:
        end = datetime.today()
        serie = web.DataReader('AAA', 'fred', end - timedelta(days=200), end).iloc[:, 0].dropna()
        if len(serie) and 0 < float(serie.iloc[-1]) < 25:
            return float(serie.iloc[-1]), f"obligațiuni corporative AAA SUA (FRED AAA, {serie.index[-1]:%m.%Y})"
    except Exception as e:
        logger.warning(f"randament AAA (FRED) indisponibil: {e}")
    return None, "Randamentul obligațiunilor AAA (FRED) nu a putut fi citit."


@st.cache_data(ttl=86400, show_spinner=False)
def get_aaa_spread():
    """Marja AAA față de titlurile SUA pe 10 ani, pe aceeași lună (FRED `AAA` și `GS10`, ambele lunare):
    (dict din fund.aaa_treasury_spread, descriere) sau (None, motiv). Fără valoare de rezervă."""
    try:
        end = datetime.today()
        start = end - timedelta(days=200)
        aaa = web.DataReader('AAA', 'fred', start, end).iloc[:, 0]
        gs10 = web.DataReader('GS10', 'fred', start, end).iloc[:, 0]
        res = fund.aaa_treasury_spread(aaa, gs10)
        if res is not None and 0 < res["aaa"] < 25 and 0 < res["treasury"] < 25:
            return res, f"marja AAA din SUA {ro(res['spread']):.2f} pp (FRED AAA − GS10, {res['month'].strftime('%m.%Y')})"
        logger.warning("marja AAA (FRED AAA/GS10): nicio lună comună validă")
    except Exception as e:
        logger.warning(f"marja AAA (FRED AAA/GS10) indisponibilă: {e}")
    return None, "Marja AAA față de titlurile SUA (FRED AAA, GS10) nu a putut fi citită."


def get_graham_yield(currency):
    """Y din formula revizuită a lui Graham, în procente, pentru valuta acțiunii.

    USD: randamentul AAA din FRED. Alte valute: nu există o serie AAA locală, deci se
    aproximează ca rată fără risc a valutei + marja AAA față de titlurile SUA pe 10 ani,
    calculată pe aceeași lună (FRED AAA − GS10); eticheta spune că e proxy.
    (None, motiv) dacă lipsește oricare componentă.
    """
    cur = (currency or "").upper()
    if cur == "USD":
        return get_aaa_yield()
    spread, spread_label = get_aaa_spread()
    if spread is None:
        return None, spread_label
    rf_local, rf_label = get_risk_free_for_currency(cur)
    if rf_local is None:
        return None, f"Nu pot aproxima randamentul AAA în {cur or 'valuta necunoscută'} (lipsește rata fără risc: {rf_label})"
    return rf_local * 100 + spread["spread"], f"proxy: {rf_label} + {spread_label}"


ECB_EU_CURVE_URL = "https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SRS_10Y_3M"
ECB_RO_10Y_URL = "https://data-api.ecb.europa.eu/service/data/IRS/M.RO.L.L40.CI.0000.RON.N.Z"
EUROSTAT_MONEY_MARKET_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/irt_st_m"


class CurveUnavailable(Exception):
    """Curba nu a putut fi citită. Excepțiile nu intră în cache-ul Streamlit: un eșec temporar se reîncearcă
    la următoarea afișare, nu rămâne 6 ore (aceeași regulă ca rata fără risc, #75)."""


@st.cache_data(ttl=21600, show_spinner=False)
def _market_curve_cached(market):
    """Panta curbei randamentelor (10 ani − 3 luni, în puncte procentuale) pe piața acțiunii:
    (spread, descriere) sau (None, motiv). Fără valoare de rezervă.

    US: ^TNX − ^IRX (Yahoo), aceeași zi; ^IRX e randament de discount și se convertește întâi în randament echivalent
    de obligațiune (`tbill_discount_to_bey`), convenția lui ^TNX și a FRED DGS3MO. EU: curba titlurilor de stat AAA din zona euro (BCE, zilnic,
    seria gata calculată 10 ani − 3 luni; proxy pentru Bund). RO: titluri de stat 10 ani (BCE, lunar)
    − ROBOR 3M (Eurostat irt_st_m, lunar), pe aceeași lună.
    """
    def _valid(x):
        return x is not None and x == x and -10 < x < 10

    if market == "US":
        try:
            t10 = yf.Ticker("^TNX").history(period="10d", auto_adjust=False)['Close']
            t3 = yf.Ticker("^IRX").history(period="10d", auto_adjust=False)['Close']
            # ^IRX e cotat ca randament de discount (4,00 ≈ 4,10 echivalent): fără conversie panta ieșea ~0,1 pp mai mare.
            res = same_day_spread(t10, tbill_discount_to_bey(t3))
            if res is not None and _valid(res["spread"]):
                return res["spread"], (f"SUA, titluri de stat 10 ani {ro(res['long']):.2f}% − 3 luni {ro(res['short']):.2f}% "
                                       f"(^TNX − ^IRX convertit în randament echivalent, {res['date']:%d.%m.%Y})")
        except Exception as e:
            logger.warning(f"curba SUA (^TNX/^IRX) indisponibilă: {e}")
        raise CurveUnavailable("curba SUA (^TNX − ^IRX) nu a putut fi citită")
    if market == "EU":
        try:
            resp = requests.get(ECB_EU_CURVE_URL, params={"lastNObservations": 5, "format": "csvdata"}, timeout=10)
            if resp.status_code == 200:
                obs = parse_ecb_csv(resp.text)
                if obs is not None and _valid(obs[1]):
                    return obs[1], f"zona euro, curba titlurilor de stat AAA 10 ani − 3 luni (BCE, {obs[0]}; proxy pentru Bund)"
            else:
                logger.warning(f"BCE (curba zonei euro): HTTP {resp.status_code}")
        except requests.RequestException as e:
            logger.warning(f"BCE (curba zonei euro) indisponibilă: {e}")
        raise CurveUnavailable("curba zonei euro (BCE) nu a putut fi citită")
    if market == "RO":
        try:
            r10 = requests.get(ECB_RO_10Y_URL, params={"lastNObservations": 6, "format": "csvdata"}, timeout=10)
            r3 = requests.get(EUROSTAT_MONEY_MARKET_URL,
                              params={"geo": "RO", "int_rt": "IRT_M3", "freq": "M", "lastTimePeriod": 6}, timeout=10)
            if r10.status_code == 200 and r3.status_code == 200:
                res = monthly_spread(parse_ecb_series(r10.text), parse_eurostat_monthly(r3.json()))
                if res is not None and _valid(res["spread"]):
                    return res["spread"], (f"România, titluri de stat 10 ani {ro(res['long']):.2f}% (BCE) − ROBOR 3M "
                                           f"{ro(res['short']):.2f}% (Eurostat), {res['month']}")
                logger.warning("curba RO: nicio lună comună între BCE și Eurostat")
            else:
                logger.warning(f"curba RO: BCE HTTP {r10.status_code}, Eurostat HTTP {r3.status_code}")
        except (requests.RequestException, ValueError) as e:
            logger.warning(f"curba RO indisponibilă: {e}")
        raise CurveUnavailable("curba României (BCE 10 ani, ROBOR 3M Eurostat) nu a putut fi citită")
    raise CurveUnavailable("nu există încă o curbă a randamentelor pentru piața acestei acțiuni (doar SUA, zona euro, România)")


def get_market_curve(market):
    """(spread, descriere) sau (None, motiv). Doar citirile reușite stau în cache 6 ore (`_market_curve_cached`)."""
    try:
        return _market_curve_cached(market)
    except CurveUnavailable as exc:
        return None, str(exc)


class RiskFreeUnavailable(Exception):
    """Rata fără risc nu a putut fi citită. Excepțiile nu intră în cache-ul Streamlit, deci un eșec
    temporar (timeout BCE/FRED/Yahoo) se reîncearcă la următoarea afișare, nu rămâne 6 ore."""


@st.cache_data(ttl=21600, show_spinner=False)
def _risk_free_cached(currency):
    """Rata fără risc pe 10 ani în valuta dată: (fracție, descrierea sursei) sau (None, motiv).

    USD: randamentul titlurilor SUA pe 10 ani (^TNX, Yahoo). EUR: Bund 10 ani (FRED, lunar).
    RON: randamentul titlurilor de stat românești pe 10 ani, seria BCE de convergență
    (Data Portal, lunar, fără cheie). Pentru celelalte valute nu există sursă automată:
    întoarce None, iar interfața cere o rată de scont manuală. Nu există valoare de
    rezervă: o rată presupusă ar arăta ca una citită din piață.
    """
    def _valid(value):
        # ^TNX și seria FRED sunt în procente (4,25 = 4,25%). Orice în afara 0–25% e eroare de date.
        return value is not None and value == value and 0 < value < 25

    cur = (currency or "").upper()
    if cur == "USD":
        try:
            closes = yf.Ticker("^TNX").history(period="5d", auto_adjust=True)['Close'].dropna()
            if len(closes) and _valid(float(closes.iloc[-1])):
                return float(closes.iloc[-1]) / 100, f"titluri SUA 10 ani (^TNX, {closes.index[-1]:%d.%m.%Y})"
        except Exception as e:
            logger.warning(f"^TNX indisponibil: {e}")
        raise RiskFreeUnavailable("Randamentul titlurilor SUA pe 10 ani (^TNX) nu a putut fi citit.")
    if cur == "EUR":
        try:
            end = datetime.today()
            serie = web.DataReader('IRLTLT01DEM156N', 'fred', end - timedelta(days=200), end).iloc[:, 0].dropna()
            if len(serie) and _valid(float(serie.iloc[-1])):
                return float(serie.iloc[-1]) / 100, f"Bund 10 ani (FRED IRLTLT01DEM156N, {serie.index[-1]:%m.%Y})"
        except Exception as e:
            logger.warning(f"Bund 10 ani (FRED) indisponibil: {e}")
        raise RiskFreeUnavailable("Randamentul Bund pe 10 ani (FRED) nu a putut fi citit.")
    if cur == "RON":
        try:
            resp = requests.get(
                "https://data-api.ecb.europa.eu/service/data/IRS/M.RO.L.L40.CI.0000.RON.N.Z",
                params={"lastNObservations": 3, "format": "csvdata"}, timeout=10,
            )
            if resp.status_code == 200:
                obs = parse_ecb_csv(resp.text)
                if obs is not None and _valid(obs[1]):
                    return obs[1] / 100, f"titluri de stat RO 10 ani (BCE, {obs[0]})"
            else:
                logger.warning(f"BCE (randament RO 10 ani): HTTP {resp.status_code}")
        except requests.RequestException as e:
            logger.warning(f"BCE (randament RO 10 ani) indisponibil: {e}")
        raise RiskFreeUnavailable("Randamentul titlurilor de stat românești pe 10 ani (BCE) nu a putut fi citit.")
    raise RiskFreeUnavailable(f"Nu există încă o sursă automată pentru rata fără risc în {cur or 'valuta necunoscută'}.")


def get_risk_free_for_currency(currency):
    """Rata fără risc pe 10 ani în valuta dată: (fracție, descrierea sursei) sau (None, motiv).
    Doar valorile citite cu succes stau în cache (6 ore); vezi `_risk_free_cached`."""
    try:
        return _risk_free_cached(currency)
    except RiskFreeUnavailable as exc:
        return None, str(exc)

# --- Funcții ajutătoare ---

def calculate_health_score_ext(info):
    score = 5
    pros = []
    cons = []
    
    try:
        # 1. Analiză Datorii
        # Pragurile se raportează la limita sectorului (150% implicit, 400% la financiare),
        # ca băncile să nu fie penalizate pentru un levier normal în industria lor.
        de = num(info, 'debtToEquity')
        de_limit = get_sector_benchmarks(info.get('sector'))['de_max']
        if de:
            if de < de_limit / 3: 
                score += 2
                pros.append("Datorii foarte mici")
            elif de > de_limit * 2: 
                score -= 3
                cons.append("Risc mare de insolvență")
            elif de > de_limit: 
                score -= 2
                cons.append("Îndatorare ridicată")

        # 2. Analiză Rentabilitate (ROE)
        # Un indicator lipsă nu aduce și nu scade puncte.
        roe = num(info, 'returnOnEquity')
        if roe is not None:
            if roe > 0.15: 
                score += 2
                pros.append("Profitabilitate excelentă (ROE)")
            elif roe < 0.05: 
                score -= 1
                cons.append("Eficiență scăzută a capitalului")

        # 3. Analiză Lichiditate
        cr = num(info, 'currentRatio')
        if cr is not None:
            if cr > 1.5: 
                score += 1
                pros.append("Lichiditate solidă")
            elif cr < 1:
                score -= 1
                cons.append("Lichiditate precară")
            
    except Exception as e:
        logger.warning(f"scor sănătate incomplet: {e}")

    # Fără niciun indicator de bilanț, scorul ar rămâne 5/10 („mediu") fără nicio bază.
    if all(num(info, k) is None for k in ('debtToEquity', 'returnOnEquity', 'currentRatio')):
        return None, pros, cons
    
    return max(1, min(10, score)), pros, cons

# get_sector_benchmarks și generate_advanced_audit_v2 sunt în analytics/audit.py (importate sus).

ALTMAN_ZONE_TEXT = {
    "safe": ("ZONĂ SIGURĂ", "#3FB950", "Risc statistic scăzut de dificultate financiară."),
    "grey": ("ZONĂ GRI", "#D29922", "Semnal neconcludent: nici sigură, nici în dificultate."),
    "distress": ("ZONĂ DE DIFICULTATE", "#F85149", "Profil asemănător companiilor care au ajuns în dificultate financiară în următorii 2 ani."),
}


def calculate_altman_z(info, symbol, fin):
    """Altman Z / Z'' din situațiile financiare (analytics/fundamentals.py).

    Întoarce un dict: value, variant ("Z" / "Z''" / None), zone ("safe" / "grey" / "distress" / None),
    label, color, message, detail (componentele). Sectorul financiar și datele lipsă dau value=None;
    apelanții tratează None ca pilon lipsă. Capitalizarea intră în Z doar când moneda situațiilor
    e aceeași cu cea de tranzacționare (sau la BVB); altfel Z-ul original rămâne N/A.
    """
    variant = fund.altman_variant(info.get('sector'), symbol)
    out = {"value": None, "variant": variant, "zone": None, "label": "N/A", "color": "#8B949E",
           "message": "", "detail": None}
    if variant is None:
        out["message"] = "Altman Z nu se aplică băncilor, asigurătorilor și fondurilor (bilanțul lor are altă structură)."
        return out
    fin_curr, trade_curr = info.get('financialCurrency'), info.get('currency')
    same_currency = (fin_curr == trade_curr) if (fin_curr and trade_curr) else str(symbol).upper().endswith(".RO")
    detail = fund.altman_z(fin.get("income"), fin.get("balance"), fin.get("q_income"), fin.get("q_balance"),
                           market_cap=num(info, 'marketCap') if same_currency else None)
    out["detail"] = detail
    value = detail["z"] if variant == "Z" else detail["z2"]
    if value is None and variant == "Z" and detail["z2"] is not None:
        # Fără capitalizare în moneda situațiilor, Z-ul original nu se poate calcula: se trece pe Z''.
        variant, value = "Z''", detail["z2"]
        out["variant"] = variant
    if value is None:
        out["message"] = "Date insuficiente în situațiile financiare (lipsesc: " + ", ".join(detail["missing"]) + ")."
        return out
    zone = fund.altman_zone(value, variant)
    out.update(value=value, zone=zone, label=ALTMAN_ZONE_TEXT[zone][0], color=ALTMAN_ZONE_TEXT[zone][1],
               message=ALTMAN_ZONE_TEXT[zone][2])
    return out

def analyze_dividend_quality(info):
    """Analizează dacă dividendele și profitul sunt sustenabile.

    Întoarce (verdicte, cash_to_income, payout_pct). Ultimele două sunt None când
    datele lipsesc sau când raportul nu are sens (profit net zero sau negativ).
    Înainte, un profit net lipsă era înlocuit cu 1, iar raportul ieșea egal cu
    fluxul de numerar în dolari (miliarde de „x").
    """
    payout = num(info, 'payoutRatio')
    net_income = num(info, 'netIncomeToCommon')
    cash_flow = num(info, 'operatingCashflow')
    
    # Calculăm Calitatea Profitului (Cash Flow / Net Income)
    quality_ratio = None
    if net_income is not None and cash_flow is not None and net_income > 0:
        quality_ratio = cash_flow / net_income
    
    verdicts = []
    
    # Analiză Payout
    if payout is not None:
        if payout > 0.80:
            verdicts.append("🚨 **Dividend Periculos:** Firma distribuie peste 80% din profit. Riscul de tăiere a dividendului este imens.")
        elif 0.30 < payout <= 0.60:
            verdicts.append("✅ **Dividend Sustenabil:** Distribuție echilibrată, lăsând loc și pentru reinvestiții.")
        
    # Analiză Calitate Profit
    if quality_ratio is not None:
        if quality_ratio < 0.7:
            verdicts.append("⚠️ **Calitate Slabă a Profitului:** Firma raportează profit, dar nu încasează suficient cash. Atenție la contabilitate!")
        elif quality_ratio > 1.2:
            verdicts.append("💎 **Profit de Înaltă Calitate:** Cash-flow-ul depășește profitul net. Semn de business ultra-sănătos.")
    elif net_income is not None and net_income <= 0:
        verdicts.append("ℹ️ **Calitatea profitului nu se poate evalua:** profitul net este zero sau negativ.")
    else:
        verdicts.append("ℹ️ **Calitatea profitului nu se poate evalua:** lipsesc profitul net sau fluxul de numerar operațional.")
        
    return verdicts, quality_ratio, (payout * 100 if payout is not None else None)
        
# --- FUNCȚIE GET STOCK DATA (FINAL - SMART MODE) ---
import requests

# 1. Creăm o sesiune 'deghizată' ca un browser Chrome real
cloud_session = requests.Session()
cloud_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive"
})

# --- FUNCȚIE GET STOCK DATA (FINAL - SMART MODE) ---
def _safe_info(t):
    """`Ticker.info` completat cu ce se poate lua din `fast_info`.

    Pe Streamlit Cloud, Yahoo poate refuza endpoint-ul de fundamentale (info, acționariat,
    opțiuni) în timp ce prețurile și situațiile financiare merg. Atunci `info` vine aproape gol.
    Moneda, capitalizarea și prețul vin din endpoint-ul de prețuri, care funcționează.
    Cheia `_fundamentals_available` spune interfeței dacă există date fundamentale.
    """
    info_error = None
    try:
        info = dict(t.info or {})
        if len(info) < 10:
            info_error = f"răspuns aproape gol de la Yahoo ({len(info)} câmpuri)"
    except Exception as e:
        logger.warning(f"info indisponibil pentru {getattr(t, 'ticker', '?')}: {e}")
        info_error = f"{type(e).__name__}: {str(e)[:300]}"
        info = {}
    try:
        fi = t.fast_info
        for key, attr in (("currency", "currency"), ("marketCap", "market_cap"),
                          ("previousClose", "previous_close"), ("currentPrice", "last_price")):
            if info.get(key) is None:
                try:
                    value = getattr(fi, attr)
                    if value is not None:
                        info[key] = value
                except Exception as e:
                    logger.warning(f"fast_info.{attr} indisponibil: {e}")
    except Exception as e:
        logger.warning(f"fast_info indisponibil: {e}")
    info["_fundamentals_available"] = any(
        num(info, k) is not None for k in ("trailingPE", "returnOnEquity", "debtToEquity", "profitMargins", "bookValue")
    )
    # Motivul tehnic, afișat sub banner: fără el nu se poate deosebi o limitare (429) de alt defect.
    info["_info_error"] = info_error
    return info


def _safe_earnings(t):
    try:
        return getattr(t, 'earnings_history', None)
    except Exception as e:
        logger.warning(f"earnings indisponibil: {e}")
        return None


@st.cache_data(ttl=900)
def get_stock_data(symbol):
    try:
        # Lăsăm yfinance să gestioneze conexiunea automat pentru a evita eroarea curl_cffi
        t = yf.Ticker(symbol)
        
        # Încercăm să descărcăm datele istorice
        hist = t.history(period="5y", auto_adjust=True)

        # Fallback pentru Bursa de Valori București (BVB)
        if hist.empty and not symbol.endswith(".RO"):
            sym_ro = symbol + ".RO"
            t_ro = yf.Ticker(sym_ro)
            hist_ro = t_ro.history(period="5y", auto_adjust=True)
            if not hist_ro.empty:
                return hist_ro, _safe_info(t_ro), _safe_earnings(t_ro), sym_ro

        if hist.empty:
            return None, None, None, symbol

        return hist, _safe_info(t), _safe_earnings(t), symbol

    except Exception as e:
        # Dacă apare o eroare de conexiune, o afișăm în consolă pentru debug, nu blocăm UI-ul
        logger.warning(f"Eroare yfinance pentru {symbol}: {e}")
        return None, None, None, symbol

def calculate_technical_indicators(df):
    if df is None or df.empty: return df
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    # RSI cu netezire Wilder și MACD cu EMA standard (analytics/technical.py, cu teste).
    # Varianta veche folosea medii simple, deci valorile difereau de TradingView/XTB.
    df['RSI'] = rsi_wilder(df['Close'], 14)
    df['MACD'], df['Signal'] = macd_lines(df['Close'])
    return df

def plot_correlation_matrix(tickers, show_cards=True):
    """Matricea de corelație (heatmap). Cu `show_cards=False` cardurile pe perechi nu se afișează aici:
    se întoarce matricea, iar apelantul le afișează unde vrea cu `render_diversification_cards`."""
    if len(tickers) < 2: return None
    try:
        # 1. Descărcare și calcul
        data = close_frame(yf.download(tickers, period="1y", progress=False, auto_adjust=True), tickers)
        # Doar zilele în care toate simbolurile au randament: fără randamente de 0 inventate pentru zilele fără ședință.
        corr_matrix, corr_days = common_day_correlation(data, min_obs=60)
        if corr_matrix.empty:
            st.info(f"Corelațiile nu pot fi calculate: doar {corr_days} zile comune de tranzacționare (minimum 60) "
                    "sau mai puțin de două simboluri cu istoric.")
            return None
        st.caption(f"Corelații pe {corr_days} zile comune de tranzacționare din ultimul an (randamente zilnice). "
                   f"Peste {ro(CORR_HIGH):.2f}: diversificare slabă; {ro(CORR_MID):.2f}–{ro(CORR_HIGH):.2f}: moderată; "
                   f"sub {ro(CORR_MID):.2f}: bună. Praguri orientative, netestate, aceleași pe carduri.")
        
        # 2. Text Heatmap (clasele din analytics/risk.py: aceleași praguri ca pe carduri)
        band_short = {"slaba": "⚠️ Slabă", "moderata": "Moderată", "buna": "✅ Bună"}
        text_matrix = []
        for i in range(len(corr_matrix)):
            row_text = []
            for j in range(len(corr_matrix)):
                val = corr_matrix.iloc[i, j]
                band = correlation_band(val)
                if i == j: label = f"{ro(1.0):.2f}"
                elif band is None: label = "N/A"
                else: label = f"{ro(val):.2f}<br>{band_short[band]}"
                row_text.append(label)
            text_matrix.append(row_text)

        # 3. Grafic Plotly: scală divergentă, albastru (negativ: pozițiile se compensează) – gri (0) – portocaliu (pozitiv)
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale=CORR_SCALE,
            zmin=-1, zmax=1, xgap=2, ygap=2,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(color=CORR_TEXT),
            colorbar=dict(title="Corelație"),
            hovertemplate="%{y} ↔ %{x}<br>Corelație: %{z:.2f}<extra></extra>"
        ))
        
        fig.update_layout(
            height=400, template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(side="bottom")
        )

        plot_ro(fig, width='stretch')
        
        if show_cards:
            render_diversification_cards(corr_matrix)
        return corr_matrix
    except Exception as e:
        st.error(f"Eroare Matrice: {e}")
        return None
    
def render_diversification_cards(corr_matrix):
    """Cardurile „Analiză Strategică a Diversificării”: o pereche de simboluri pe card, cu corelația ei."""
    # 4. RAPORT VIZUAL STILIZAT (Cards)
    st.markdown("### 📋 Analiză Strategică a Diversificării")
    
    cols = st.columns(len(corr_matrix.columns) - 1 if len(corr_matrix.columns) <= 3 else 2)
    col_idx = 0

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            score = corr_matrix.iloc[i, j]
            t1, t2 = corr_matrix.columns[i], corr_matrix.columns[j]
            
            # Clasa perechii: aceleași praguri ca în heatmap (analytics/risk.py, orientative, netestate)
            band = correlation_band(score)
            if band is None:
                continue
            label = CORR_BANDS[band]
            if band == "slaba":
                color, icon, bg_light = "#F85149", "🚫", "rgba(248, 81, 73, 0.1)"
            elif band == "moderata":
                color, icon, bg_light = "#DBAB09", "⚠️", "rgba(219, 171, 9, 0.1)"
            else:
                color, icon, bg_light = "#3FB950", "🛡️", "rgba(63, 185, 80, 0.1)"

            # Randare Card HTML/CSS
            with cols[col_idx % len(cols)]:
                st.markdown(f"""
                    <div style="
                        background-color: #161B22; 
                        border-left: 5px solid {color}; 
                        padding: 15px; 
                        border-radius: 10px; 
                        margin-bottom: 10px;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: #8B949E; font-size: 12px; font-weight: bold; text-transform: uppercase;">{label}</span>
                            <span style="font-size: 20px;">{icon}</span>
                        </div>
                        <h4 style="margin: 10px 0; color: var(--tk-text);">{t1} <span style="color: {color};">↔</span> {t2}</h4>
                        <div style="background: {bg_light}; padding: 5px 10px; border-radius: 5px; display: inline-block;">
                            <span style="color: {color}; font-family: monospace; font-size: 18px; font-weight: bold;">{ro(score):.2f}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            col_idx += 1
    st.caption(f"Corelația randamentelor zilnice pe perechi. Peste {ro(CORR_HIGH):.2f}: slabă; {ro(CORR_MID):.2f}–{ro(CORR_HIGH):.2f}: moderată; "
               f"sub {ro(CORR_MID):.2f} (inclusiv negativă): bună. Praguri orientative, netestate.")


@st.cache_data(ttl=3600, show_spinner=False)
def _benchmark_since(bench_ticker, start_iso):
    """Închiderile ajustate ale benchmark-ului de la `start_iso` (YYYY-MM-DD), în cache 1 oră. Înainte se descărcau la
    fiecare clic, o dată pe fiecare tab. Fără date → excepție (nu intră în cache)."""
    _yf_limiter.wait_if_needed()
    spy_data = yf.download(bench_ticker, start=start_iso, progress=False, auto_adjust=True)['Close']
    if isinstance(spy_data, pd.DataFrame):
        spy_data = spy_data.iloc[:, 0]
    spy_data = spy_data.dropna()
    if spy_data.empty:
        raise ValueError(f"Yahoo: {bench_ticker} fără închideri de la {start_iso}")
    return spy_data

def render_benchmark_comparison(portfolio_curve, bench_ticker="SPY", bench_name="S&P 500"):
    """Compară performanța portofoliului cu benchmark-ul folosind procente în tooltip."""
    if portfolio_curve is None or portfolio_curve.empty:
        return
    
    try:
        start_date = portfolio_curve.index[0]
        try:
            spy_data = _benchmark_since(bench_ticker, f"{pd.Timestamp(start_date):%Y-%m-%d}")
        except ValueError as exc:   # Yahoo fără date: mesajul de sincronizare de mai jos, ca înainte
            logger.warning("benchmark %s: %s", bench_ticker, exc)
            spy_data = pd.Series(dtype=float)
        
        combined = pd.DataFrame({'Portfolio': portfolio_curve, 'Benchmark': spy_data}).ffill().dropna()
        
        if combined.empty:
            st.warning("Nu s-au putut sincroniza datele pentru benchmark.")
            return

        # Calculăm evoluția procentuală față de punctul zero (start)
        # Formula: ((Valoare Curentă / Valoare Start) - 1) * 100
        port_perf = ((combined['Portfolio'] / combined['Portfolio'].iloc[0]) - 1) * 100
        bench_perf = ((combined['Benchmark'] / combined['Benchmark'].iloc[0]) - 1) * 100
        
        port_ret = port_perf.iloc[-1]
        bench_ret = bench_perf.iloc[-1]
        alpha = port_ret - bench_ret 
        if not (np.isfinite(port_ret) and np.isfinite(bench_ret)):
            # Curba care pornește de la 0 sau date lipsă ar afișa „+nan%”: mai bine N/A explicit.
            logger.warning("benchmark %s: randament nedefinit (portofoliu %s, benchmark %s)", bench_ticker, port_ret, bench_ret)
            st.info(f"Alpha față de {bench_name}: N/A. Randamentul portofoliului sau al benchmark-ului nu poate fi calculat "
                    "pe fereastra aleasă (curba pornește de la 0 sau lipsesc date).")
            return

        fig = go.Figure()
        
        # Adăugăm linia Portofoliului
        fig.add_trace(go.Scatter(
            x=port_perf.index, 
            y=port_perf, 
            name='Portofoliul Tău', 
            line=dict(color='#3FB950', width=3),
            hovertemplate="<b>Data:</b> %{x}<br><b>Evoluție:</b> %{y:.2f}%<extra></extra>"
        ))
        
        # Adăugăm linia Benchmark-ului
        fig.add_trace(go.Scatter(
            x=bench_perf.index, 
            y=bench_perf, 
            name=bench_name, 
            line=dict(color="#0678FA", dash='dash'),
            hovertemplate="<b>Benchmark:</b> %{x}<br><b>Evoluție:</b> %{y:.2f}%<extra></extra>"
        ))
        
        fig.update_layout(
            title=f"Performanță Relativă vs {bench_name} (%)",
            height=400, template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(ticksuffix="%", gridcolor="#446E9E"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        plot_ro(fig, width='stretch')

        # Afișare Alpha Card (Rămâne neschimbat, dar folosim variabilele noi)
        alpha_color = "#3FB950" if alpha > 0 else "#F85149"
        st.markdown(f"""
            <div style="background-color: #161B22; padding: 20px; border-radius: 12px; border-top: 4px solid {alpha_color}; text-align: center;">
                <h4 style="color: #8B949E; margin-bottom: 5px;">Alpha (Diferență față de {bench_name})</h4>
                <h1 style="color: {alpha_color}; margin: 0;">{ro(alpha):+.2f}%</h1>
                <p style="color: #8B949E; font-size: 14px;">Portofoliu: {ro(port_ret):+.2f}% | {bench_name}: {ro(bench_ret):+.2f}%</p>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Benchmark Eroare: {str(e)}")

# --- FUNCȚII NOI PENTRU REZUMAT ZILNIC (DAILY BRIEFING) ---

BRIEFING_US_INDICES = ["^GSPC", "^IXIC", "^DJI"]
# Universul SUA urmărit (lista unică: tools/univers_urmarit_sua.txt, prin data/universes.py; nu S&P 500).
BRIEFING_US_UNIVERSE = universes.symbols("us")
BRIEFING_BET_PROXY = "TVBETETF.RO"
# Etapa C: indicii Europei (cardurile tabului Europa) și ETF-urile sectoriale SPDR (Wall Street). Nu intră în lățime și topuri.
BRIEFING_EU_INDICES = ["^STOXX", "^GDAXI", "^FCHI", "^FTSE"]
BRIEFING_US_SECTORS = ["XLK", "XLC", "XLY", "XLP", "XLF", "XLV", "XLI", "XLE", "XLB", "XLU", "XLRE"]


def briefing_bvb_universe():
    """Universul BVB (lista unică tools/univers_bvb.txt: BET Plus + BET-FI + BETAeRO; și testul istoric)."""
    return universes.symbols("bvb")


def briefing_bvb_sections():
    """{simbol: piața} din secțiunile „# == Nume ==” ale tools/univers_bvb.txt."""
    return universes.sections("bvb")


BRIEF_RETRY_CHUNK = 20       # simbolurile refuzate se recer în grupuri mici
BRIEF_RETRY_PASSES = 2
BRIEF_MIN_COVERAGE = 0.8     # sub 80% din simboluri cu preț, citirea e considerată parțială și NU intră în cache


class _PartialBriefing(Exception):
    """Citire Yahoo incompletă: datele se folosesc o dată, fără cache, ca la următoarea deschidere să se reîncerce."""
    def __init__(self, data):
        super().__init__("citire parțială")
        self.data = data


def _download_briefing(syms):
    """Închiderile neajustate și volumele pe 6 luni, într-o cerere; simbolurile rămase fără preț (Yahoo refuză uneori
    o parte dintr-o cerere mare) se recer în grupuri de BRIEF_RETRY_CHUNK, de cel mult BRIEF_RETRY_PASSES ori."""
    _yf_limiter.wait_if_needed()
    raw = yf.download(syms, period="6mo", progress=False, auto_adjust=False, group_by="ticker")
    close, volume = close_frame(raw, syms), field_frame(raw, "Volume", syms)
    for attempt in range(BRIEF_RETRY_PASSES):
        missing = briefing_calc.symbols_without_prices(close, syms)
        if not missing:
            break
        logger.warning("rezumat: %d simboluri fără preț, reîncercarea %d", len(missing), attempt + 1)
        time.sleep(2 * (attempt + 1))
        for i in range(0, len(missing), BRIEF_RETRY_CHUNK):
            chunk = missing[i:i + BRIEF_RETRY_CHUNK]
            _yf_limiter.wait_if_needed()
            try:
                raw2 = yf.download(chunk, period="6mo", progress=False, auto_adjust=False, group_by="ticker")
            except Exception as exc:  # rețea / Yahoo
                logger.warning("rezumat, reîncercare %s...: %s", chunk[0], exc)
                continue
            close = briefing_calc.merge_price_frames(close, close_frame(raw2, chunk))
            volume = briefing_calc.merge_price_frames(volume, field_frame(raw2, "Volume", chunk))
    return close, volume


@st.cache_data(ttl=900, show_spinner=False)
def _briefing_market(syms):
    """O piață a Rezumatului (cache 15 min). Sub BRIEF_MIN_COVERAGE → _PartialBriefing (excepția nu intră în cache)."""
    syms = list(syms)
    close, volume = _download_briefing(syms)
    data = {"close": close, "volume": volume}
    if briefing_calc.coverage(close, syms) < BRIEF_MIN_COVERAGE:
        raise _PartialBriefing(data)
    return data


def briefing_market_symbols(key):
    """Simbolurile cererii Yahoo a unei piețe din Rezumatul Zilei (`bvb`, `us`, `eu`). Aceeași listă oriunde e
    folosită (Rezumatul, topurile din Piață Globală), ca piața să fie descărcată o singură dată și citită din cache."""
    if key == "bvb":
        return briefing_bvb_universe() + [BRIEFING_BET_PROXY]
    if key == "us":
        return BRIEFING_US_UNIVERSE + BRIEFING_US_INDICES + BRIEFING_US_SECTORS
    if key == "eu":
        return universes.symbols("eu") + BRIEFING_EU_INDICES
    raise KeyError(key)


def get_briefing_market(key):
    """(data, partial) pentru o piață a Rezumatului: data = {"close", "volume"}; partial = True dacă Yahoo a dat sub
    BRIEF_MIN_COVERAGE din simboluri (datele se folosesc, dar nu intră în cache)."""
    syms = briefing_market_symbols(key)
    try:
        return _briefing_market(tuple(syms)), False
    except _PartialBriefing as exc:
        logger.warning("rezumat %s: citire parțială (%d din %d simboluri fără preț)", key,
                       len(briefing_calc.symbols_without_prices(exc.data["close"], syms)), len(syms))
        return exc.data, True


def get_daily_briefing_data():
    """Închiderile NEAJUSTATE și volumele pe ~6 luni (Yahoo), pentru Rezumatul Zilei: dict bvb_close, bvb_volume
    (universul BVB + ETF-ul BET), us_close, us_volume (universul SUA + indicii + ETF-urile sectoriale), eu_close,
    eu_volume (universul Europa + indicii) și partial = {piață: True dacă Yahoo a dat sub 80% din simboluri}.
    Fiecare piață are cererea și cache-ul ei: o citire parțială nu se păstrează și nu le strică pe celelalte.
    Prețul neajustat dă variația zilei cum o raportează bursa; 6 luni ajung pentru σ pe 60 de ședințe.
    Fără niciun preț pe nicio piață → excepție."""
    out = {"partial": {}}
    for key in ("bvb", "us", "eu"):
        data, partial = get_briefing_market(key)
        out[f"{key}_close"], out[f"{key}_volume"] = data["close"], data["volume"]
        out["partial"][key] = partial
    if out["bvb_close"].empty and out["us_close"].empty and out["eu_close"].empty:
        raise ValueError("Yahoo: nicio închidere pentru Rezumatul Zilei")
    return out


@st.cache_data(ttl=900, show_spinner=False)
def get_portfolio_day_closes(symbols):
    """Închiderile pe ~6 luni pentru pozițiile deschise din registru („Portofoliul meu azi”): Close din Yahoo cu
    auto_adjust=False, adică ajustat doar la splituri (cantitatea din registru e cea de după split), nu și la dividende.
    6 luni ajung pentru σ pe 60 de ședințe. Fără niciun preț → excepție (nu intră în cache)."""
    symbols = list(symbols)
    _yf_limiter.wait_if_needed()
    raw = yf.download(symbols, period="6mo", progress=False, auto_adjust=False, group_by="ticker")
    closes = close_frame(raw, symbols)
    if closes is None or closes.empty:
        raise ValueError("Yahoo: nicio închidere pentru pozițiile din registru")
    return closes


PDAY_MARKET_HOURS = {"bvb": "BVB", "us": "NYSE", "eu": "XETRA"}
PDAY_HELP = {
    "grand": ("Tot ce ai la brokeri, în lei la ultimul curs BNR: acțiunile și ETF-urile deschise (cantitatea × ultima "
              "închidere) + titlurile de stat și obligațiunile (prețul din foaia `Preturi_manuale`, altfel nominal 100, "
              "proxy) + numerarul, toate din registrul `Tranzactii`. Dedesubt: P/L-ul ultimei ședințe (doar acțiunile și "
              "ETF-urile au preț zilnic) și procentul lui față de valoarea totală de la închiderea precedentă. "
              "„3. Portofoliu” arată aceleași poziții pe fiecare monedă și folosește cursul de piață Yahoo, deci totalul "
              "în lei de acolo poate diferi puțin."),
    "equity": ("Acțiunile și ETF-urile deschise: cantitatea × ultima închidere × ultimul curs BNR. Dedesubt: P/L-ul "
               "ultimei ședințe a pieței fiecăreia, cantitatea de acum × (închiderea − închiderea precedentă), în lei; "
               "procentul e față de valoarea acelorași poziții la închiderea precedentă. Doar efectul prețului: efectul "
               "cursului din ziua respectivă nu e inclus. Pozițiile fără cotație în ședință intră în valoare (la ultima "
               "lor cotație), dar nu și în P/L."),
    "bonds": ("Titlurile de stat și obligațiunile din registru: cantitatea × prețul din foaia `Preturi_manuale` (pe unitate, "
              "fără dobânda acumulată), altfel nominalul 100 (proxy), × cursul BNR. Fără variație zilnică."),
    "cash": "Numerarul la brokeri din registrul `Tranzactii` (suma mișcărilor pe broker și monedă), × cursul BNR.",
    "market": ("Valoarea de azi a acțiunilor și ETF-urilor din această piață (cantitatea × ultima închidere × cursul BNR) "
               "și, dedesubt, P/L-ul lor în ultima ședință (data din etichetă), în lei și în procente față de "
               "închiderea precedentă."),
    "Contribuție": ("P/L-ul poziției în lei ÷ valoarea tuturor acțiunilor și ETF-urilor incluse la închiderea precedentă, "
                    "în puncte procentuale. Suma contribuțiilor = variația acțiunilor și ETF-urilor."),
    "σ": ("Variația zilei ÷ abaterea standard a variațiilor zilnice din ultimele 60 de ședințe (fără ziua curentă; "
          "minimum 20). Sub 1σ obișnuită · 1–2σ mai mare decât de obicei · 2–3σ neobișnuită (~5% din zile) · peste 3σ extremă (praguri orientative, netestate)."),
}


def rezumat_fx():
    """Ultimul curs BNR pe valută pentru Rezumatul Zilei (toate sumele în lei): (dict cod → lei, dict cod → data)."""
    fx_hist, _failed = get_bnr_fx_history()
    return pday_calc.fx_from_bnr(fx_hist)


def render_portfolio_today(sessions, fx, fx_dates):
    """„💼 Portofoliul meu azi” (Rezumatul Zilei, etapa B), toate sumele în lei la cursul BNR. Calculele sunt în
    analytics/portfolio_day.py. `sessions` = {piață: data ultimei ședințe din universul întreg al Rezumatului}."""
    st.markdown("### 💼 Portofoliul meu azi")
    values = load_ledger_values()
    if values is None:
        st.info("Foaia `Tranzactii` nu a putut fi citită din `portofoliu_db`.")
        return
    ledger_df, _ = ledger_calc.parse_ledger(values)
    if ledger_df.empty:
        st.info("Foaia `Tranzactii` nu conține tranzacții.")
        return
    positions, _ = ledger_calc.build_positions(ledger_df)
    holdings = pday_calc.held_positions(positions, ledger_calc.NOMINAL_CLASSES, ledger_calc.QTY_EPS)
    closes = pd.DataFrame()
    if not holdings.empty:
        with st.spinner("Citim ultima ședință pentru pozițiile tale..."):
            try:
                closes = get_portfolio_day_closes(tuple(sorted(holdings["Symbol"])))
            except Exception as exc:  # rețea / Yahoo
                logger.warning("portofoliul azi: %s", exc)
    rows, tot = pday_calc.portfolio_day(holdings, closes, fx, sessions, pday_calc.last_trade_dates(ledger_df))
    manual_prices, _ = ledger_calc.parse_manual_prices(load_manual_price_values())
    bonds = pday_calc.bond_values(positions, fx, manual_prices, ledger_calc.NOMINAL_CLASSES)
    cash = pday_calc.cash_values(ledger_calc.cash_balances(ledger_df), fx)
    grand = pday_calc.grand_total(tot, bonds, cash)

    def _lei(v):
        return "N/A" if v is None or pd.isna(v) else ("+" if v > 0 else "") + format_ro(v, 0, " lei")

    def _val(v):
        return "N/A" if v is None or pd.isna(v) else format_ro(v, 0, " lei")

    def _pp(v, d=2):
        return None if v is None or pd.isna(v) else ("+" if v > 0 else "") + format_ro(v * 100, d, "%")

    def _cpp(v):
        return "N/A" if v is None or pd.isna(v) else ("+" if v > 0 else "") + format_ro(v * 100, 2, " pp")

    def _day(pnl, pct):
        """Delta-ul cardului: P/L-ul zilei în lei și în procente („-276 lei · -0,83%”); fără P/L → fără delta."""
        if pnl is None or pd.isna(pnl):
            return None
        return _lei(pnl) + (f" · {_pp(pct)}" if _pp(pct) else "")

    # Rândul 1: totalul și din ce e făcut
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Valoare totală azi (lei)", _val(grand["value_ron"]), _day(grand["pnl_ron"], grand["pct"]), help=PDAY_HELP["grand"])
    g2.metric("Acțiuni și ETF", _val(tot["value_ron"]), _day(tot["pnl_ron"], tot["pct"]),
              help=PDAY_HELP["equity"] + f" Evaluate: {tot['n_valued']} din {tot['held']}; în P/L: {tot['n']} din {tot['held']}.")
    g3.metric("Titluri de stat", _val(grand["bonds_ron"]), help=PDAY_HELP["bonds"] + f" Poziții: {len(bonds)}.")
    g4.metric("Numerar", _val(grand["cash_ron"]), help=PDAY_HELP["cash"])
    # Rândul 2: acțiunile și ETF-urile pe piață, fiecare cu data ultimei ei ședințe
    mkts = list(tot["by_market"].items())
    if mkts:
        cols = st.columns(len(mkts))
        for col, (mkt, b) in zip(cols, mkts):
            sd = b["session"]
            lbl = f"{pday_calc.MARKET_LABEL[mkt]} · {sd:%d.%m}" if sd is not None else pday_calc.MARKET_LABEL[mkt]
            col.metric(lbl, _val(b["value_ron"]), _day(b["pnl_ron"], b["pct"]),
                       help=PDAY_HELP["market"] + f" Evaluate: {b['n_valued']} din {b['held']}; în P/L: {b['n']} din {b['held']}.")
    partial = [pday_calc.MARKET_LABEL[m] for m, b in mkts
               if briefing_calc.session_is_partial(b["session"], briefing_calc.market_state(PDAY_MARKET_HOURS[m], now_ro()))]
    if partial:
        st.caption(f"⏳ Ședință în desfășurare la {', '.join(partial)}: P/L-ul de acolo e parțial, pe ultimele prețuri.")

    r = rows.set_index("Symbol")
    name = lambda s: s.replace(".RO", "")
    if tot["top"] or tot["bottom"]:
        def _who(syms):
            return ", ".join(f"{name(s)} {_lei(r.loc[s, 'PnL_RON'])} ({_cpp(r.loc[s, 'Contribution'])})" for s in syms) or "—"
        st.markdown(f"**Au tras în sus:** {_who(tot['top'])}  \n**Au tras în jos:** {_who(tot['bottom'])}")
    if tot["alerts"]:
        st.markdown("**⚠️ Mișcări peste 2σ:** " + ", ".join(
            f"{name(s)} {_pp(r.loc[s, 'Pct'])} ({format_ro(r.loc[s, 'Sigma'], 1, 'σ')})" for s in tot["alerts"]))
    if tot["excluded"]:
        st.caption("Neincluse în P/L: " + "; ".join(f"{name(s)} ({why})" for s, why in tot["excluded"]) + ".")
    if tot["approx"]:
        st.caption("Cantitate schimbată în ziua ședinței (tranzacție în registru), P/L aproximativ: "
                   + ", ".join(name(s) for s in tot["approx"]) + ".")
    if grand["missing"]:
        st.caption("Neincluse în valoarea totală: " + ", ".join(grand["missing"]) + ".")
    proxy = bonds.loc[bonds["PriceSource"] == ledger_calc.PRICE_NOMINAL, "Symbol"].tolist() if len(bonds) else []
    if proxy:
        st.caption("Titluri la nominal 100 (proxy, fără preț în `Preturi_manuale`): " + ", ".join(proxy) + ".")
    used = sorted(set(rows["Currency"].dropna()) | set(bonds["Currency"].dropna()) | set(cash["Currency"].dropna()))
    fx_used = [c for c in used if c != "RON" and c in fx_dates]
    if fx_used:
        st.caption("Toate sumele în lei, la cursul BNR: " + "; ".join(f"{c} {format_ro(fx[c], 4)} ({fx_dates[c]:%d.%m.%Y})"
                                                                       for c in fx_used)
                   + ". Fără efectul cursului din ziua ședinței.")
    elif any(c != "RON" for c in used):
        st.caption("Cursurile BNR nu au putut fi citite: sumele în valută nu intră în totaluri.")

    n_lines = len(rows) + len(bonds)
    with st.expander(f"📋 Toate pozițiile ({n_lines}) și numerarul"):
        eq_view = pd.DataFrame({
            "Simbol": [name(s) for s in rows["Symbol"]],
            "Piață": [pday_calc.MARKET_LABEL[m] for m in rows["Market"]],
            "Ședința": rows["Date"].values,
            "Variație": rows["Pct"].values,
            "σ": rows["Sigma"].values,
            "P/L (lei)": rows["PnL_RON"].values,
            "Contribuție": rows["Contribution"].values,
            "Valoare (lei)": rows["Value_RON"].values,
            "Stare": ["" if s == pday_calc.STATUS_OK else s for s in rows["Status"]],
        })
        bond_view = pd.DataFrame({
            "Simbol": bonds["Symbol"].values, "Piață": "Titluri de stat",
            "Ședința": bonds["PriceDate"].values, "Variație": None, "σ": None, "P/L (lei)": None, "Contribuție": None,
            "Valoare (lei)": bonds["Value_RON"].values,
            "Stare": ["nominal 100 (proxy)" if s == ledger_calc.PRICE_NOMINAL else "preț manual" for s in bonds["PriceSource"]],
        })
        # object pe ambele: coloanele goale ale titlurilor de stat (Variație, σ, P/L) nu mai schimbă tipul
        # la concatenare (FutureWarning pandas 2.1); text_frame face oricum totul text
        view = (pd.concat([eq_view.astype(object), bond_view.astype(object)], ignore_index=True)
                if len(bond_view) else eq_view)
        st.dataframe(
            text_frame(view, {
                "Ședința": lambda d: f"{pd.Timestamp(d):%d.%m.%Y}",
                "Variație": lambda v: _pp(v), "σ": lambda v: ("+" if v > 0 else "") + format_ro(v, 1, "σ"),
                "P/L (lei)": _lei, "Contribuție": _cpp, "Valoare (lei)": _val,
            }).style.map(sign_color, subset=["Variație", "σ", "P/L (lei)", "Contribuție"]),
            width='stretch', hide_index=True,
            column_config={"σ": st.column_config.Column(help=PDAY_HELP["σ"]),
                           "Contribuție": st.column_config.Column(help=PDAY_HELP["Contribuție"]),
                           "Ședința": st.column_config.Column(help="Acțiuni și ETF: ziua ultimei cotații folosite. "
                                                                    "Titluri de stat: data prețului din `Preturi_manuale`."),
                           "P/L (lei)": st.column_config.Column(help="Cantitatea × (închiderea − închiderea precedentă) × cursul BNR."),
                           "Valoare (lei)": st.column_config.Column(help="Cantitatea × prețul × cursul BNR.")})
        if len(cash):
            st.dataframe(
                text_frame(pd.DataFrame({"Numerar": [f"{b} · {c}" for b, c in zip(cash["Broker"], cash["Currency"])],
                                         "Valoare (lei)": cash["Value_RON"].values}), {"Valoare (lei)": _val}),
                width='stretch', hide_index=True,
                column_config={"Valoare (lei)": st.column_config.Column(help="Soldul din registru × cursul BNR.")})
    st.markdown("---")
    return {"holdings": holdings, "totals": tot, "rows": rows}


@st.cache_data(ttl=86400, show_spinner=False)
def _yahoo_names(sym):
    """(longName, shortName) din Yahoo pentru un simbol, în cache 24 de ore (pentru filtrul de știri).
    O eroare de rețea nu se salvează în cache (excepția iese din funcție), deci se reîncearcă la următoarea deschidere."""
    _yf_limiter.wait_if_needed()
    info = yf.Ticker(sym).info or {}
    return info.get("longName") or None, info.get("shortName") or None


@st.cache_data(ttl=900, show_spinner=False)
def get_holdings_news(symbols):
    """Știrile din ultimele 48 de ore despre pozițiile tale (Rezumatul Zilei, etapa D): fluxul Yahoo pe fiecare simbol
    plus fluxurile generale (RO și internaționale), păstrate doar dacă titlul menționează compania
    (data/news.py). Întoarce (știri, simboluri fără nume din Yahoo — filtrate doar după simbol)."""
    names, no_name, own = {}, [], {}
    for sym in symbols:
        try:
            names[sym] = tuple(n for n in _yahoo_names(sym) if n)
        except Exception as exc:  # rețea / Yahoo
            logger.warning("știri pe poziții, nume %s: %s", sym, exc)
            names[sym] = ()
        if not names[sym]:
            no_name.append(sym)
        own[sym] = get_company_news_rss(sym, names=names[sym], limit=10)
    items = holdings_news(names, fetch_news_data(), own, now=now_ro().replace(tzinfo=None))
    return items, no_name


def render_calendar_line():
    """Calendarul macro SUA de azi și mâine pe un rând, sus în Rezumatul Zilei (același calendar ca în „4. Piață
    Globală”: FRED + calendarul oficial Fed pentru FOMC). Întoarce publicările de azi/mâine (pentru „3 lucruri”)."""
    today = now_ro().date()
    cal_data, cal_msg = get_fred_calendar(today)
    src = ("Sursa: FRED, calendarul publicărilor (doar ziua, fără oră; majoritatea dimineața la New York, după-amiaza în "
           "România) și calendarul oficial Fed pentru FOMC (14:00 New York). Calendarul pe 14 zile: „4. Piață Globală”.")
    if cal_data is None:
        st.markdown(f"📅 **Calendar macro SUA:** indisponibil acum ({cal_msg})", help=src)
        return []
    cal_rows, _missing = fred_cal.upcoming(cal_data, today)
    soon, nxt = fred_cal.near(cal_rows)
    if soon:
        text = "; ".join(f"{'azi' if r['in_days'] == 0 else 'mâine'} ({r['weekday']} {r['date']:%d.%m}) {r['label']}"
                         for r in soon)
    elif nxt is not None:
        text = (f"azi și mâine nimic urmărit; următoarea: {nxt['label']}, {nxt['weekday']} {nxt['date']:%d.%m} "
                f"(peste {nxt['in_days']} zile)")
    else:
        text = f"nicio publicare urmărită în următoarele {fred_cal.CALENDAR_DAYS} zile"
    st.markdown(f"📅 **Calendar macro SUA:** {text[0].upper() + text[1:]}", help=src)
    return soon


def _holdings_news_html(items):
    return compact_html("".join(
        f"""<div style="margin-bottom:10px; border-bottom:1px solid #30363D; padding-bottom:5px;">
        <span style="background:#21262D; color:#C9D1D9; border-radius:4px; padding:1px 6px; font-size:12px; font-weight:600; margin-right:6px;">{html.escape(it['symbol'].replace('.RO', ''))}</span>
        <a href="{html.escape(safe_link(it['link']) or '#', quote=True)}" style="color:#58A6FF; text-decoration:none; font-weight:600;" target="_blank" rel="noopener noreferrer">{html.escape(it['title'])}</a>
        <div style="font-size:12px; color:#8B949E;">{html.escape(str(it['source']))} • {it['time']:%d.%m %H:%M}</div></div>"""
        for it in items))


def render_holdings_news(holdings):
    """„Despre acțiunile tale” (tabul „Știri” din Rezumatul Zilei): știrile din ultimele 48 de ore despre acțiunile
    din registru (fără ETF-uri și titluri de stat). Întoarce lista de știri (pentru „3 lucruri”)."""
    st.subheader("💼 Despre acțiunile tale (ultimele 48 de ore)")
    syms = [] if holdings is None or holdings.empty else sorted(
        holdings.loc[holdings["Class"].astype(str).str.upper() != "ETF", "Symbol"].astype(str).unique())
    if not syms:
        st.info("Nicio acțiune deschisă în registrul `Tranzactii`.")
        return []
    with st.spinner("Căutăm știrile despre pozițiile tale..."):
        try:
            items, no_name = get_holdings_news(tuple(syms))
        except Exception as exc:  # rețea / RSS
            logger.warning("știri pe poziții: %s", exc)
            items, no_name = None, []
    if items is None:
        st.info("Știrile nu au putut fi citite acum. Se reîncearcă la următoarea deschidere a paginii.")
        items = []
    elif not items:
        st.info("Nicio știre din ultimele 48 de ore care să menționeze acțiunile tale.")
    else:
        st.markdown(_holdings_news_html(items), unsafe_allow_html=True)
    st.caption(f"{len(syms)} acțiuni din registru (fără ETF-uri și titluri de stat). Se păstrează doar știrile al căror "
               "titlu conține numele întreg al companiei sau simbolul (rezumatul nu contează); cel mult 3 pe companie, 15 în total, "
               "ora României. Fluxurile RO acoperă puține companii mici, deci lipsa unei știri nu înseamnă că nu s-a "
               "întâmplat nimic." + (f" Fără nume în Yahoo (filtrate doar după simbol): {', '.join(no_name)}." if no_name else ""))
    return items


@st.cache_data(ttl=900, show_spinner=False)
def get_us_index_news(limit=10):
    """Știrile Yahoo (RSS) pentru ^GSPC și ^IXIC, fără dubluri, cele mai noi primele (cache 15 min)."""
    seen, out = set(), []
    for item in sorted(get_company_news_rss("^GSPC") + get_company_news_rss("^IXIC"),
                       key=lambda x: x.get("date_str", ""), reverse=True):
        if item.get("title") and item["title"] not in seen:
            seen.add(item["title"])
            out.append(item)
    return out[:limit]

# --- FUNCȚII PORTOFOLIU (RESCRISE PENTRU GOOGLE SHEETS) ---
def load_portfolio():
    """Foaia `Portofoliu` (prima foaie) → DataFrame Symbol, Date, Quantity, AvgPrice, Currency, cu numere.

    Se citește cu UNFORMATTED_VALUE: celulele numerice vin ca numere, deci „1.250” (mii, format RO)
    rămâne 1250. Înainte, get_all_records primea textul formatat și îl citea ca 1,25.
    Problemele de citire sunt în `df.attrs["problems"]`. Fără cache: foaia se actualizează manual.
    """
    sheet = connect_to_gsheets()  # Sheet1 implicit = portofoliu
    if not sheet:
        return pd.DataFrame()
    try:
        values = sheet.get_all_values(value_render_option="UNFORMATTED_VALUE",
                                      date_time_render_option="FORMATTED_STRING")
    except gspread.exceptions.GSpreadException as e:
        logger.warning(f"foaia Portofoliu nu a putut fi citită: {e}")
        return pd.DataFrame()
    df, problems = portfolio_sheet.parse_portfolio_values(values)
    df.attrs["problems"] = problems
    return df

# --- FUNCȚII WATCHLIST ---
@st.cache_data(ttl=300, show_spinner=False)
def load_watchlist_values():
    """Conținutul brut al foii `watchlist` (antet + rânduri, numere neformatate); None dacă foaia lipsește
    sau nu poate fi citită. Cache 5 minute: cardul „Țintă intrare” din Analiză o citește la fiecare rerun."""
    ws = connect_to_gsheets("watchlist")
    if ws is None:
        return None
    try:
        # UNFORMATTED_VALUE: ținta vine ca număr; textul formatat românesc „1.250” ar fi citit ca 1,25
        return ws.get_all_values(value_render_option="UNFORMATTED_VALUE", date_time_render_option="FORMATTED_STRING")
    except gspread.exceptions.GSpreadException as e:
        logger.warning(f"foaia watchlist nu a putut fi citită: {e}")
        return None

def load_watchlist():
    """Foaia `watchlist` ca DataFrame (Symbol, TargetPrice număr sau None, Notes); `attrs["problems"]` = ce nu s-a putut citi."""
    df, problems = watchlist_calc.parse_watchlist(load_watchlist_values())
    df.attrs["problems"] = problems
    return df

def add_to_watchlist(symbol, target, note):
    """Adaugă o intrare nouă în watchlist. Întoarce (reușit, mesaj).

    Simbolul deja prezent nu se adaugă a doua oară: cardul din Analiză ar lua doar prima țintă.
    Duplicatul se verifică pe foaia recitită acum (nu pe cache), ca o editare manuală recentă să conteze.
    Simbolul se rezolvă ca în Analiză: întâi cel scris, apoi cu `.RO`; se salvează primul care are preț Yahoo,
    ca tabelul și cardul „Țintă intrare” să găsească același simbol. Fără preț → nu se salvează."""
    written = watchlist_calc.normalize_symbol(symbol)
    symbol = next((c for c in watchlist_calc.symbol_candidates(written) if c in get_watchlist_quotes([c])), None)
    if symbol is None:
        return False, (f"{written}: Yahoo nu are preț pentru {', nici pentru '.join(watchlist_calc.symbol_candidates(written))}. "
                       "Verifică simbolul (Europa: cu sufixul bursei, ex. SAP.DE; BVB: TLV sau TLV.RO).")
    ws = connect_to_gsheets("watchlist")
    if ws is None:
        return False, "Foaia watchlist nu poate fi accesată."
    try:
        values = ws.get_all_values(value_render_option="UNFORMATTED_VALUE", date_time_render_option="FORMATTED_STRING")
        if watchlist_calc.symbol_rows(values, symbol):
            current = watchlist_calc.parse_watchlist(values)[0]
            t = current.loc[current["Symbol"] == symbol, "TargetPrice"].iloc[0]
            t_txt = format_ro(t, 4, trim=True) if pd.notna(t) else "fără țintă"
            return False, f"{symbol} e deja în watchlist (ținta {t_txt}). Șterge-l întâi dacă vrei altă țintă."
        row = watchlist_calc.sheet_row(values[0] if values else [], symbol, target, note, f"{now_ro():%Y-%m-%d}")
        if row is None:
            return False, "Foaia watchlist nu are coloanele Symbol și TargetPrice în antet (rândul 1)."
        ws.append_row(row)  # în ordinea antetului; data intră doar dacă foaia are coloana Date
    except (gspread.exceptions.GSpreadException, requests.RequestException) as e:
        logger.warning(f"watchlist, adăugare {symbol}: {e}")
        return False, f"Eroare la salvarea în foaia watchlist: {e}"
    load_watchlist_values.clear()  # golește doar cache-ul foii watchlist
    currency = watchlist_currency(symbol)
    resolved = f" (scris {written}; găsit ca {symbol})" if symbol != written else ""
    return True, f"Adăugat {symbol} la ținta {format_ro(target, 4, trim=True)} {currency}{resolved}."

def remove_from_watchlist(symbol):
    """Șterge simbolul din watchlist: toate rândurile cu el în coloana Symbol, nu prima celulă egală din foaie
    (înainte `ws.find` căuta în toată foaia, inclusiv în note). Întoarce (reușit, mesaj)."""
    ws = connect_to_gsheets("watchlist")
    if ws is None:
        return False, "Foaia watchlist nu poate fi accesată."
    try:
        rows = watchlist_calc.symbol_rows(ws.get_all_values(), symbol)
        if not rows:
            return False, f"{symbol} nu mai e în coloana Symbol a foii watchlist (poate a fost șters între timp)."
        for r in rows:  # de jos în sus: numerele rândurilor rămase nu se schimbă
            ws.delete_rows(r)
    except (gspread.exceptions.GSpreadException, requests.RequestException) as e:
        logger.warning(f"watchlist, ștergere {symbol}: {e}")
        return False, f"Eroare la ștergerea din foaia watchlist: {e}"
    finally:
        load_watchlist_values.clear()  # golește doar cache-ul foii watchlist (și după o ștergere parțială)
    return True, f"Șters {symbol}" + (f" ({len(rows)} rânduri)." if len(rows) > 1 else ".")

        # load_portfolio nu are cache: nu e nimic de golit

@st.cache_data(ttl=900, show_spinner=False)
def _portfolio_history_closes(tickers):
    """Închiderile ajustate pe 5 ani pentru simbolurile unui tab de monedă (tuplu), în cache 15 minute. Înainte se
    descărcau la fiecare clic din pagină, o dată pe fiecare tab. Fără nicio închidere → excepție (nu intră în cache)."""
    tickers = list(tickers)
    _yf_limiter.wait_if_needed()
    hist_data = yf.download(tickers, period="5y", group_by='ticker', progress=False, auto_adjust=True)
    closes = close_frame(hist_data, tickers)   # tratează toate formele întoarse de yf.download (un simbol sau mai multe)
    if closes is None or closes.empty:
        raise ValueError("Yahoo: nicio închidere pentru pozițiile din tab")
    return closes

def calculate_portfolio_performance(df, history_range="1A"):
    """Evaluează pozițiile și construiește curba de valoare a portofoliului.

    Întoarce (tabel poziții, curbă, variație zilnică abs, variație zilnică %, note).
    Calculele sunt în analytics/portfolio.py (testate în tests/test_portfolio.py).
    `note` spune ce simboluri nu au preț sau istoric și de unde începe curba.
    """
    if df.empty: return pd.DataFrame(), pd.Series(dtype=float), 0, 0, {}
    
    positions = df.copy()
    # smart_to_float înțelege și formatul românesc ("100,5"); pd.to_numeric îl transforma în 0.
    positions['Quantity'] = positions['Quantity'].apply(smart_to_float)
    positions['AvgPrice'] = positions['AvgPrice'].apply(smart_to_float)
    
    tickers = positions['Symbol'].unique().tolist()
    
    # --- Descărcăm totul dintr-o singură lovitură ---
    with st.spinner("Actualizăm portofoliul prin Motor Asincron..."):
        current_prices = get_fast_live_prices(tickers)
        # Istoricul pentru grafic, o cerere pe tab (cache 15 min în _portfolio_history_closes)
        try:
            closes = _portfolio_history_closes(tuple(tickers))
        except Exception as exc:  # rețea / Yahoo: fără istoric, valorile rămân N/A (value_positions)
            logger.warning("istoric portofoliu %s: %s", tickers, exc)
            closes = pd.DataFrame()

    df_result, total_daily_pl_abs, total_daily_pl_pct, notes = value_positions(positions, current_prices, closes)
    portfolio_curve, curve_notes = build_portfolio_curve(positions, closes)
    notes.update(curve_notes)

    # Fereastră calendaristică (1A = un an de date, ~252 de ședințe), nu 365 de rânduri
    portfolio_curve = slice_window(portfolio_curve, history_range)
    
    return df_result, portfolio_curve, total_daily_pl_abs, total_daily_pl_pct, notes

# --- REGISTRUL DE TRANZACȚII (foaia `Tranzactii`) ---
LEDGER_FX_SYMBOLS = {"EUR": "EURRON=X", "USD": "USDRON=X"}
TAB_BENCHMARKS = {"USD": ("SPY", "S&P 500 (SPY)"), "EUR": ("EXSA.DE", "STOXX Europe 600 (EXSA.DE)"),
                  "RON": ("TVBETETF.RO", "BET (ETF TVBETETF)")}
LEDGER_CLASS_ORDER = ["Actiuni RO", "Actiuni SUA", "Actiuni UE", "ETF", "Obligatiuni", "Altele"]

# Explicațiile din semnul întrebării de lângă fiecare indicator: cum se calculează și ce înseamnă.
LEDGER_HELP = {
    "total": "**Cum se calculează:** valoarea de piață a tuturor pozițiilor deschise plus numerarul de la brokeri, "
             "fiecare monedă convertită în RON la cursul de azi.\n\n**Ce înseamnă:** cât ai primi, aproximativ, dacă ai vinde tot azi "
             "(înainte de comisioane și impozite).",
    "positions": "**Cum se calculează:** cantitate × preț curent pentru fiecare poziție, convertit în RON la cursul de azi. "
                 "Obligațiunile și titlurile de stat sunt la prețul din foaia Preturi_manuale, iar fără el la nominal (100).\n\n**Ce înseamnă:** partea din portofoliu care e investită.",
    "cash": "**Cum se calculează:** suma tuturor intrărilor și ieșirilor de numerar din registru, pe fiecare cont, convertită în RON la cursul de azi.\n\n"
            "**Ce înseamnă:** banii neinvestiți care stau la brokeri. Trebuie să coincidă cu soldul din aplicațiile brokerilor la data ultimului export.",
    "net_invested": "**Cum se calculează:** depuneri − retrageri + soldurile de deschidere din 2020, fiecare convertită în RON la cursul din ziua ei. "
                    "Schimburile valutare și transferurile între conturi nu se numără.\n\n**Ce înseamnă:** câți bani din buzunar ai băgat în total în portofoliu.",
    "profit": "**Cum se calculează:** valoarea totală de azi − banii depuși net.\n\n**Ce înseamnă:** câștigul total în RON, din toate sursele: "
              "creșterea prețurilor, dividende, cupoane, dobânzi și cursul valutar, minus comisioane și impozite. "
              "Procentul de dedesubt îl raportează la banii depuși, fără să țină cont de cât timp au stat investiți.",
    "period": "Prima și ultima zi pentru care există valoare calculată. TWR și indicatorii de risc sunt măsurați pe acest interval.",
}
# Indicatorii de randament și risc: explicațiile cu praguri din ui/card_help.py (aceleași ca în taburile pe monede)
LEDGER_HELP.update({k: PORTF_HELP[k] for k in ("xirr", "twr", "twr_annual", "vol", "sharpe", "sortino", "max_dd",
                                               "var_hist", "cvar", "var_param")})
LEDGER_COLUMN_HELP = {
    "Clasă": "Clasa de active din coloana Class a foii Tranzactii. Steluța marchează clasele în care cel puțin o poziție e evaluată la nominal (proxy), fără preț manual.",
    "Monedă": "Moneda în care sunt exprimate toate sumele de pe rând. Rândurile nu se adună între ele decât după conversie.",
    "Poziții": "Câte simboluri din clasă au cantitate deținută mai mare ca zero.",
    "Cost deschis": "Cât ai plătit, cu tot cu comisioane, pentru ce deții acum. Cost mediu ponderat: la o vânzare parțială costul scade proporțional. Acțiunile gratuite au cost 0.",
    "Valoare azi": "Cantitate × preț curent pentru pozițiile deschise. N/A dacă lipsește prețul vreunei poziții.",
    "Nerealizat": "Valoare azi − cost deschis. Profitul sau pierderea pe hârtie la ce deții încă.",
    "Realizat": "Suma, pe toate vânzările, a încasării nete minus costul mediu al acțiunilor vândute. Profitul deja încasat.",
    "Dividende / cupoane": "Dividendele și cupoanele încasate. La Tradeville sunt nete (impozitul e deja reținut); la XTB sunt brute, iar impozitul apare separat.",
    "Impozite": "Impozitele raportate separat de broker (reținere la sursă pe dividende, taxa RO la vânzare). Valoare negativă.",
    "Comisioane": "Comisioanele raportate separat de broker (de ex. taxa SEC). Comisioanele de tranzacționare sunt deja incluse în cost și în realizat.",
    "Rezultat total": "Realizat + nerealizat + dividende/cupoane + impozite + comisioane. Câștigul total al clasei, în moneda ei.",
    "XIRR (% pe an)": "Randamentul anual ponderat cu banii, în moneda clasei: rata care egalează cumpărările cu vânzările, dividendele și valoarea de azi. N/A sub 90 de zile.",
    "Fluxuri": "Câte mișcări de bani au intrat în calculul XIRR. Cu puține fluxuri (sub ~20), randamentul e orientativ.",
}

@st.cache_data(ttl=300, show_spinner=False)
def load_ledger_values():
    """Conținutul brut al foii `Tranzactii` (antet + rânduri); None dacă foaia lipsește sau nu poate fi citită."""
    ws = connect_to_gsheets("Tranzactii")
    if ws is None:
        return None
    try:
        return ws.get_all_values()
    except gspread.exceptions.GSpreadException as e:
        logger.warning(f"foaia Tranzactii nu a putut fi citită: {e}")
        return None

MANUAL_PRICE_STALE_DAYS = 30  # peste atât, prețul manual e semnalat ca vechi

@st.cache_data(ttl=600, show_spinner=False)
def load_iv_history_values():
    """Conținutul brut al foii `IV_istoric` (valori neformatate); None dacă foaia lipsește sau nu poate fi citită."""
    ws = connect_to_gsheets("IV_istoric")
    if ws is None:
        return None
    try:
        return ws.get_all_values(value_render_option="UNFORMATTED_VALUE", date_time_render_option="FORMATTED_STRING")
    except gspread.exceptions.GSpreadException as e:
        logger.warning(f"foaia IV_istoric nu a putut fi citită: {e}")
        return None

def update_iv_history(symbol, iv_pct):
    """Salvează IV-ul la bani de azi în `IV_istoric` (o dată pe zi și simbol, doar în timpul ședinței SUA)
    și întoarce {"rank": iv_rank_percentile(...), "note": text pentru interfață}."""
    values = load_iv_history_values()
    df, problems = iv_history.parse_iv_history(values)
    now = now_ro()  # cu fus orar; se convertește la New York în iv_history
    session = iv_history.us_session_date(now)
    note = ""
    if values is None:
        note = "Foaia IV_istoric nu a putut fi citită; IV Rank rămâne N/A."
    elif iv_pct is not None and iv_history.us_session_open(now) and iv_history.should_save(df, symbol, session):
        ws = connect_to_gsheets("IV_istoric")
        try:
            ws.append_row(iv_history.make_row(session, symbol, iv_pct / 100), value_input_option="RAW")
            load_iv_history_values.clear()  # golește doar cache-ul foii IV_istoric
            df, problems = iv_history.parse_iv_history(load_iv_history_values())
            note = f"IV-ul de azi ({ro(iv_pct):.1f}%) a fost salvat în IV_istoric pentru {session:%d.%m.%Y}."
        except (gspread.exceptions.GSpreadException, AttributeError) as e:
            logger.warning(f"IV_istoric, scriere eșuată pentru {symbol}: {e}")
            note = "IV-ul de azi nu a putut fi salvat în IV_istoric."
    series = iv_history.history_for(df, symbol)
    rank = opt_calc.iv_rank_percentile(series, None if iv_pct is None else iv_pct / 100, as_of=pd.Timestamp(session))
    span = f", {rank['first']:%d.%m.%Y} – {rank['last']:%d.%m.%Y}" if rank["n"] else ""
    note = " ".join(x for x in [note, f"Istoric IV la bani pentru {symbol}: {rank['n']} zile în ultimul an{span}. "
                                "Se salvează automat o valoare pe zi, doar cât bursa SUA e deschisă."] + problems if x)
    return {"rank": rank, "note": note}

@st.cache_data(ttl=300, show_spinner=False)
def load_manual_price_values():
    """Conținutul brut al foii `Preturi_manuale` (antet + rânduri); None dacă foaia lipsește sau nu poate fi citită."""
    ws = connect_to_gsheets("Preturi_manuale")
    if ws is None:
        return None
    try:
        return ws.get_all_values()
    except gspread.exceptions.GSpreadException as e:
        logger.warning(f"foaia Preturi_manuale nu a putut fi citită: {e}")
        return None

@st.cache_data(ttl=900, show_spinner=False)
def get_last_closes(symbols):
    """Ultima închidere pentru fiecare simbol (dict); simbolurile fără date lipsesc din rezultat."""
    symbols = list(symbols)
    if not symbols:
        return {}
    _yf_limiter.wait_if_needed()
    try:
        data = yf.download(symbols, period="5d", group_by='ticker', progress=False, auto_adjust=False)
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.warning(f"ultimele închideri indisponibile: {e}")
        return {}
    closes = close_frame(data, symbols)
    out = {}
    for sym in symbols:
        series = closes[sym].dropna() if sym in closes.columns else pd.Series(dtype="float64")
        if not series.empty:
            out[sym] = float(series.iloc[-1])
    return out

@st.cache_data(ttl=900, show_spinner=False)
def _download_watchlist_quotes(symbols):
    """Ultimul preț și data lui pentru simbolurile din watchlist (tuplu), din `yf.download` pe o lună.

    O lună, nu 5 zile: după o săptămână de sărbători la BVB tot există un ultim preț (data lui se afișează).
    auto_adjust=False: prețul real tranzacționat, cel cu care se compară ținta. Excepțiile urcă la apelant,
    ca o cerere eșuată să nu rămână în cache 15 minute."""
    symbols = list(symbols)
    _yf_limiter.wait_if_needed()
    data = yf.download(symbols, period="1mo", group_by='ticker', progress=False, auto_adjust=False)
    return watchlist_calc.last_quotes(close_frame(data, symbols))

@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False, max_entries=500)
def _yahoo_currency(symbol):
    """Moneda cotației raportată de Yahoo (`fast_info.currency`, de ex. „GBp”, „USD”) sau None.
    Cache 7 zile: moneda unei linii de cotare nu se schimbă. Excepțiile urcă (nu se pun în cache)."""
    _yf_limiter.wait_if_needed()
    cur = yf.Ticker(symbol).fast_info.currency
    return str(cur).strip() if cur else None

def watchlist_currency(symbol):
    """Eticheta de monedă pentru un simbol din watchlist: din Yahoo în afara SUA și BVB, altfel după sufix."""
    yahoo = None
    if watchlist_calc.needs_yahoo_currency(symbol):
        try:
            yahoo = _yahoo_currency(symbol)
        except (requests.RequestException, KeyError, ValueError, AttributeError, TypeError) as e:
            logger.warning(f"monedă Yahoo indisponibilă pentru {symbol}: {e}")
    return watchlist_calc.currency_label(yahoo, symbol)

def get_watchlist_quotes(symbols):
    """{simbol: {"price", "date"}} pentru watchlist; simbolurile fără date lipsesc (apar „N/A”), eroarea Yahoo → {}."""
    symbols = tuple(sorted(set(symbols)))
    if not symbols:
        return {}
    try:
        return _download_watchlist_quotes(symbols)
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.warning(f"Watchlist, prețuri: {e}")
        return {}

@st.cache_data(ttl=900, show_spinner=False)
def _watchlist_moment_closes(symbols):
    """Închiderile AJUSTATE pe 2 ani pentru coloana Moment din Watchlist (tuplu de simboluri), într-o singură cerere.
    2 ani ajung: trendul folosește ultimele 200 de ședințe, momentumul 12-1 ultimele 12 luni (aceleași valori ca pe
    istoricul de 5 ani din Analiză). Excepțiile urcă la apelant, ca o cerere eșuată să nu rămână în cache."""
    symbols = list(symbols)
    _yf_limiter.wait_if_needed()
    data = yf.download(symbols, period="2y", group_by='ticker', progress=False, auto_adjust=True)
    return _naive_index(close_frame(data, symbols))

def get_watchlist_moments(symbols):
    """{simbol: textul coloanei Moment}; Yahoo indisponibil → „N/A” peste tot (mesajul e în log)."""
    symbols = tuple(sorted(set(symbols)))
    try:
        closes = _watchlist_moment_closes(symbols) if symbols else pd.DataFrame()
    except (requests.RequestException, ValueError, KeyError, yf.exceptions.YFException) as e:
        logger.warning(f"Watchlist, moment: {e}")
        closes = pd.DataFrame()
    return watchlist_calc.moment_cells(closes, symbols)

@st.cache_data(ttl=24 * 3600, show_spinner=False, max_entries=300)
def _watchlist_global_fundamental(symbol):
    """Scorul fundamental SUA/UE al unui rând din Watchlist, cu aceleași funcții ca secțiunea din Analiză Companie
    (`get_stock_data`, `resolve_beta_alpha`, `global_fundamental_inputs`, `gfund.fundamental_score`). Cache 24 h:
    prețul din evaluare (DCF invers, E/P) e cel de la momentul calculului, afișat în „at”.
    Întoarce {"score", "label", "na", "at"}: valorile, nu textul (formatul se aplică la afișare, deci o schimbare de
    format nu rămâne 24 h în cache). ETF / fond / indice → na „N/A (ETF)” etc. Yahoo fără răspuns complet (info refuzat,
    situațiile anuale lipsă) → excepție: rezultatul degradat nu intră în cache și se reîncearcă la următorul calcul."""
    hist, info, _, real_sym = get_stock_data(symbol)
    if hist is None or hist.empty or not info:
        raise ValueError(f"Yahoo: fără istoric sau fără info pentru {symbol}")
    if info.get("_info_error"):
        raise ValueError(f"Yahoo: info incomplet pentru {symbol} ({info['_info_error']})")
    qtype = (info.get('quoteType') or "").upper()
    if qtype in NON_COMPANY_QUOTE_TYPES:
        return {"score": None, "label": None, "na": f"N/A ({qtype})", "at": now_ro()}
    risk_stats = resolve_beta_alpha(real_sym, info, hist)
    inp = global_fundamental_inputs(real_sym, info, hist, risk_stats)
    if not inp["financial"] and {"financials", "balance_sheet", "cashflow"} <= set(inp["statement_errors"]):
        raise ValueError(f"Yahoo: situațiile financiare anuale indisponibile pentru {symbol}")
    res = gfund.fundamental_score(inp)
    return {"score": res["score"], "label": res["label"], "na": None, "at": now_ro()}

def watchlist_bvb_fundamentals(ro_symbols, fallback_prices):
    """{simbol .RO: textul coloanei Scor fundamental} din fișierele „Analiză companii”, exact ca secțiunea din Analiză:
    prețurile curente din `get_bvb_file_prices`, iar pentru emitentul evaluat, dacă lipsește, ultimul preț din
    Watchlist (`fallback_prices`; în Analiză e ultimul preț din istoric). Secțiunea transversală se reface pe fiecare
    emitent, ca prețul de rezervă să schimbe doar rândul lui. Întoarce (dict, eroare sau None).
    Bancă (alt șablon, sărită) → „N/A (bancă)”; fără fișier cu date → „N/A (fără fișier)”."""
    files, err = list_company_files()
    if err:
        return {s: "N/A" for s in ro_symbols}, err
    data = build_bvb_dataset(tuple((f["id"], f["name"], f["modified"]) for f in files))
    prices = get_bvb_file_prices(tuple(sorted(data["companii"])))
    base = {t: current_indicators(df, prices.get(t)) for t, df in data["companii"].items()}
    banks = {ticker_from_title(s.split(" (")[0]) for s in data["sarite"] if "bancă" in s}
    out = {}
    for sym in ro_symbols:
        tk = sym[:-3]
        if tk not in data["companii"]:   # aceleași cazuri ca mesajul din Analiză
            out[sym] = "N/A (bancă)" if tk in banks else "N/A (fără fișier)"
            continue
        indicators = dict(base)
        if prices.get(tk) is None:
            indicators[tk] = current_indicators(data["companii"][tk], positive_or_none(fallback_prices.get(sym)))
        cs = cross_section(indicators)
        out[sym] = (watchlist_calc.bvb_fundamental_cell(cs.loc[tk, "scor"], cs.loc[tk, "treapta"])
                    if tk in cs.index and "scor" in cs.columns else "N/A")
    return out, None

def _naive_index(frame):
    """Index de date fără fus orar (Yahoo întoarce uneori index cu fus)."""
    if getattr(frame.index, "tz", None) is not None:
        frame = frame.copy()
        frame.index = frame.index.tz_localize(None)
    return frame

@st.cache_data(ttl=21600, show_spinner=False)
def get_ledger_fx_history(start):
    """Cursul zilnic EUR/RON și USD/RON de la `start`: DataFrame cu coloanele EUR și USD (gol dacă Yahoo refuză)."""
    symbols = list(LEDGER_FX_SYMBOLS.values())
    _yf_limiter.wait_if_needed()
    try:
        data = yf.download(symbols, start=start, group_by='ticker', progress=False, auto_adjust=False)
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.warning(f"istoric curs valutar indisponibil: {e}")
        return pd.DataFrame()
    closes = _naive_index(close_frame(data, symbols))
    closes = closes.rename(columns={sym: cur for cur, sym in LEDGER_FX_SYMBOLS.items()})
    # Yahoo are uneori cotații valutare greșite pentru o zi; un curs care sare peste 10% și revine e eroare de date.
    for cur in closes.columns:
        cleaned, fixed = ledger_calc.despike(closes[cur].dropna(), jump=1.10, tolerance=0.03, max_days=3)
        if fixed:
            logger.warning(f"curs {cur}/RON corectat în {len(fixed)} zile cu cotație aberantă: {[f'{d:%Y-%m-%d}' for d in fixed]}")
        closes[cur] = cleaned.reindex(closes.index)
    return closes

@st.cache_data(ttl=21600, show_spinner=False)
def get_ledger_market_history(symbols, start):
    """Închiderile zilnice, dezajustate la splituri, pentru simbolurile din registru (o coloană per simbol)."""
    symbols = list(symbols)
    if not symbols:
        return pd.DataFrame()
    _yf_limiter.wait_if_needed()
    try:
        data = yf.download(symbols, start=start, group_by='ticker', progress=False, auto_adjust=False, actions=True)
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.warning(f"istoric de preț indisponibil pentru registru: {e}")
        return pd.DataFrame()
    closes = _naive_index(close_frame(data, symbols))
    splits = field_frame(data, "Stock Splits", symbols)
    splits = _naive_index(splits) if not splits.empty else splits
    out = {}
    for sym in closes.columns:
        series = closes[sym].dropna()
        # Yahoo ajustează închiderile la splituri; registrul are cantitățile reale din acea zi.
        out[sym] = ledger_calc.unadjust_splits(series, splits[sym]) if sym in splits.columns else series
    return pd.DataFrame(out)

def render_ledger_total(ledger_df, positions, total_ron, asof):
    """Rezultatul întregului portofoliu în RON: depuneri nete, profit, XIRR, apoi (la cerere) curba reală și TWR."""
    st.markdown("**Întregul portofoliu, în RON** · din registrul `Tranzactii`")
    start = (ledger_df["Date"].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    fx_history = get_ledger_fx_history(start)
    ext, flow_problems = ledger_calc.external_flows(ledger_df, fx_history)
    result = ledger_calc.money_weighted_result(ext, total_ron, asof)

    m1, m2, m3 = st.columns(3)
    m1.metric("Bani depuși, net (RON)", _ledger_num(result["net_invested"]), help=LEDGER_HELP["net_invested"])
    profit_pct = (result["profit"] / result["net_invested"] * 100) if result["profit"] is not None and result["net_invested"] else None
    m2.metric("Profit total (RON)", _ledger_num(result["profit"]), None if profit_pct is None else f"{ro(profit_pct):.1f}% din banii depuși",
              help=LEDGER_HELP["profit"])
    m3.metric("XIRR total (% pe an)", "N/A" if result["xirr"] is None else f"{ro(result['xirr'] * 100):.1f}%", help=LEDGER_HELP["xirr"])
    st.caption(
        f"Din {len(ext)} depuneri, retrageri și solduri de deschidere, fiecare convertită în RON la cursul din ziua ei. "
        "Profit total = valoarea de azi − banii depuși net; include și efectul cursului valutar. "
        "XIRR ține cont de momentul fiecărei depuneri."
    )
    if flow_problems:
        st.warning(f"⚠️ Curs valutar istoric indisponibil pentru {len(flow_problems)} fluxuri (de ex. {flow_problems[0]}) "
                   "Depunerile nete, profitul și XIRR-ul total nu pot fi calculate.")

    traded = positions[~positions["Class"].isin(ledger_calc.NOMINAL_CLASSES)]["Symbol"].tolist()
    if not st.toggle(f"Arată curba reală a portofoliului și TWR (descarcă istoricul pentru {len(traded)} simboluri)", key="ledger_curve"):
        return
    index = pd.bdate_range(ledger_df["Date"].min().normalize(), pd.Timestamp(asof).normalize())
    with st.spinner("Descărcăm istoricul de preț pentru simbolurile din registru..."):
        market = get_ledger_market_history(tuple(sorted(traded)), start)
    quantities = ledger_calc.daily_quantities(ledger_df, index)
    cash_daily = ledger_calc.daily_cash(ledger_df, index)
    trade_prices = ledger_calc.ledger_trade_prices(ledger_df, index)
    symbol_class = dict(zip(positions["Symbol"], positions["Class"]))
    symbol_currency = dict(zip(positions["Symbol"], positions["Currency"]))
    price_table, sources = ledger_calc.build_price_table(index, {s: symbol_class.get(s, "") for s in quantities.columns}, market, trade_prices)
    spike_fixes = []
    curve, blockers = ledger_calc.portfolio_curve_base(quantities, price_table, cash_daily, symbol_currency, fx_history,
                                                       trade_days=ledger_calc.cash_trade_days(ledger_df), corrections=spike_fixes)
    flows_daily = ledger_calc.flows_on_index(ext, index)
    cumulative, twr_total = ledger_calc.twr(curve, flows_daily)

    valid_curve = curve.dropna()
    if valid_curve.empty or twr_total is None:
        st.info("Curba nu poate fi calculată: lipsesc cursurile valutare istorice sau prețurile pentru toate zilele.")
        return
    years = (cumulative.index[-1] - cumulative.index[0]).days / 365.25
    twr_annual = ((1 + twr_total) ** (1 / years) - 1) if (years >= 1 and twr_total > -1) else None
    t1, t2, t3 = st.columns(3)
    t1.metric("TWR total", f"{ro(twr_total * 100):.1f}%", help=LEDGER_HELP["twr"])
    t2.metric("TWR anualizat", "N/A" if twr_annual is None else f"{ro(twr_annual * 100):.1f}%", help=LEDGER_HELP["twr_annual"])
    t3.metric("Perioada măsurată", f"{cumulative.index[0]:%d.%m.%Y} – {cumulative.index[-1]:%d.%m.%Y}", help=LEDGER_HELP["period"])

    deposits = flows_daily.cumsum().reindex(valid_curve.index)
    fig_ledger = go.Figure()
    fig_ledger.add_trace(go.Scatter(x=valid_curve.index, y=valid_curve.values, mode="lines", name="Valoarea portofoliului",
                                    line=dict(color="#3987E5", width=2), hovertemplate="%{y:,.0f} RON<extra>Valoare</extra>"))
    fig_ledger.add_trace(go.Scatter(x=deposits.index, y=deposits.values, mode="lines", name="Bani depuși, net",
                                    line=dict(color="#C98500", width=2, shape="hv"), hovertemplate="%{y:,.0f} RON<extra>Depus net</extra>"))
    fig_ledger.update_layout(
        height=420, template="plotly_dark", hovermode="x unified", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        title="Valoarea portofoliului și banii depuși (RON)", yaxis_title="RON", margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickformat=",.0f"),
    )
    plot_ro(fig_ledger, width='stretch')
    st.caption(
        "Distanța dintre cele două linii e profitul. TWR măsoară randamentul investițiilor fără efectul momentului depunerilor "
        "(comparabil cu un indice); XIRR îl include. Valorile în valută sunt convertite la cursul din fiecare zi."
    )
    proxy_syms = sorted(s for s, src in sources.items() if src in ("proxy", "mixed") and (quantities[s].abs() > 1e-6).any())
    notes = []
    if proxy_syms:
        notes.append(f"Fără istoric complet de piață, evaluate în acele zile la prețul ultimei tranzacții (proxy): {', '.join(proxy_syms)}.")
    if blockers:
        notes.append(f"Zile fără valoare (nici preț de piață, nici tranzacție anterioară) din cauza: {', '.join(blockers)}. "
                     "TWR e calculat doar după ultima astfel de zi.")
    if (quantities.columns.map(lambda s: symbol_class.get(s) in ledger_calc.NOMINAL_CLASSES)).any():
        notes.append("Obligațiunile și titlurile de stat sunt la nominal în toată perioada (proxy); "
                     "prețul din `Preturi_manuale` intră doar în valoarea de azi, nu și în curbă și TWR.")
    if spike_fixes:
        shown = ", ".join(f"{sym} ({day:%d.%m.%Y})" for sym, day in spike_fixes[:8])
        notes.append(f"Vârfuri de o zi-două scoase din curbă (split sau consolidare datată diferit de sursa de prețuri, ori cotație greșită): {shown}"
                     + (f" și încă {len(spike_fixes) - 8}." if len(spike_fixes) > 8 else "."))
    for note in notes:
        st.caption("ℹ️ " + note)

    # --- Risc, pe randamentele zilnice din TWR (fără efectul depunerilor) ---
    daily_returns = (1.0 + cumulative).pct_change(fill_method=None).dropna()
    rf_ron, rf_label = get_risk_free_for_currency("RON")
    risk = performance_risk(daily_returns, rf_ron)
    st.markdown("**Risc, pe curba reală**")
    if risk is None:
        st.info(f"Prea puține zile cu valoare ({len(daily_returns)}) pentru indicatori de risc; sunt necesare cel puțin 60.")
        return
    pct = lambda v, d=1: "N/A" if v is None else f"{ro(v * 100):.{d}f}%"
    ratio = lambda v: "N/A" if v is None else f"{ro(v):.2f}"
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Volatilitate anualizată", pct(risk["vol_ann"]), help=LEDGER_HELP["vol"])
    r2.metric("Sharpe", ratio(risk["sharpe"]), help=LEDGER_HELP["sharpe"])
    r3.metric("Sortino", ratio(risk["sortino"]), help=LEDGER_HELP["sortino"])
    r4.metric("Scădere maximă de la vârf", pct(risk["max_drawdown"]), help=LEDGER_HELP["max_dd"])
    in_ron = (lambda v: "") if total_ron is None else (lambda v: f"{ro(v * total_ron):,.0f} RON la valoarea de azi")
    v1, v2, v3 = st.columns(3)
    v1.metric("VaR 95% istoric (1 zi)", pct(risk["var95_hist"], 2), in_ron(risk["var95_hist"]), delta_color="off", help=LEDGER_HELP["var_hist"])
    v2.metric("CVaR 95% (1 zi)", pct(risk["cvar95"], 2), in_ron(risk["cvar95"]), delta_color="off", help=LEDGER_HELP["cvar"])
    v3.metric("VaR 95% parametric (1 zi)", pct(risk["var95_param"], 2), in_ron(risk["var95_param"]), delta_color="off", help=LEDGER_HELP["var_param"])
    st.caption(
        f"Din {risk['n']} randamente zilnice, {risk['start']:%d.%m.%Y} – {risk['end']:%d.%m.%Y}, în RON. "
        + (f"Rata fără risc: {ro(rf_ron * 100):.2f}% ({rf_label}); e pe 10 ani, deci proxy pentru o rată pe termen scurt. "
           if rf_ron is not None else f"Sharpe și Sortino lipsesc: {rf_label} ")
        + "VaR istoric = pierderea de o zi depășită în 5% din zile; CVaR = pierderea medie din acele zile; "
          "VaR parametric presupune distribuție normală și subestimează cozile. "
          "Volatilitatea e subestimată de pozițiile fără preț zilnic de piață (obligațiuni la nominal, simboluri la prețul ultimei tranzacții) și de numerar."
    )

def _ledger_num(value, decimals=2, suffix=""):
    """Număr pentru tabel; valoarea lipsă se afișează „N/A”, nu 0."""
    if value is None or pd.isna(value):
        return "N/A"
    # format românesc: 1.250,50 (punct la mii, virgulă la zecimale)
    return format_ro(value, decimals, suffix)

def render_ledger_import(values):
    """Importul unui export de la broker în foaia `Tranzactii`: previzualizare, apoi scriere doar a rândurilor noi."""
    with st.expander("📥 Importă un export de la broker (Tradeville, XTB, IBKR)"):
        st.caption(
            "Încarcă raportul de activitate Tradeville sau exportul XTB (.xlsx), ori Flex Query-ul IBKR `Terminal_complet` (.csv), "
            "exact cum l-ai descărcat. "
            "Rândurile deja existente în foaie sunt recunoscute după ID și sărite, deci poți încărca oricând exportul complet. "
            "Nimic nu se scrie până nu apeși butonul de adăugare."
        )
        files = st.file_uploader("Fișiere .xlsx / .csv", type=["xlsx", "csv"], accept_multiple_files=True, key="ledger_import_files")
        if not files:
            return
        if not values or "ID" not in values[0]:
            st.error("Foaia `Tranzactii` nu are antetul așteptat (coloana ID); importul e oprit.")
            return
        id_col = values[0].index("ID")
        known = {row[id_col] for row in values[1:] if len(row) > id_col and row[id_col]}
        fresh_all = []
        for uploaded in files:
            try:
                if uploaded.name.lower().endswith(".csv"):
                    sheets = {"csv": broker_import.read_csv_rows(uploaded)}
                else:
                    sheets = broker_import.read_workbook(uploaded)
            except (zipfile.BadZipFile, KeyError, ValueError, OSError, broker_import.csv.Error) as e:
                st.error(f"**{uploaded.name}**: fișierul nu a putut fi citit ({e}).")
                continue
            broker, records, import_problems, control = broker_import.parse_export(sheets)
            if broker is None:
                st.error(f"**{uploaded.name}**: {import_problems[0]}")
                continue
            fresh = broker_import.new_records(records, known | {r["ID"] for r in fresh_all})
            st.markdown(f"**{uploaded.name}** — {broker}: {len(records)} operațiuni citite, **{len(fresh)} noi**, "
                        f"{len(records) - len(fresh)} deja în foaie.")
            if control and control.get("reported") is not None and abs(control["reported"] - control["parsed"]) > 0.01:
                st.warning(f"⚠️ Numerarul citit ({ro(control['parsed']):,.2f}) diferă de totalul din raport ({ro(control['reported']):,.2f}): "
                           "importul e incomplet. Vezi operațiunile nerecunoscute de mai jos.")
            cash_gaps = {cur: pair for cur, pair in (control.get("cash") or {}).items() if abs(pair[0] - pair[1]) > 0.01}
            for cur, (reported, parsed) in cash_gaps.items():
                st.warning(f"⚠️ {cur}: numerarul citit ({ro(parsed):,.2f}) diferă de variația din Cash Report IBKR "
                           f"({ro(reported):,.2f}): importul e incomplet. Vezi operațiunile nerecunoscute de mai jos.")
            if control.get("positions") is not None:
                ledger_now, _ = ledger_calc.parse_ledger(values)
                position_gaps = broker_import.position_gaps(ledger_now.to_dict("records") + fresh_all + fresh, control["positions"])
                for symbol, mine, theirs in position_gaps:
                    st.warning(f"⚠️ {symbol}: în registru {ro(mine):g}, la IBKR {ro(theirs):g}. Cauze posibile: split sau altă "
                               "acțiune corporativă (neimportate), ori un export care nu începe de la deschiderea contului.")
                if control.get("cash") and not cash_gaps and not position_gaps:
                    st.caption("✅ Numerarul pe fiecare monedă și pozițiile se potrivesc cu raportul IBKR.")
            for problem in import_problems[:20]:
                st.write(f"- ⚠️ {problem}")
            fresh_all.extend(fresh)
        if not fresh_all:
            st.info("Nimic nou de adăugat: toate operațiunile din fișiere sunt deja în foaie.")
            return
        fresh_all = sorted(fresh_all, key=lambda r: (r["Date"], r["ID"]))
        st.dataframe(pd.DataFrame(fresh_all)[ledger_calc.COLUMNS], hide_index=True, width='stretch', height=300)
        if st.button(f"Adaugă {len(fresh_all)} rânduri în foaia Tranzactii", key="ledger_import_write"):
            ws = connect_to_gsheets("Tranzactii")
            if ws is None:
                st.error("Foaia `Tranzactii` nu a putut fi deschisă; nu s-a scris nimic.")
                return
            try:
                ws.append_rows(broker_import.to_sheet_rows(fresh_all), value_input_option="USER_ENTERED")
            except gspread.exceptions.GSpreadException as e:
                st.error(f"Scrierea în foaie a eșuat: {e}")
                return
            load_ledger_values.clear()
            st.success(f"Adăugate {len(fresh_all)} rânduri. Verifică mai jos numerarul pe broker față de soldul din aplicația brokerului.")
            st.rerun()

def render_ledger_results():
    """Rezultatele din tranzacțiile reale: pe clase (în moneda fiecăreia) și valoarea totală de azi în RON.

    Calculele sunt în analytics/ledger.py (testate în tests/test_ledger.py). Aici doar se citesc
    foaia și prețurile și se afișează. Întoarce pozițiile din registru (pentru taburile pe monedă)
    sau None când registrul nu poate fi citit ori e gol.
    """
    st.subheader("📒 Poziții și rezultate")
    values = load_ledger_values()
    if values is None:
        st.info("Foaia `Tranzactii` nu a putut fi citită din `portofoliu_db`.")
        return None
    render_ledger_import(values)
    ledger_df, problems = ledger_calc.parse_ledger(values)
    if ledger_df.empty:
        st.info("Foaia `Tranzactii` nu conține tranzacții. " + " ".join(problems))
        return None
    positions, position_problems = ledger_calc.build_positions(ledger_df)
    problems = problems + position_problems

    # Pozițiile deschise vin din registrul `Tranzactii` (exporturile brokerilor), ca toate celelalte
    # rezultate și ca taburile pe monedă de mai jos.
    held = positions[positions["Quantity"] > ledger_calc.QTY_EPS]
    live_symbols = held.loc[~held["Class"].isin(ledger_calc.NOMINAL_CLASSES), "Symbol"].tolist()
    with st.spinner("Citim prețurile pentru pozițiile din registru..."):
        prices = get_fast_live_prices(live_symbols + list(LEDGER_FX_SYMBOLS.values()))
        # Prețul live poate lipsi (Yahoo refuză des cererile de pe server): completăm cu ultima închidere.
        no_live = tuple(sorted(s for s in live_symbols + list(LEDGER_FX_SYMBOLS.values()) if not prices.get(s)))
        closes_last = get_last_closes(no_live)
    manual_prices, manual_problems = ledger_calc.parse_manual_prices(load_manual_price_values())
    problems = problems + manual_problems
    valued = ledger_calc.value_open_positions(positions, prices, closes_last, manual_prices=manual_prices)
    asof = now_ro().replace(tzinfo=None)
    results = ledger_calc.class_results(ledger_df, positions, valued, asof)
    cash = ledger_calc.cash_balances(ledger_df)
    fx_rates = {cur: prices.get(sym) or closes_last.get(sym) for cur, sym in LEDGER_FX_SYMBOLS.items()}
    fx_from_close = [cur for cur, sym in LEDGER_FX_SYMBOLS.items() if not prices.get(sym) and closes_last.get(sym)]
    by_currency, total_ron, missing_fx = ledger_calc.consolidate(results, cash, fx_rates)
    last_trade = ledger_df["Date"].max()

    positions_ron = (by_currency["Positions"] * by_currency["Rate"]).sum(min_count=1) if total_ron is not None else None
    cash_ron = (by_currency["Cash"] * by_currency["Rate"]).sum(min_count=1) if total_ron is not None else None
    k1, k2, k3 = st.columns(3)
    k1.metric("Valoare totală azi (RON)", _ledger_num(total_ron), help=LEDGER_HELP["total"])
    k2.metric("Poziții (RON)", _ledger_num(positions_ron), help=LEDGER_HELP["positions"])
    k3.metric("Numerar la brokeri (RON)", _ledger_num(cash_ron), help=LEDGER_HELP["cash"])
    st.caption(
        f"Totul vine din registrul `Tranzactii`, completat din exporturile brokerilor ({len(ledger_df)} tranzacții, "
        f"{ledger_df['Date'].min():%d.%m.%Y} – {last_trade:%d.%m.%Y}): poziții, numerar, realizat, dividende, "
        "impozite, comisioane și XIRR. Tranzacțiile de după ultima dată apar după importul unui export nou. "
        "Totalul în RON e la cursul de azi."
    )
    from_close = sorted(valued.loc[valued["PriceSource"] == ledger_calc.PRICE_CLOSE, "Symbol"])
    if from_close or fx_from_close:
        st.caption("ℹ️ Preț live indisponibil, s-a folosit ultima închidere pentru: "
                   + ", ".join(from_close + [f"curs {cur}/RON" for cur in fx_from_close]) + ".")
    if missing_fx:
        st.warning(f"⚠️ Curs valutar indisponibil pentru: **{', '.join(missing_fx)}**. Totalul în RON nu poate fi calculat.")
    if st.button("🔄 Reîncarcă prețurile manuale", key="manual_prices_reload",
                 help="Citește din nou foaia Preturi_manuale. Fără buton, modificările din foaie apar în cel mult 5 minute."):
        load_manual_price_values.clear()  # golește doar cache-ul foii Preturi_manuale
        st.rerun()
    manual_rows = valued[valued["PriceSource"] == ledger_calc.PRICE_MANUAL]
    if not manual_rows.empty:
        st.caption("ℹ️ Preț trecut de mână (foaia `Preturi_manuale`): " + "; ".join(
            f"{r.Symbol} {_ledger_num(r.Price, 4)}" + (" (fără dată)" if pd.isna(r.PriceDate) else f" din {r.PriceDate:%d.%m.%Y}")
            for r in manual_rows.itertuples(index=False)) + ".")
        age_days = (pd.Timestamp(asof).normalize() - manual_rows["PriceDate"]).dt.days
        stale = sorted(manual_rows.loc[manual_rows["PriceDate"].isna() | (age_days > MANUAL_PRICE_STALE_DAYS), "Symbol"])
        if stale:
            st.warning(f"⚠️ Preț manual fără dată sau mai vechi de {MANUAL_PRICE_STALE_DAYS} de zile pentru: **{', '.join(stale)}**. "
                       "Actualizează foaia `Preturi_manuale`; valoarea de mai jos folosește prețul vechi.")
    missing_prices = sorted(valued.loc[valued["PriceSource"] == ledger_calc.PRICE_MISSING, "Symbol"])
    if missing_prices:
        st.warning(f"⚠️ Fără preț disponibil pentru: **{', '.join(missing_prices)}**. Clasele lor nu au valoare, total și XIRR.")

    order = {name: i for i, name in enumerate(LEDGER_CLASS_ORDER)}
    results = results.assign(_o=results["Class"].map(order).fillna(len(order))).sort_values(["_o", "Currency"])
    st.markdown("**Rezultat pe clase de active** (fiecare în moneda ei)")
    st.dataframe(pd.DataFrame({
        "Clasă": results["Class"] + np.where(results["Proxy"], " *", ""),
        "Monedă": results["Currency"],
        "Poziții": results["OpenPositions"].astype(int),
        "Cost deschis": results["CostBasis"].map(_ledger_num),
        "Valoare azi": results["MarketValue"].map(_ledger_num),
        "Nerealizat": results["Unrealized"].map(_ledger_num),
        "Realizat": results["Realized"].map(_ledger_num),
        "Dividende / cupoane": results["Income"].map(_ledger_num),
        "Impozite": results["Taxes"].map(_ledger_num),
        "Comisioane": results["Fees"].map(_ledger_num),
        "Rezultat total": results["Total"].map(_ledger_num),
        "XIRR (% pe an)": results["XIRR"].map(lambda v: _ledger_num(v, 1, "%")),
        "Fluxuri": results["Flows"].astype(int),
    }), hide_index=True, width='stretch',
        column_config={name: st.column_config.Column(help=text) for name, text in LEDGER_COLUMN_HELP.items()})
    st.caption(
        "Rezultat total = realizat + nerealizat + dividende/cupoane + impozite + comisioane. "
        "XIRR = randament anualizat ponderat cu banii, în moneda clasei; cu puține fluxuri (sub ~20) e orientativ. "
        "Costul e mediu ponderat; acțiunile gratuite au cost 0. "
        "* Obligațiunile și titlurile de stat fără preț în foaia `Preturi_manuale` sunt evaluate la nominal (proxy): "
        "nu există preț de piață în sursele aplicației. Prețul manual e cel cotat, fără dobânda acumulată. "
        f"XIRR nu se afișează când fluxurile acoperă sub {ledger_calc.MIN_XIRR_DAYS} de zile. "
        "Impozite 0 la Tradeville nu înseamnă fără impozit: brokerul raportează dividendele nete, cu impozitul deja reținut."
    )

    render_ledger_total(ledger_df, positions, total_ron, asof)

    with st.expander("Poziții deschise din registru"):
        st.dataframe(pd.DataFrame({
            "Simbol": valued["Symbol"], "Clasă": valued["Class"], "Monedă": valued["Currency"],
            "Cantitate": valued["Quantity"].map(lambda v: format_ro(v, 4, trim=True)),
            "Cost mediu": valued["AvgCost"].map(lambda v: _ledger_num(v, 4)),
            "Preț": valued["Price"].map(lambda v: _ledger_num(v, 4)),
            "Sursă preț": [src if pd.isna(day) else f"{src} ({day:%d.%m.%Y})"
                           for src, day in zip(valued["PriceSource"], valued["PriceDate"])],
            "Valoare": valued["MarketValue"].map(_ledger_num),
            "Nerealizat": valued["Unrealized"].map(_ledger_num),
            "Nerealizat %": valued["UnrealizedPct"].map(lambda v: _ledger_num(v, 1, "%")),
        }), hide_index=True, width='stretch')
    with st.expander("Numerar și curs valutar"):
        st.dataframe(pd.DataFrame({
            "Monedă": by_currency["Currency"],
            "Poziții": by_currency["Positions"].map(_ledger_num),
            "Numerar": by_currency["Cash"].map(_ledger_num),
            "Curs RON": by_currency["Rate"].map(lambda v: _ledger_num(v, 4)),
            "Valoare în RON": by_currency["ValueBase"].map(_ledger_num),
        }), hide_index=True, width='stretch')
        st.dataframe(cash.assign(Cash=cash["Cash"].map(_ledger_num)).rename(columns={"Currency": "Monedă", "Cash": "Numerar"}),
                     hide_index=True, width='stretch')
    if problems:
        with st.expander(f"⚠️ {len(problems)} probleme în registru"):
            for problem in problems[:50]:
                st.write(f"- {problem}")
    return positions


@st.cache_data(ttl=86400, show_spinner=False)
def _yahoo_sector(sym):
    """(sector, quoteType) din Yahoo pentru un simbol, în cache 24 de ore.
    O eroare de rețea nu se salvează în cache (excepția iese din funcție), deci se reîncearcă la următoarea deschidere."""
    _yf_limiter.wait_if_needed()
    info = yf.Ticker(sym).info or {}
    return info.get('sector') or None, info.get('quoteType') or None

def get_portfolio_sectors(df_current):
    """Expunerea pe sectoare a pozițiilor din tab: (tabel Sector/Valoare/Pondere %/Simboluri, fără preț, surse).

    Sectorul: la BVB din harta proprie (analytics/peers.py; Yahoo nu îl trimite), altfel din Yahoo `info`.
    ETF-urile și fondurile sunt „ETF / fond”, nu „Nedefinit”. Calculul e în analytics/portfolio.py (testat).
    """
    if df_current.empty:
        return pd.DataFrame(), [], {}
    values = df_current.groupby('Symbol', sort=False)['MarketValue'].sum(min_count=1).to_dict()
    labels, sources = {}, {}
    for sym in values:
        bvb_sec = bvb_sector(sym) if str(sym).upper().endswith('.RO') else None
        if bvb_sec:
            labels[sym], sources[sym] = pf_calc.sector_label(bvb_sec), "harta BVB"
            continue
        try:
            sec, qtype = _yahoo_sector(sym)
            labels[sym], sources[sym] = pf_calc.sector_label(sec, qtype), "Yahoo"
        except Exception as exc:  # yfinance aruncă tipuri diferite (rețea, JSON, HTTP)
            logger.warning("get_portfolio_sectors %s: %s", sym, exc)
            labels[sym], sources[sym] = pf_calc.SECTOR_UNKNOWN, "eroare Yahoo"
    df_sec, unpriced = pf_calc.sector_weights(values, labels)
    return df_sec, unpriced, sources

# --- CURS VALUTAR FAȚĂ DE LEU (BNR + Yahoo live) ---
FX_LIVE_SYMBOLS = {"EUR": "EURRON=X", "USD": "USDRON=X", "GBP": "GBPRON=X", "CHF": "CHFRON=X"}

@st.cache_data(ttl=3600, show_spinner=False)
def _bnr_xml_frame(url):
    """Un fișier XML BNR → DataFrame (data_bnr.parse_bnr_xml). Eroare HTTP sau fișier fără cursuri → excepție,
    deci un eșec nu se salvează în cache și se reîncearcă la următoarea deschidere."""
    resp = requests.get(url, timeout=15, headers={"User-Agent": "terminal-investitii/1.0 (uz personal)"})
    resp.raise_for_status()
    frame = bnr_data.parse_bnr_xml(resp.content)
    if frame.empty:
        raise ValueError(f"BNR: fișier fără cursuri ({url})")
    return frame

def get_bnr_fx_history():
    """Istoricul cursurilor BNR: anul trecut + anul curent + ultimele 10 zile (pentru variația pe un an și de la 1 ianuarie).
    Fiecare fișier se caută întâi pe curs.bnr.ro (serverul din 2026), apoi pe www.bnr.ro.
    Întoarce (istoric, fișiere care n-au putut fi citite, cu motivul pe server). Istoricul e gol doar dacă n-a mers niciun fișier."""
    year = now_ro().year
    paths = [bnr_data.BNR_YEAR_PATH.format(year=year - 1), bnr_data.BNR_YEAR_PATH.format(year=year), bnr_data.BNR_10DAYS_PATH]
    frames, failed = [], []
    for path in paths:
        reasons = []
        for url in bnr_data.bnr_urls(path):
            try:
                frames.append(_bnr_xml_frame(url))
                break
            except (requests.RequestException, ValueError) as exc:
                logger.warning("BNR %s: %s", url, exc)
                reasons.append(f"{url.split('/')[2]}: {bnr_data.short_error(exc)}")
        else:
            failed.append(f"{path.rsplit('/', 1)[-1]} ({'; '.join(reasons)})")
    return bnr_data.merge_histories(*frames), failed

@st.cache_data(ttl=300, show_spinner=False)
def get_fx_live():
    """Cursul de piață (Yahoo, întârziat câteva minute) pentru valutele principale: dict cod -> lei/unitate.
    Fără niciun curs → excepție (nu se salvează în cache)."""
    prices = get_fast_live_prices(list(FX_LIVE_SYMBOLS.values()))
    out = {code: prices.get(sym) for code, sym in FX_LIVE_SYMBOLS.items()}
    if not any(v for v in out.values()):
        raise ValueError("Yahoo: niciun curs valutar live")
    return out

@st.cache_data(ttl=1800, show_spinner=False)
def get_market_closes():
    """Închiderile zilnice pe 2 ani pentru piețele din analytics/markets.py (MARKETS). Fără niciun simbol → excepție
    (nu se salvează în cache)."""
    syms = [m[2] for m in markets_calc.MARKETS]
    _yf_limiter.wait_if_needed()
    closes = close_frame(yf.download(syms, period="2y", progress=False, auto_adjust=True), syms)
    if closes is None or closes.empty:
        raise ValueError("Yahoo: nicio închidere pentru indici")
    return closes

# --- DOBÂNZI COMPARATE (FRED, BCE, Eurostat) ---
ECB_DATA_URL = "https://data-api.ecb.europa.eu/service/data/{key}"

@st.cache_data(ttl=21600, show_spinner=False)
def _ecb_rate_series(key, n_obs):
    """O serie BCE (procente) ca serie indexată pe perioadă. Eroare sau serie goală → excepție (nu se salvează în cache)."""
    resp = requests.get(ECB_DATA_URL.format(key=key), params={"lastNObservations": n_obs, "format": "csvdata"}, timeout=15)
    resp.raise_for_status()
    ser = parse_ecb_series(resp.text)
    if ser.empty:
        raise ValueError(f"BCE {key}: fără observații")
    return ser

@st.cache_data(ttl=21600, show_spinner=False)
def _fred_rate_series(code, days=800):
    """O serie FRED pe ultimele `days` zile. Eroare sau serie goală → excepție (nu se salvează în cache)."""
    end = datetime.today()
    ser = web.DataReader(code, 'fred', end - timedelta(days=days), end).iloc[:, 0].dropna()
    if ser.empty:
        raise ValueError(f"FRED {code}: fără observații")
    return ser

@st.cache_data(ttl=21600, show_spinner=False)
def _robor3m_series(n_months=30):
    """ROBOR 3M lunar (Eurostat irt_st_m, RO, IRT_M3). Eroare sau serie goală → excepție."""
    resp = requests.get(EUROSTAT_MONEY_MARKET_URL, params={"geo": "RO", "int_rt": "IRT_M3", "freq": "M",
                                                         "lastTimePeriod": n_months}, timeout=15)
    resp.raise_for_status()
    ser = parse_eurostat_monthly(resp.json())
    if ser.empty:
        raise ValueError("Eurostat ROBOR 3M: fără observații")
    return ser

def get_rates_dashboard():
    """Rândurile tabelului „Dobânzi și risc de țară”: (Piață, Indicator, statistici rates_calc.rate_stats, sursă).
    O serie care nu poate fi citită dă un rând N/A cu sursa marcată „indisponibil”, nu o valoare presupusă."""
    def load(fn, *args):
        try:
            return fn(*args)
        except Exception as exc:  # rețea / format: rândul rămâne N/A
            logger.warning("dobânzi %s %s: %s", getattr(fn, "__name__", fn), args, exc)
            return None
    d = rates_calc.to_daily
    m = rates_calc.to_monthly
    us = {c: load(_fred_rate_series, c) for c in ("DFEDTARL", "DFEDTARU", "DFF", "DGS3MO", "DGS2", "DGS10")}
    cpi = load(_fred_rate_series, "CPIAUCSL", 900)
    us_cpi = rates_calc.yoy_monthly(cpi) if cpi is not None else None
    dfr = load(_ecb_rate_series, "FM/B.U2.EUR.4F.KR.DFR.LEV", 600)
    bund = load(_ecb_rate_series, "IRS/M.DE.L.L40.CI.0000.EUR.N.Z", 30)
    # BCE a închis setul ICP pe 04.02.2026 (ultima lună: 12.2025) și l-a înlocuit cu setul HICP (ECOICOP v2,
    # furnizor Eurostat = „4D0”). Se citesc ambele chei și se păstrează seria cu luna cea mai recentă.
    de_hicp = rates_calc.freshest(load(_ecb_rate_series, "HICP/M.DE.N.000000.4D0.ANR", 30),
                                  load(_ecb_rate_series, "ICP/M.DE.N.000000.4.ANR", 30))
    ro10 = load(_ecb_rate_series, "IRS/M.RO.L.L40.CI.0000.RON.N.Z", 30)
    ro_hicp = rates_calc.freshest(load(_ecb_rate_series, "HICP/M.RO.N.000000.4D0.ANR", 30),
                                  load(_ecb_rate_series, "ICP/M.RO.N.000000.4.ANR", 30))
    robor = load(_robor3m_series)

    def row(market, label, ser, source, inputs):
        ok = ser is not None and len(ser) > 0 and all(x is not None for x in inputs)
        st_ = rates_calc.rate_stats(ser) if ok else rates_calc.rate_stats(None)
        stale = rates_calc.stale_note(st_["period"], now_ro().date()) if ok else None
        src = (f"{source} — ⚠️ {stale}" if stale else source) if ok else f"{source} — indisponibil"
        return {"Piață": market, "Indicator": label, **st_, "Sursă": src}

    rows = [
        row("SUA", "Dobânda Fed, interval țintă: limita de jos", d(us["DFEDTARL"]) if us["DFEDTARL"] is not None else None,
            "FRED DFEDTARL, zilnic (decizia FOMC)", [us["DFEDTARL"]]),
        row("SUA", "Dobânda Fed, interval țintă: limita de sus", d(us["DFEDTARU"]) if us["DFEDTARU"] is not None else None,
            "FRED DFEDTARU, zilnic (decizia FOMC)", [us["DFEDTARU"]]),
        row("SUA", "Dobânda Fed (efectivă)", d(us["DFF"]), "FRED DFF, zilnic", [us["DFF"]]),
        row("SUA", "Titluri de stat 3 luni", d(us["DGS3MO"]), "FRED DGS3MO, zilnic", [us["DGS3MO"]]),
        row("SUA", "Titluri de stat 2 ani", d(us["DGS2"]), "FRED DGS2, zilnic", [us["DGS2"]]),
        row("SUA", "Titluri de stat 10 ani", d(us["DGS10"]), "FRED DGS10, zilnic", [us["DGS10"]]),
        row("SUA", "Inflația CPI, an/an", us_cpi, "FRED CPIAUCSL, lunar", [cpi]),
        row("SUA", "Panta 10 ani − 2 ani", rates_calc.daily_diff(us["DGS10"], us["DGS2"]) if us["DGS10"] is not None and us["DGS2"] is not None else None,
            "FRED, aceeași zi", [us["DGS10"], us["DGS2"]]),
        row("SUA", "Dobânda reală 10 ani (10 ani − CPI)", rates_calc.monthly_diff(us["DGS10"], us_cpi) if us["DGS10"] is not None and us_cpi is not None else None,
            "FRED, aceeași lună", [us["DGS10"], us_cpi]),
        row("Zona euro", "Dobânda de depozit BCE", d(dfr) if dfr is not None else None, "BCE FM DFR", [dfr]),
        row("Germania", "Bund 10 ani", m(bund) if bund is not None else None, "BCE IRS, lunar", [bund]),
        row("Germania", "Inflația HICP, an/an", m(de_hicp) if de_hicp is not None else None, "BCE HICP, lunar", [de_hicp]),
        row("Germania", "Dobânda reală Bund (10 ani − HICP)", rates_calc.monthly_diff(bund, de_hicp) if bund is not None and de_hicp is not None else None,
            "BCE, aceeași lună", [bund, de_hicp]),
        row("România", "ROBOR 3 luni", m(robor) if robor is not None else None, "Eurostat irt_st_m, lunar", [robor]),
        row("România", "Titluri de stat 10 ani", m(ro10) if ro10 is not None else None, "BCE IRS, lunar", [ro10]),
        row("România", "Inflația HICP, an/an", m(ro_hicp) if ro_hicp is not None else None, "BCE HICP, lunar", [ro_hicp]),
        row("România", "Dobânda reală 10 ani (10 ani − HICP)", rates_calc.monthly_diff(ro10, ro_hicp) if ro10 is not None and ro_hicp is not None else None,
            "BCE, aceeași lună", [ro10, ro_hicp]),
        row("România", "Prima de risc de țară (RO 10 ani − Bund 10 ani)", rates_calc.monthly_diff(ro10, bund) if ro10 is not None and bund is not None else None,
            "BCE, aceeași lună", [ro10, bund]),
        row("România", "Panta 10 ani − ROBOR 3M", rates_calc.monthly_diff(ro10, robor) if ro10 is not None and robor is not None else None,
            "BCE + Eurostat, aceeași lună", [ro10, robor]),
    ]
    return rows

class FredCalendarUnavailable(Exception):
    """Calendarul FRED nu a putut fi citit. Excepțiile nu intră în cache: eșecul se reîncearcă la următoarea afișare."""


def fred_api_key():
    """Cheia FRED din secrets (`FRED_API_KEY`) sau None. Nu se loghează și nu se afișează niciodată."""
    try:
        key = st.secrets.get("FRED_API_KEY")
    except (FileNotFoundError, KeyError) as exc:   # fără secrets.toml (rulare locală)
        logger.info("FRED_API_KEY indisponibil: %s", type(exc).__name__)
        return None
    key = str(key).strip() if key else ""
    return key or None


@st.cache_data(ttl=21600, show_spinner=False)
def _fred_calendar_cached(start_iso, end_iso, _api_key):
    """Lista `release_dates` din FRED pentru [start, end], inclusiv datele viitoare. Cheia (`_api_key`) nu intră în
    cheia de cache. Erorile aruncă FredCalendarUnavailable cu un mesaj fără cheie (URL-ul cu cheia nu se loghează)."""
    params = {"api_key": _api_key, "file_type": "json", "realtime_start": start_iso, "realtime_end": end_iso,
              "include_release_dates_with_no_data": "true", "sort_order": "asc", "limit": 1000}
    try:
        resp = requests.get(fred_cal.FRED_RELEASE_DATES_URL, params=params, timeout=15)
    except requests.RequestException as exc:
        raise FredCalendarUnavailable(f"FRED nu răspunde ({type(exc).__name__})") from None
    try:
        body = resp.json()
    except ValueError:
        body = None
    if resp.status_code != 200:
        raise FredCalendarUnavailable(fred_cal.error_text(resp.status_code, body, _api_key))
    if not isinstance(body, dict) or not isinstance(body.get("release_dates"), list):
        raise FredCalendarUnavailable("FRED a trimis un răspuns fără lista release_dates")
    return body["release_dates"]


def get_fred_calendar(today):
    """(release_dates, mesaj): lista FRED pe CALENDAR_DAYS zile de la `today`, sau (None, motivul)."""
    key = fred_api_key()
    if key is None:
        return None, "Cheia FRED lipsește din secrets (FRED_API_KEY)."
    start, end = fred_cal.window(today)
    try:
        return _fred_calendar_cached(start.isoformat(), end.isoformat(), key), None
    except FredCalendarUnavailable as exc:
        logger.warning("calendar FRED: %s", exc)
        return None, str(exc)

def get_us_curve_history(days=3700):
    """Panta SUA 10 ani − 3 luni pe ~10 ani (FRED DGS10 − DGS3MO, aceeași zi), pentru graficul din „Dobânzi”.
    Eroare la oricare serie → serie goală; eșecul nu intră în cache (`_fred_rate_series` aruncă excepția)."""
    try:
        return rates_calc.daily_diff(_fred_rate_series("DGS10", days), _fred_rate_series("DGS3MO", days))
    except Exception as exc:  # rețea / FRED
        logger.warning("curba SUA istoric: %s", exc)
        return pd.Series(dtype="float64")

@st.cache_data(ttl=1800, show_spinner=False)
def get_vix_curve_closes():
    """Închiderile ^VIX și ^VIX3M pe un an (Yahoo). Fără niciuna → excepție (nu se salvează în cache)."""
    _yf_limiter.wait_if_needed()
    closes = close_frame(yf.download(["^VIX", "^VIX3M"], period="1y", progress=False, auto_adjust=False), ["^VIX", "^VIX3M"])
    if closes is None or closes.empty:
        raise ValueError("Yahoo: VIX / VIX3M fără date")
    return closes

@st.cache_data(ttl=1800, show_spinner=False)
def get_breadth_closes():
    """Închiderile ajustate RSP, IWM, SPY pe 2 ani (Yahoo). Fără niciuna → excepție (nu se salvează în cache)."""
    syms = sorted({x for a, b, _ in markets_calc.BREADTH_PAIRS for x in (a, b)})
    _yf_limiter.wait_if_needed()
    closes = close_frame(yf.download(syms, period="2y", progress=False, auto_adjust=True), syms)
    if closes is None or closes.empty:
        raise ValueError("Yahoo: RSP/IWM/SPY fără date")
    return closes

CNN_FEAR_GREED_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"


@st.cache_data(ttl=3600, show_spinner=False)
def get_cnn_fear_greed():
    """Indicele Fear & Greed de la CNN (endpoint public neoficial, fără cheie). Excepție la eșec (nu se salvează
    în cache), ca următoarea deschidere să reîncerce; interfața afișează N/A."""
    resp = requests.get(CNN_FEAR_GREED_URL, timeout=10, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "application/json", "Referer": "https://edition.cnn.com/markets/fear-and-greed"})
    resp.raise_for_status()
    parsed = stress_calc.parse_cnn_fear_greed(resp.json())
    if parsed is None:
        raise ValueError("CNN Fear & Greed: răspuns fără scor")
    return parsed


@st.cache_data(ttl=21600, show_spinner=False)
def get_spy_trailing_pe():
    """P/E-ul pe ultimele 12 luni al SPY (Yahoo `info`), proxy pentru S&P 500. Lipsă sau ≤ 0 → excepție (nu se salvează în cache)."""
    _yf_limiter.wait_if_needed()
    pe = num(yf.Ticker("SPY").info or {}, "trailingPE")
    if pe is None or pe <= 0:
        raise ValueError("Yahoo: SPY fără trailingPE")
    return float(pe)

def get_equity_risk_rows():
    """Prima de risc a acțiunilor (proxy) pentru BVB și SUA: list de dict (Piață, rezultatul equity_risk_gap, bază)."""
    rows = []
    # BVB: EPS-ul din foaia BVB la prețul curent al fiecărei companii, mediana; minus titlurile RO 10 ani (BCE, lunar)
    sheet = load_bvb_fundamentals()
    syms = sorted(k for k in sheet if not str(k).startswith("_"))
    prices = get_bvb_peer_prices(tuple(syms)) if syms else {}
    yields = [markets_calc.earnings_yield(sheet[k]["info"].get("trailingEps"), prices.get(k)) for k in syms]
    try:
        ro10 = rates_calc.to_monthly(_ecb_rate_series("IRS/M.RO.L.L40.CI.0000.RON.N.Z", 30))
        ro_bond, ro_when = float(ro10.iloc[-1]), rates_calc.period_label(ro10.index[-1])
    except Exception as exc:  # rețea / BCE
        logger.warning("prima de risc BVB: %s", exc)
        ro_bond, ro_when = None, None
    g = markets_calc.equity_risk_gap(yields, ro_bond)
    rows.append({"Piață": "BVB", **g, "Bază": (f"mediana EPS ÷ prețul curent pe {g['n']} companii din foaia BVB (din {len(syms)}); "
                                               f"titluri de stat RO 10 ani, BCE {ro_when or 'indisponibil'}. Mediană simplă, nu BET ponderat: proxy.")})
    # SUA: 1 ÷ P/E-ul SPY (Yahoo); minus titlurile SUA pe 10 ani (FRED DGS10, ultima zi)
    try:
        spy_ep = [1.0 / get_spy_trailing_pe()]
    except Exception as exc:  # rețea / Yahoo
        logger.warning("prima de risc SUA (P/E): %s", exc)
        spy_ep = []
    try:
        us10 = rates_calc.to_daily(_fred_rate_series("DGS10"))
        us_bond, us_when = float(us10.iloc[-1]), f"{us10.index[-1]:%d.%m.%Y}"
    except Exception as exc:  # rețea / FRED
        logger.warning("prima de risc SUA (10 ani): %s", exc)
        us_bond, us_when = None, None
    g2 = markets_calc.equity_risk_gap(spy_ep, us_bond)
    rows.append({"Piață": "S&P 500", **g2, "Bază": ("1 ÷ P/E-ul pe 12 luni al SPY (Yahoo, " + ("disponibil" if spy_ep else "indisponibil") + "); "
                                                    f"titluri SUA 10 ani, FRED DGS10 {us_when or 'indisponibil'}. Proxy.")})
    return rows

@st.cache_data(ttl=1800, show_spinner=False)
def get_rotation_ratio():
    """Raportul Nasdaq Composite / Dow Jones pe un an (zilele comune). Fără date → excepție (nu se salvează în cache)."""
    _yf_limiter.wait_if_needed()
    closes = close_frame(yf.download(['^IXIC', '^DJI'], period="1y", progress=False, auto_adjust=True), ['^IXIC', '^DJI'])
    both = closes.dropna(how="any") if closes is not None else pd.DataFrame()
    if both.empty or not {'^IXIC', '^DJI'} <= set(both.columns):
        raise ValueError("Yahoo: ^IXIC / ^DJI fără zile comune")
    return both['^IXIC'] / both['^DJI']

# --- FUNCȚIE GLOBAL MARKET ---
GLOBAL_INDICES = {
    'S&P 500': '^GSPC', 'Dow Jones': '^DJI', 'Nasdaq': '^IXIC',
    'DAX (GER)': '^GDAXI', 'FTSE 100 (UK)': '^FTSE', 'Bursa RO (BET) 🇷🇴': 'TVBETETF.RO'
}
GLOBAL_COMMODITIES = {
    'Aur (Gold)': 'GC=F', 'Argint (Silver)': 'SI=F',
    'Petrol (WTI)': 'CL=F', 'Petrol (Brent)': 'BZ=F', 'Copper': 'HG=F', 'Gaz Natural': 'NG=F'
}


@st.cache_data(ttl=300, show_spinner=False)
def _global_quote_closes():
    """Închiderile zilnice NEAJUSTATE pe o lună pentru indicii și mărfurile din „4. Piață Globală”, într-o singură
    cerere Yahoo (înainte: câte un `fast_info` pe simbol). Fără nicio închidere → excepție (nu intră în cache)."""
    syms = list(GLOBAL_INDICES.values()) + list(GLOBAL_COMMODITIES.values())
    _yf_limiter.wait_if_needed()
    closes = close_frame(yf.download(syms, period="1mo", progress=False, auto_adjust=False, group_by="ticker"), syms)
    if closes is None or closes.empty:
        raise ValueError("Yahoo: nicio închidere pentru indici și mărfuri")
    return closes


def get_global_market_data():
    """Tabelele vechi din „4. Piață Globală”: indicii, mărfurile și topurile SUA / Europa.

    Indici și mărfuri: o cerere Yahoo (`_global_quote_closes`, cache 5 min), variația fiecărei serii față de
    închiderea ei precedentă (analytics/briefing.quote_table). Topurile: aceleași date și reguli ca în
    „6. Rezumatul Zilei” (`get_briefing_market`, cache comun de 15 min; ultima ședință a pieței, doar titlurile
    tranzacționate). Înainte: ~230 de cereri `fast_info`, una după alta, la fiecare 5 minute.
    Întoarce un dict: indices, commodities, us_gain, us_lose, us_session, us_partial, eu_gain, eu_lose, eu_session,
    eu_partial. O sursă picată dă tabele goale / N/A, nu valori presupuse."""
    try:
        closes = _global_quote_closes()
    except Exception as exc:  # rețea / Yahoo
        logger.warning("Piață Globală indici și mărfuri: %s", exc)
        closes = pd.DataFrame()
    out = {"indices": briefing_calc.quote_table(closes, GLOBAL_INDICES),
           "commodities": briefing_calc.quote_table(closes, GLOBAL_COMMODITIES)}
    for key in ("us", "eu"):
        try:
            data, partial = get_briefing_market(key)
        except Exception as exc:  # rețea / Yahoo: topul rămâne gol
            logger.warning("Piață Globală topuri %s: %s", key, exc)
            data, partial = {"close": pd.DataFrame(), "volume": pd.DataFrame()}, True
        cols = [x for x in universes.symbols(key) if x in data["close"].columns]
        table, session = briefing_calc.session_table(data["close"][cols], data["volume"].reindex(columns=cols))
        out[f"{key}_gain"], out[f"{key}_lose"] = briefing_calc.top_frames(table)
        out[f"{key}_session"], out[f"{key}_partial"] = session, partial
    return out

@st.cache_data(ttl=300)
def get_sector_performance():
    """Descarcă și calculează performanța zilnică a celor 11 sectoare majore."""
    sectors_map = {
        'XLK': 'Tehnologie', 'XLF': 'Financiar', 'XLE': 'Energie',
        'XLV': 'Sănătate', 'XLY': 'Consum Discreționar',
        'XLP': 'Consum de Bază', 'XLI': 'Industrial',
        'XLU': 'Utilități', 'XLB': 'Materiale',
        'XLRE': 'Imobiliare', 'XLC': 'Comunicații'
    }
    try:
        # Descărcăm datele pe ultimele 5 zile pentru a fi siguri că prindem "ieri" și "azi"
        data = yf.download(list(sectors_map.keys()), period="5d", group_by='ticker', progress=False, auto_adjust=True)
        
        results = []
        for ticker, name in sectors_map.items():
            try:
                # Verificăm dacă structura descărcată este MultiIndex sau simplă
                if isinstance(data.columns, pd.MultiIndex):
                    df_t = data[ticker]['Close'].dropna()
                else:
                    # Dacă din vreo eroare descarcă doar un ticker
                    df_t = data['Close'].dropna()

                if len(df_t) >= 2:
                    curr_price = df_t.iloc[-1]
                    prev_price = df_t.iloc[-2]
                    pct_change = ((curr_price - prev_price) / prev_price) * 100
                    results.append({'Simbol': ticker, 'Sector': name, 'Variație %': pct_change})
            except Exception as exc:  # înainte: except: gol
                logger.warning("get_sector_performance: %s", exc)
                continue
                
        df = pd.DataFrame(results)
        if not df.empty:
            # Sortăm crescător pentru ca pe graficul orizontal (Plotly) cele mai mari să fie sus
            df = df.sort_values(by='Variație %', ascending=True)
        return df
    except Exception as e:
        logger.warning(f"Eroare sectoare: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_credit_risk_data(period="1y"):
    """
    Descarcă datele pentru piața de credit pe o perioadă specificată.
    """
    try:
        data = yf.download(['HYG', 'IEF'], period=period, progress=False, auto_adjust=True)['Close']  # randament total (cu cupoane)
        if 'HYG' in data.columns and 'IEF' in data.columns:
            ratio = data['HYG'] / data['IEF']
            return ratio.dropna()
        return pd.Series()
    except Exception as e:
        logger.warning(f"Eroare date bonduri: {e}")
        return pd.Series()

@st.cache_data(ttl=3600, show_spinner=False)
def get_sector_month_returns():
    """Randamentul pe o lună calendaristică al celor 11 ETF-uri sectoriale SUA (XLK…XLC), pentru pilonul
    „Rotația sectorială” din verdictul macro. dict sector -> randament (fracție); sectoarele fără istoric lipsesc.
    O eroare de rețea nu se salvează în cache (excepția iese din funcție)."""
    sectors_map = {
        'XLK': 'Tehnologie', 'XLF': 'Financiar', 'XLE': 'Energie',
        'XLV': 'Sănătate', 'XLY': 'Consum Discreționar',
        'XLP': 'Consum de Bază', 'XLI': 'Industrial',
        'XLU': 'Utilități', 'XLB': 'Materiale',
        'XLRE': 'Imobiliare', 'XLC': 'Comunicații'
    }
    _yf_limiter.wait_if_needed()
    closes = close_frame(yf.download(list(sectors_map), period="3mo", progress=False, auto_adjust=True), list(sectors_map))
    out = {}
    for tk, name in sectors_map.items():
        if tk in closes.columns:
            r = macro_calc.period_return(closes[tk], months=1)
            if r is not None:
                out[name] = r
    return out

@st.cache_data(ttl=300, show_spinner=False)
def get_vix_last():
    """Ultima închidere ^VIX și data ei; o eroare sau lipsa datelor ridică excepție (nu se salvează în cache)."""
    _yf_limiter.wait_if_needed()
    h = yf.Ticker("^VIX").history(period="10d", auto_adjust=False)['Close'].dropna()
    if h.empty:
        raise ValueError("^VIX fără date")
    return float(h.iloc[-1]), h.index[-1]

@st.cache_data(ttl=3600)
def get_cross_asset_correlation():
    """
    Descarcă ETF-urile majore pentru a calcula corelația claselor de active.
    SPY = Acțiuni, TLT = Obligațiuni (20Y+), GLD = Aur, USO = Petrol, UUP = Dolar
    """
    tickers = {
        'Acțiuni (SPY)': 'SPY', 
        'Bonduri (TLT)': 'TLT', 
        'Aur (GLD)': 'GLD', 
        'Petrol (USO)': 'USO', 
        'Dolar (UUP)': 'UUP'
    }
    try:
        # Descărcăm date pe 3 luni pentru a prinde regimul curent (nu prea vechi, nu prea scurt)
        data = yf.download(list(tickers.values()), period="3mo", progress=False, auto_adjust=True)['Close']
        
        # Redenumim coloanele cu numele noastre intuitive
        rename_map = {v: k for k, v in tickers.items()}
        data = data.rename(columns=rename_map)
        
        # Randamente doar pe zilele comune (cripto se tranzacționează și în weekend): fără zerouri inventate.
        corr_matrix, _ = common_day_correlation(data, min_obs=30)
        return corr_matrix
    except Exception as e:
        logger.warning(f"Eroare Cross-Asset: {e}")
        return pd.DataFrame()

import pandas_datareader.data as web

@st.cache_data(ttl=86400)
def get_fred_macro_data():
    """Descărcare individuală pentru reziliență maximă. Dacă un simbol e picat, restul merg."""
    end = datetime.today()
    # 500 de zile: variația an/an are nevoie de 13 observații lunare, iar FRED publică cu întârziere.
    start = end - timedelta(days=500)
    # Serii exprimate ca indice/nivel, pentru care rata relevantă este variația an/an.
    yoy_codes = ['CPIAUCSL', 'CPILFESL', 'PCEPILFE', 'RSAFS', 'INDPRO']
    
    indicators = {
        'CPIAUCSL': 'CPI General (indice)',
        'CPILFESL': 'Core CPI (indice)',
        'PCEPILFE': 'Core PCE (indice, favorita FED)',
        'UNRATE': 'Rata Șomajului (%)',
        'PAYEMS': 'Nonfarm Payrolls (NFP Oficial)',
        macro_calc.FRED_ADP: 'ADP Private Payrolls',
        'JTSJOL': 'JOLTS (Job Openings)',
        'FEDFUNDS': 'Dobânda Cheie FED (%)',
        'RSAFS': 'Vânzări de Retail (Consum)',
        'INDPRO': 'Producție Industrială',
        'HOUST': 'Housing Starts (Case Noi)',
        'UMCSENT': 'Sentiment Consumatori (U. Mich)'
    }
    
    results = []
    
    # Descarcă fiecare indicator separat pentru a nu bloca tot tabelul
    for code, name in indicators.items():
        try:
            df_item = web.DataReader(code, 'fred', start, end)
            col_data = macro_calc.fred_in_thousands(code, df_item[code].dropna())  # ADP: persoane → mii
            
            if len(col_data) >= 2:
                val_curr = col_data.iloc[-1]
                val_prev = col_data.iloc[-2]
                
                if code in ['PAYEMS', macro_calc.FRED_ADP, 'JTSJOL']:
                    change = val_curr - val_prev
                    trend = f"{ro(change):+.0f}K"
                elif code in ['UNRATE', 'FEDFUNDS', 'UMCSENT']:
                    change = val_curr - val_prev
                    trend = f"{ro(change):+.2f} pct."
                else:
                    change = ((val_curr - val_prev) / val_prev) * 100
                    trend = f"{ro(change):+.2f}%"
                
                yoy = yoy_pct(col_data) if code in yoy_codes else None
                    
                results.append({
                    "Indicator": name,
                    "Valoare Curentă": val_curr,
                    "Lună Precedentă": val_prev,
                    "Evoluție (MoM)": trend,
                    "Evoluție (An/An)": f"{ro(yoy):+.2f}%" if yoy is not None else "—",
                    "_simbol": code,
                    "_change_raw": change,
                    "_yoy": yoy
                })
        except Exception as exc:  # un indicator picat nu oprește tabelul; rândul lui lipsește
            logger.warning("get_fred_macro_data %s: %s", code, exc)
            continue
            
    return pd.DataFrame(results)

def interpret_macro_data_ai(df_macro):
    """Interpretarea Barometrului FED (analytics/macro.py: interpret_fred, testată)."""
    return macro_calc.interpret_fred(df_macro)

# ==================================================================================================
# IDEI ȘI STRATEGII (analytics/ideas.py, data/ucits.py). Propunerea: claude/propunere_tab_idei_2026-11-14.md.
# Decizii Daniel (14.11.2026): totul deodată, risc 1% pe idee, jurnal în foaia nouă `Idei_jurnal`.
# ==================================================================================================
IDEAS_ETF_SYMBOLS = tuple(dict.fromkeys(
    [s for s, _ in ideas_calc.ASSET_CLASSES] + list(ideas_calc.US_SECTORS) + list(ideas_calc.EU_SECTORS)
    + [s for s, _, _ in ideas_calc.REGIONS] + ["USDRON=X", "EURRON=X", "GBPRON=X", "AOR", "ACWI"]
    + ucits_symbols([s for s, _ in ideas_calc.ASSET_CLASSES] + list(ideas_calc.US_SECTORS))))
IDEAS_HELP = {
    "capital": "Valoarea portofoliului pe care o folosești ca referință. Cantitatea propusă = 1% din capital ÷ "
               "(prețul de intrare − stopul), convertit în lei la cursul de piață; plafon: poziția ≤ 25% din capital. "
               "Cu 0, fișele arată doar stopul.",
    "risk": "Pierderea maximă acceptată pe o idee dacă stopul e atins: 1% din capital (decizia ta, 14.11.2026). "
            "Golurile de preț peste stop (după rezultate, peste weekend) pot depăși acest procent.",
    "growth": "Indicele Chicago Fed al activității naționale, media pe 3 luni (FRED CFNAIMA3): 85 de serii lunare. "
              "Peste 0 = economia SUA crește peste trend; sub −0,70 a însoțit istoric recesiunile. "
              "Între −0,10 și +0,10: „la graniță”, fără cadran (prag netestat). Publicat cu ~3 săptămâni întârziere.",
    "inflation": "Inflația de bază SUA (FRED CPILFESL, fără alimente și energie): ritmul pe ultimele 3 luni anualizat, "
                 "(CPI_t / CPI_t−3)^4 − 1, minus ritmul pe 12 luni, CPI_t / CPI_t−12 − 1. Pozitiv = inflația "
                 "accelerează. Sub ±0,25 pp: „la graniță” (prag netestat).",
    "stress": "VIX ÷ VIX3M: frica pe 30 de zile față de cea pe 3 luni. Peste 1 = structură inversată (panică pe termen "
              "scurt). Un episod începe la prima zi ≥ 1 după cel puțin 20 de ședințe sub 1. Randamentul ulterior e al "
              "SPY (cu dividende), pe 3 și 6 luni calendaristice; comparat cu media tuturor zilelor din aceeași perioadă.",
}


def _pct_txt(x, decimals=1, sign=True):
    """Fracție → „+12,3%” (format românesc); lipsă → „N/A”."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "N/A"
    return (("+" if sign and x > 0 else "") + format_ro(x * 100, decimals)) + "%"


def _months_txt(n):
    return "N/A" if n is None else ("1 lună" if n == 1 else f"{n} luni")


def _days_txt(n):
    return "azi" if n == 0 else "mâine" if n == 1 else f"peste {n} zile"


def _ideas_fetch_error(what, exc):
    logger.warning("Idei și strategii, %s: %s", what, exc)
    return f"{what}: date indisponibile acum ({type(exc).__name__})"


@st.cache_data(ttl=3600, show_spinner=False)
def get_ideas_prices(symbols, period="3y"):
    """Close / High / Low / Volume AJUSTATE (dividende incluse) pentru simbolurile date (tuplu), într-o singură cerere
    Yahoo. dict câmp → DataFrame (index fără fus orar). Fără nicio închidere → excepție (nu intră în cache)."""
    syms = list(symbols)
    _yf_limiter.wait_if_needed()
    data = yf.download(syms, period=period, progress=False, auto_adjust=True, group_by="column")
    out = {f: _naive_index(field_frame(data, f, syms)) for f in ("Close", "High", "Low", "Volume")}
    if out["Close"].empty:
        raise ValueError("Yahoo: nicio închidere pentru simbolurile cerute")
    return out


@st.cache_data(ttl=86400, show_spinner=False)
def get_stress_history():
    """^VIX, ^VIX3M și SPY (ajustat) pe tot istoricul Yahoo (VIX3M din 2007). Fără date → excepție."""
    syms = ["^VIX", "^VIX3M", "SPY"]
    _yf_limiter.wait_if_needed()
    closes = _naive_index(close_frame(yf.download(syms, period="max", progress=False, auto_adjust=True), syms))
    if closes is None or closes.empty:
        raise ValueError("Yahoo: VIX / VIX3M / SPY fără date")
    return closes


@st.cache_data(ttl=86400, show_spinner=False)
def get_stress_stats():
    """Statistica episoadelor de stres (analytics/ideas.stress_episodes) pe istoricul din `get_stress_history`,
    în cache o zi: calculul parcurge ~4.500 de zile și nu trebuie refăcut la fiecare clic. Eroarea urcă (fără cache)."""
    sh = get_stress_history()
    return ideas_calc.stress_episodes(sh.get("^VIX"), sh.get("^VIX3M"), sh.get("SPY"))


@st.cache_data(ttl=86400, show_spinner=False)
def _etf_trailing_pe(sym):
    """P/E-ul pe ultimele 12 luni raportat de Yahoo pentru un ETF (media ponderată a deținerilor) sau None."""
    _yf_limiter.wait_if_needed()
    return num(yf.Ticker(sym).info or {}, "trailingPE")


@st.cache_data(ttl=43200, show_spinner=False, max_entries=200)
def get_ideas_stock_detail(symbol):
    """Datele unei acțiuni preselectate (cache 12 h): nume, sector, monedă, ultimul și următorul anunț de rezultate,
    surpriza, revizuirea estimării EPS pe 90 de zile, Altman (zona), Piotroski, datorie netă / EBITDA.
    Din situațiile financiare, cu aceleași funcții ca în Analiză Companie. Valorile lipsă sunt None."""
    t = yf.Ticker(symbol)
    info = _safe_info(t)
    errors = []
    try:
        ed = t.get_earnings_dates(limit=8)
    except Exception as exc:  # yfinance: rețea, HTTP, format; metoda lipsește la unele versiuni
        logger.warning("idei, rezultate %s: %s", symbol, exc)
        ed = None
        errors.append("datele anunțurilor de rezultate")
    try:
        et = t.eps_trend
    except Exception as exc:  # yfinance: rețea, HTTP, format
        logger.warning("idei, eps_trend %s: %s", symbol, exc)
        et = None
        errors.append("tendința estimărilor EPS")
    today = pd.Timestamp(now_ro().date())
    ann, surprise, nxt = ideas_calc.last_announcement(ed, today)
    rev, rev_period = ideas_calc.eps_revision(et)
    fin = get_financial_statements(symbol)
    financial = fund.is_financial_issuer(info.get("sector"), symbol)
    altman = calculate_altman_z(info, symbol, fin)
    pio = fund.piotroski(fin.get("income"), fin.get("balance"), fin.get("cashflow"), financial=financial)
    lev = fund.leverage_ratios(fin.get("income"), fin.get("balance"), fin.get("q_income"), fin.get("q_balance"))
    return {"name": info.get("shortName") or info.get("longName") or symbol, "sector": info.get("sector"),
            "currency": info.get("currency"), "quote_type": info.get("quoteType"),
            "ann_date": ann, "surprise": surprise, "next_date": nxt, "revision": rev, "rev_period": rev_period,
            "altman_zone": altman["zone"], "altman_value": altman["value"], "altman_variant": altman["variant"],
            "pio_passed": pio["passed"], "pio_evaluable": pio["evaluable"],
            "nd_ebitda": lev["net_debt_to_ebitda"], "financial": bool(financial),
            "errors": errors + [f"situația {e}" for e in fin.get("errors", [])], "at": now_ro()}


@st.cache_data(ttl=300, show_spinner=False)
def load_ideas_journal_values():
    """Conținutul brut al foii `Idei_jurnal`; None dacă foaia nu există încă sau nu poate fi citită."""
    ws = connect_to_gsheets(ideas_calc.JOURNAL_SHEET)
    if ws is None:
        return None
    try:
        return ws.get_all_values(value_render_option="UNFORMATTED_VALUE", date_time_render_option="FORMATTED_STRING")
    except gspread.exceptions.GSpreadException as e:
        logger.warning(f"foaia {ideas_calc.JOURNAL_SHEET} nu a putut fi citită: {e}")
        return None


def save_idea_to_journal(idea):
    """Scrie ideea în `Idei_jurnal` (o dată pe zi pentru același simbol și aceeași strategie). Foaia se creează la
    prima scriere, cu antetul JOURNAL_COLUMNS (aprobat de Daniel, 14.11.2026). Întoarce (ok, mesaj)."""
    today = now_ro().date()
    strat = ideas_calc.strategy_name(idea["strategy"])
    if idea.get("entry") is None or idea.get("stop") is None:
        return False, "Fără preț de intrare sau fără stop, ideea nu se poate salva."
    values = load_ideas_journal_values()
    df, _ = ideas_calc.parse_journal(values, portfolio_sheet.cell_number)
    if ideas_calc.already_logged(df, today, idea["symbol"], strat):
        return False, f"{idea['symbol']} ({strat}) e deja în jurnal azi."
    row = ideas_calc.journal_row(today, idea["symbol"], idea.get("exec_symbol"), strat, idea["entry"], idea["stop"],
                                 ideas_calc.STRATEGIES[idea["strategy"]]["horizon"], idea.get("benchmark"),
                                 idea.get("signals", ""))
    try:
        ws = connect_to_gsheets(ideas_calc.JOURNAL_SHEET)
        if ws is None:
            book = _get_spreadsheet()
            if book is None:
                return False, "portofoliu_db nu e accesibil; ideea nu a fost salvată."
            ws = book.add_worksheet(title=ideas_calc.JOURNAL_SHEET, rows=1000, cols=len(ideas_calc.JOURNAL_COLUMNS))
            ws.append_row(ideas_calc.JOURNAL_COLUMNS, value_input_option="RAW")
        ws.append_row(row, value_input_option="RAW")
    except (gspread.exceptions.GSpreadException, requests.RequestException, AttributeError) as exc:
        logger.warning("Idei_jurnal, scriere eșuată pentru %s: %s", idea["symbol"], exc)
        return False, "Scrierea în Idei_jurnal a eșuat; ideea nu a fost salvată."
    load_ideas_journal_values.clear()   # doar cache-ul jurnalului
    return True, (f"Salvată în {ideas_calc.JOURNAL_SHEET}: {idea['symbol']}, {strat}, intrare "
                  f"{price_ro(idea['entry'])}, stop {price_ro(idea['stop'])}, orizont "
                  f"{ideas_calc.STRATEGIES[idea['strategy']]['horizon']} zile.")


def _ideas_last(px, sym):
    try:
        s = px["Close"][sym].dropna()
    except (KeyError, TypeError):
        return None
    return float(s.iloc[-1]) if len(s) else None


def _ideas_close(px, sym):
    try:
        return px["Close"][sym].dropna()
    except (KeyError, TypeError):
        return pd.Series(dtype=float)


def ideas_atr_stop(px, sym):
    """Stopul de referință: 2,5 × ATR (Wilder, 14) sub închidere, mobil (analytics/technical.atr_trailing_stop)."""
    try:
        df = pd.DataFrame({"High": px["High"][sym], "Low": px["Low"][sym], "Close": px["Close"][sym]}).dropna()
    except (KeyError, TypeError):
        return None
    if len(df) < 30:
        return None
    v = atr_trailing_stop(df, window=14, multiplier=ideas_calc.ATR_MULTIPLIER)["ATR_Stop"].iloc[-1]
    return float(v) if pd.notna(v) and v > 0 else None


def ideas_cash(currency):
    """(randamentul numerarului pe 12 luni, sursa) — USD: FRED DGS3MO; EUR: rata BCE la depozite (DFR). Proxy."""
    try:
        if currency == "USD":
            return ideas_calc.cash_return_12m(_fred_rate_series("DGS3MO", 500)), "titluri SUA la 3 luni (FRED DGS3MO)"
        ser = _ecb_rate_series("FM/B.U2.EUR.4F.KR.DFR.LEV", 600).copy()
        ser.index = pd.to_datetime(ser.index, errors="coerce")
        return ideas_calc.cash_return_12m(ser[ser.index.notna()]), "rata BCE la depozite (DFR)"
    except Exception as exc:  # FRED / BCE: rețea, HTTP, serie goală
        return None, _ideas_fetch_error(f"numerar {currency}", exc)


def ideas_quadrant():
    """(cadranul, mesaj de eroare sau None) din FRED CFNAIMA3 și CPILFESL."""
    series, msgs = {}, []
    for code in ("CFNAIMA3", "CPILFESL"):
        try:
            series[code] = _fred_rate_series(code, 1500)
        except Exception as exc:  # FRED: rețea, cod inexistent, serie goală
            series[code] = None
            msgs.append(_ideas_fetch_error(f"FRED {code}", exc))
    return ideas_calc.growth_inflation_quadrant(series["CFNAIMA3"], series["CPILFESL"]), "; ".join(msgs) or None


def ideas_events(today):
    """(publicările cu impact mare în următoarele 14 zile, mesaj) din calendarul FRED + Fed."""
    cal_data, cal_msg = get_fred_calendar(today)
    if cal_data is None:
        rows = [{"date": d, "weekday": fred_cal.WEEKDAYS_RO[d.weekday()], "in_days": (d - today).days,
                 "label": fred_cal.FOMC_LABEL, "key": "fomc"} for d in fred_cal.fomc_in_window(today)]
        return rows, f"calendarul FRED indisponibil ({cal_msg}); doar FOMC din calendarul Fed"
    rows, _ = fred_cal.upcoming(cal_data, today)
    return ideas_calc.high_impact_events(rows), None


def ideas_holdings():
    """Simbolurile din foaia `Portofoliu` (litere mari); mulțime goală dacă foaia nu se poate citi."""
    df = load_portfolio()
    if df is None or df.empty or "Symbol" not in df.columns:
        return set()
    return set(df["Symbol"].astype(str).str.upper())


def render_idea(idea, ctx):
    """Fișa unei idei, într-un expander: teza, pașii (fundamental și tehnic separat), execuția, invalidarea,
    dimensionarea, evenimentele, baza și butonul de jurnal."""
    strat = ideas_calc.STRATEGIES[idea["strategy"]]
    badge = {"aliniat": " · aliniat cu cadranul", "contra": " · contra cadranului"}.get(idea.get("alignment"), "")
    held = " · deții deja" if idea.get("held") else ""
    with st.expander(f"{idea['symbol']} — {idea['name']} · {strat['name']}{badge}{held}"):
        st.markdown(f"**Teza.** {idea['thesis']}")
        for label, text in idea["steps"]:
            st.markdown(f"- **{label}:** {text}")
        entry, stop = idea.get("entry"), idea.get("stop")
        stop_pct = (stop / entry - 1) if (entry and stop) else None
        # Execuție
        ex_sym, ex_desc = idea.get("exec_symbol"), idea.get("exec_desc", "")
        check = idea.get("exec_check")
        chk_txt = ""
        if check is not None:
            if check.get("corr") is None:
                chk_txt = f" Verificare: Yahoo fără destule date comune ({check.get('n', 0)} săptămâni)."
            else:
                chk_txt = (f" Verificare: corelația randamentelor săptămânale cu {idea['symbol']} pe 12 luni "
                           f"{format_ro(check['corr'], 2)} ({check['n']} săptămâni)"
                           + ("." if check["ok"] else " — sub 0,80: verifică indicele la emitent."))
        st.markdown(f"- **Execuție:** {ex_sym or 'N/A'} — {ex_desc}.{chk_txt}")
        # Invalidare
        st.markdown(f"- **Invalidare:** {strat['exit']}. Stop de referință (2,5 × ATR): {price_ro(stop)} "
                    f"({_pct_txt(stop_pct)} față de {price_ro(entry)}).")
        # Dimensionare: pe instrumentul cumpărabil, cu aceeași distanță procentuală până la stop
        capital = ctx["capital"]
        ex_price, ex_cur = idea.get("exec_price"), idea.get("exec_currency")
        if not capital:
            size_txt = "introdu capitalul de referință sus pentru cantitate."
        elif ex_price is None or stop_pct is None:
            size_txt = "N/A (lipsește prețul instrumentului sau stopul)."
        else:
            fx = 1.0 if ex_cur == "RON" else (ctx["fx"].get("GBP") / 100 if ex_cur == "GBp" and ctx["fx"].get("GBP")
                                              else ctx["fx"].get(ex_cur))
            sz = ideas_calc.position_size(capital, ex_price, ex_price * (1 + stop_pct), fx)
            if sz["qty"] is None:
                size_txt = f"N/A ({sz['reason']})."
            else:
                size_txt = (f"{sz['qty']} × {ex_sym} la {price_ro(ex_price)} {ex_cur} ≈ {format_ro(sz['value_ron'], 0)} lei "
                            f"({_pct_txt(sz['weight'], 1, sign=False)} din capital); pierdere la stop ≈ "
                            f"{format_ro(sz['risk_ron'], 0)} lei" + (" (plafonat la 25% din capital)." if sz["capped"] else "."))
        st.markdown(f"- **Dimensionare (risc 1%):** {size_txt}")
        # Evenimente
        ev = list(ctx["events_txt"])
        ed = idea.get("earnings_days")
        if ed is not None and 0 <= ed <= ideas_calc.EARNINGS_SOON_DAYS:
            ev.insert(0, f"⚠️ rezultate trimestriale peste {ed} zile: prețul poate sări peste stop")
        st.markdown(f"- **Evenimente (14 zile):** {'; '.join(ev) if ev else 'nimic cu impact mare'}.")
        for title, link in idea.get("news", []):
            st.markdown(f"  - [{title}]({link})" if link else f"  - {title}")
        base = strat.get("test") or ("Testat în aplicație" if idea["strategy"] == "bvb" else ideas_calc.UNTESTED.capitalize())
        st.caption(f"Baza: {strat['evidence']}. {base}. Orizont: {strat['horizon']} de zile; benchmark în jurnal: "
                   f"{idea.get('benchmark') or 'N/A'}.")
        if st.button("📓 Salvează în jurnal", key=f"idea_save_{idea['strategy']}_{idea['symbol']}"):
            ok, msg = save_idea_to_journal(idea)
            (st.success if ok else st.warning)(msg)


def _ideas_exec(px, sym):
    """(simbol cumpărabil, descriere, preț, valută, verificarea corelației) pentru un ETF semnal."""
    ex_sym, ex_desc = ucits_for(sym)
    if ex_sym is None:
        return None, ex_desc, None, None, None
    # Moneda liniei de cotare din Yahoo (cache 7 zile): la Londra același ETF poate cota în USD sau în pence (GBp);
    # dedusă din sufix, dimensionarea ieșea de 100 de ori greșită (SJPA.L, SGLN.L, captura din 14.11).
    try:
        cur = _yahoo_currency(ex_sym) or ucits_currency(ex_sym)
    except (requests.RequestException, KeyError, ValueError, AttributeError, TypeError) as exc:
        logger.warning("idei, moneda %s: %s", ex_sym, exc)
        cur = ucits_currency(ex_sym)
    # Corelația în USD (ETF-ul semnal e în USD): fără conversie, cursul EUR/USD cobora corelația la obligațiuni
    # (CBU0.DE 0,70 în captura din 15.11), deși instrumentul e corect.
    ucits_usd = ideas_calc.in_usd(_ideas_close(px, ex_sym), cur, _ideas_close(px, "USDRON=X"),
                                  _ideas_close(px, "EURRON=X"), _ideas_close(px, "GBPRON=X"))
    check = None if ex_sym == sym else ideas_calc.tracking_check(_ideas_close(px, sym), ucits_usd)
    return ex_sym, ex_desc, _ideas_last(px, ex_sym), cur, check


def _ideas_etf_idea(px, ctx, strategy, sym, name, thesis, steps, signals):
    ex_sym, ex_desc, ex_price, ex_cur, check = _ideas_exec(px, sym)
    return {"strategy": strategy, "symbol": sym, "name": name, "thesis": thesis, "steps": steps,
            "entry": _ideas_last(px, sym), "stop": ideas_atr_stop(px, sym), "exec_symbol": ex_sym,
            "exec_desc": ex_desc, "exec_price": ex_price, "exec_currency": ex_cur, "exec_check": check,
            "alignment": ideas_calc.alignment(sym, ctx["quad"]["key"]),
            "held": bool({sym, ex_sym} & ctx["holdings"]),
            "benchmark": ideas_calc.STRATEGIES[strategy]["benchmark"], "signals": signals}


def _align_txt(sym, quad):
    a = ideas_calc.alignment(sym, quad["key"])
    if a is None:
        return "fără cadran (date lipsă sau la graniță)"
    return {"aliniat": f"aliniat cu {quad['key']} ({quad['label']})", "contra": f"contra cadranului {quad['key']}",
            "neutru": f"neutru față de {quad['key']}"}[a]


def render_ideas_regime(quad, quad_msg, events, ev_msg, ctx):
    st.subheader("🧭 Cadranul creștere × inflație (SUA)")
    c1, c2 = st.columns(2)
    c1.metric("Creștere: CFNAI-MA3", format_ro(quad["growth"], 2), help=IDEAS_HELP["growth"])
    c2.metric("Accelerarea inflației de bază", "N/A" if quad["accel"] is None else f"{format_ro(quad['accel'], 2)} pp",
              help=IDEAS_HELP["inflation"])
    dates = (f"CFNAI: {quad['growth_date']:%m.%Y}" if quad["growth_date"] is not None else "CFNAI: N/A") + "; " + \
            (f"CPI de bază: {quad['inflation_date']:%m.%Y} (3 luni anualizat {_pct_txt(quad['ann3'], 2)}, "
             f"12 luni {_pct_txt(quad['yoy'], 2)})" if quad["inflation_date"] is not None else "CPI de bază: N/A")
    if quad["key"]:
        st.markdown(f"**{quad['key']} — {quad['label']}** (de {_months_txt(quad['months'])}). Pe acest cadran, {quad['text']}.")
    else:
        st.markdown("**Fără cadran:** cel puțin o axă lipsește sau e la graniță; ideile nu primesc etichetă de aliniere.")
    st.caption(f"Ultimele date: {dates}. Înclinarea pe cadrane e o medie istorică (Ilmanen 2011; Neville et al. 2021), "
               "orientativă și netestată în aplicație; 2022 (acțiuni și obligațiuni în scădere simultan) arată limitele."
               + (f" {quad_msg}." if quad_msg else ""))

    st.subheader("🌡️ Episoade de stres (VIX / VIX3M)")
    try:
        st_res = get_stress_stats()
    except Exception as exc:  # yfinance: rețea, HTTP, date lipsă
        st.info(_ideas_fetch_error("istoricul VIX", exc))
        st_res = None
    if st_res and st_res["ratio"] is not None:
        state = "STRUCTURĂ INVERSATĂ (stres)" if st_res["in_stress"] else "normal (contango)"
        st.metric("VIX ÷ VIX3M acum", format_ro(st_res["ratio"], 2), state, delta_color="off", help=IDEAS_HELP["stress"])
        rows = []
        for months, s in st_res["stats"].items():
            rows.append({"Orizont": f"{months} luni", "Episoade (N)": s["n"],
                         "SPY după episod, medie": _pct_txt(s["mean"]), "Pozitive": _pct_txt(s["pos"], 0, sign=False),
                         "SPY toate zilele, medie": _pct_txt(s["base_mean"]),
                         "Toate zilele, pozitive": _pct_txt(s["base_pos"], 0, sign=False)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        last = st_res["last_episode"]
        st.caption((f"Ultimul episod: {last:%d.%m.%Y}. " if last is not None else "") +
                   ("Statistică descriptivă, în eșantion, fără costuri." if st_res["enough"] else
                    f"Eșantion mic (N = {st_res['n']} < {ideas_calc.STRESS_MIN_N}): fără verdict."))

    st.subheader("🌍 Regimul piețelor")
    try:
        mk_closes = get_market_closes()
    except Exception as exc:  # yfinance
        mk_closes = None
        st.info(_ideas_fetch_error("indicii", exc))
    if mk_closes is not None:
        rows = []
        for _key, name, sym, _cur, _note in markets_calc.MARKETS:
            r = markets_calc.market_regime(mk_closes[sym] if sym in mk_closes.columns else None)
            rows.append({"Piață": name, "Trend (din 3)": "N/A" if r["trend_n"] is None else f"{r['trend_n']}/3",
                         "Momentum 12-1": _pct_txt(r["momentum"]), "Față de maximul pe 52 săpt.": _pct_txt(r["from_high"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        st.caption("Trend: preț > SMA 50, preț > SMA 200, SMA 50 > SMA 200. Verdictul macro complet (credit, VIX, "
                   "dolar, sectoare, curbă, sentiment): secțiunea Piață globală.")

    st.subheader("📅 Evenimente cu impact mare (14 zile)")
    if events:
        for r in events:
            st.markdown(f"- {r['weekday']} {r['date']:%d.%m} ({_days_txt(r['in_days'])}): {r['label']}")
    else:
        st.markdown("Nicio publicare FOMC / CPI / NFP / PCE în următoarele 14 zile.")
    if ev_msg:
        st.caption(ev_msg)


def render_ideas_etf(px, px_err, ctx):
    quad = ctx["quad"]
    if px is None:
        st.warning(px_err or "Prețurile ETF indisponibile.")
        return
    cash_usd, cash_usd_src = ctx["cash"]["USD"]
    cash_eur, cash_eur_src = ctx["cash"]["EUR"]
    st.caption(f"Numerar pe 12 luni (proxy, media randamentelor): USD {_pct_txt(cash_usd, 2, sign=False)} "
               f"({cash_usd_src}); EUR {_pct_txt(cash_eur, 2, sign=False)} ({cash_eur_src}). Prețuri ajustate cu "
               "dividendele; semnalul se calculează pe ETF-ul din SUA, execuția pe echivalentul UCITS.")

    # A. Trend pe clase de active
    st.subheader("A. Trend pe clase de active")
    rows, trend_ideas = [], []
    for sym, name in ideas_calc.ASSET_CLASSES:
        t = ideas_calc.trend_signal(_ideas_close(px, sym), cash_usd)
        rows.append({"Clasa": name, "Simbol": sym, "Stare": t["state"] or "N/A", "12 luni": _pct_txt(t["r12"]),
                     "Peste media pe 10 luni": "N/A" if t["above_sma10m"] is None else
                     (f"da, de {_months_txt(t['months_above'])}" if t["above_sma10m"] else "nu"),
                     "Cadran": _align_txt(sym, quad), "UCITS": ucits_for(sym)[0] or "—"})
        if t["state"] == "ÎN TREND":
            thesis = (f"{name} e în trend: {_pct_txt(t['r12'])} pe 12 luni, peste numerar ({_pct_txt(cash_usd, 2, sign=False)}), "
                      f"și peste media închiderilor lunare pe 10 luni de {_months_txt(t['months_above'])}.")
            steps = [("Fundamental", "nu se aplică (clasă de active); evaluarea piețelor: tabul ETF-uri, C"),
                     ("Tehnic", f"închidere {price_ro(t['last'])} față de media pe 10 luni {price_ro(t['sma10m'])}; "
                                f"12 luni {_pct_txt(t['r12'])}"),
                     ("Regim", _align_txt(sym, quad))]
            sig = f"trend: 12l {_pct_txt(t['r12'])} vs numerar {_pct_txt(cash_usd, 2, sign=False)}; peste SMA10l {_months_txt(t['months_above'])}; {quad['key'] or 'fără cadran'}"
            trend_ideas.append(_ideas_etf_idea(px, ctx, "trend", sym, name, thesis, steps, sig))
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    st.caption("ÎN TREND: randament pe 12 luni > numerar ȘI închiderea > media ultimelor 10 închideri lunare. "
               "IEȘIT: ambele false. MIXT: una din două. Regula reduce scăderile mari, nu crește neapărat randamentul.")
    for idea in sorted(trend_ideas, key=lambda i: {"aliniat": 0, "neutru": 1, None: 1, "contra": 2}[i["alignment"]]):
        render_idea(idea, ctx)

    # B. Rotația sectoarelor: doar tabel informativ (decizia lui Daniel, 14.11.2026, după testul istoric 2000–2026:
    # rotația SUA nu a bătut SPY pe ansamblu; cea UE e netestată). Fără fișe de idei; strategiile rămân în
    # ideas.STRATEGIES pentru rândurile deja salvate în Idei_jurnal.
    for title, universe, cash, strategy in (("B. Puterea sectoarelor SUA (momentum dublu, informativ)", ideas_calc.US_SECTORS, cash_usd, "sector_us"),
                                            ("B. Puterea sectoarelor Europa (STOXX 600, informativ)", ideas_calc.EU_SECTORS, cash_eur, "sector_eu")):
        st.subheader(title)
        tab = ideas_calc.dual_momentum_table({s: _ideas_close(px, s) for s in universe}, cash)
        view = pd.DataFrame({
            "Loc": [("N/A" if pd.isna(r) else f"{int(r)}") for r in tab["rank"]],
            "Sector": [universe[s] for s in tab["symbol"]], "Simbol": tab["symbol"],
            "Momentum 12-1": [_pct_txt(x) for x in tab["mom"]], "12 luni": [_pct_txt(x) for x in tab["r12"]],
            "Peste SMA 200": ["N/A" if x is None else ("da" if x else "nu") for x in tab["above200"]],
            "Bate numerarul": ["N/A" if x is None else ("da" if x else "nu") for x in tab["beats_cash"]],
            "Primele 3": ["✅" if x else "" for x in tab["selected"]]})
        st.dataframe(view, hide_index=True, width='stretch')
        missing = [s for s, m in zip(tab["symbol"], tab["mom"]) if m is None]
        if missing:
            st.caption("Fără istoric suficient la Yahoo (simbol de verificat): " + ", ".join(missing) + ".")
        st.caption("Context, nu idee de tranzacționare: arată unde e puterea pieței (util la analiza unei acțiuni din "
                   "sector). " + (ideas_calc.STRATEGIES["sector_us"]["test"] + "." if strategy == "sector_us" else
                                  "Netestat; pe SUA, aceeași regulă nu a adus randament în plus față de piață."))
    st.caption("EU: simbolurile sectoarelor STOXX 600 nu au fost verificate din mediul de dezvoltare; cele fără date "
               "apar mai sus.")

    # C. Regiuni
    st.subheader("C. Regiuni (în USD; ordinea e aceeași în lei)")
    reg_closes = {}
    for sym, _name, cur in ideas_calc.REGIONS:
        c = _ideas_close(px, sym)
        reg_closes[sym] = ideas_calc.to_usd(c, _ideas_close(px, "USDRON=X")) if cur == "RON" else c
    tab = ideas_calc.dual_momentum_table(reg_closes, cash_usd, top=1)
    names = {s: n for s, n, _ in ideas_calc.REGIONS}
    pes = {}
    for sym in tab["symbol"]:
        if sym.endswith(".RO"):
            pes[sym] = None
            continue
        try:
            pes[sym] = _etf_trailing_pe(sym)
        except Exception as exc:  # yfinance: rețea, HTTP
            logger.warning("idei, P/E %s: %s", sym, exc)
            pes[sym] = None
    st.dataframe(pd.DataFrame({
        "Loc": [("N/A" if pd.isna(r) else f"{int(r)}") for r in tab["rank"]],
        "Regiune": [names[s] for s in tab["symbol"]], "Simbol": tab["symbol"],
        "Momentum 12-1": [_pct_txt(x) for x in tab["mom"]], "12 luni": [_pct_txt(x) for x in tab["r12"]],
        "Filtru absolut": ["N/A" if x is None else ("trece" if x else "nu trece") for x in tab["passes"]],
        "E/P (1 ÷ P/E Yahoo)": [_pct_txt(1 / pes[s], 1, sign=False) if pes.get(s) else "N/A" for s in tab["symbol"]],
        "Idee": ["✅" if x else "" for x in tab["selected"]]}), hide_index=True, width='stretch')
    st.caption("E/P: randamentul profiturilor (P/E-ul ponderat al deținerilor, raportat de Yahoo) — informativ, nu intră "
               "în clasament; mare = piață mai ieftină. România: TVBETETF convertit în USD cu USDRON.")
    for _, r in tab[tab["selected"]].iterrows():
        sym = r["symbol"]
        ep = _pct_txt(1 / pes[sym], 1, sign=False) if pes.get(sym) else "N/A"
        thesis = (f"{names[sym]} are cel mai mare momentum 12-1 dintre regiuni ({_pct_txt(r['mom'])}, în USD) și trece "
                  "filtrul absolut.")
        steps = [("Fundamental", f"E/P {ep} (informativ)"),
                 ("Tehnic", f"momentum 12-1 {_pct_txt(r['mom'])}; 12 luni {_pct_txt(r['r12'])}"),
                 ("Regim", _align_txt(sym, quad))]
        render_idea(_ideas_etf_idea(px, ctx, "region", sym, names[sym], thesis, steps,
                                    f"regiune loc 1; mom12-1 {_pct_txt(r['mom'])}; E/P {ep}"), ctx)

    with st.expander("Echivalentele UCITS: verificarea automată"):
        rows = []
        for sym in [s for s, _ in ideas_calc.ASSET_CLASSES] + list(ideas_calc.US_SECTORS):
            ex_sym, ex_desc, ex_price, ex_cur, check = _ideas_exec(px, sym)
            rows.append({"Semnal": sym, "UCITS": ex_sym or "—", "Descriere": ex_desc,
                         "Ultimul preț": price_ro(ex_price), "Monedă": ex_cur or "N/A", "Corelație săpt. 12 luni (în USD)":
                         "N/A" if not check or check["corr"] is None else format_ro(check["corr"], 2),
                         "Stare": "—" if ex_sym is None else ("fără date Yahoo" if ex_price is None else
                                  ("de verificat" if not check or not check["ok"] else "consistent"))})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        st.caption("Simbolurile UCITS au fost alese fără acces la Yahoo din mediul de dezvoltare. „consistent”: Yahoo are "
                   "date și corelația săptămânală cu ETF-ul semnal ≥ 0,80. Verifică ISIN-ul la broker înainte de ordin.")


def render_ideas_stocks(ctx):
    st.caption("Pas 1 (o descărcare): preț > SMA 200 și momentum 12-1 > 0, primele "
               f"{ideas_calc.PREFILTER_US} din SUA și {ideas_calc.PREFILTER_EU} din Europa după momentum (universurile din "
               "tools/univers_urmarit_*.txt). Pas 2: filtrul de calitate și cei trei catalizatori; idee = cel puțin 2 "
               "catalizatori și nicio excludere de calitate. ~5 cereri Yahoo pe acțiune, cache 12 ore.")
    c1, c2 = st.columns(2)
    if c1.button("🔎 Scanează acțiunile", key="ideas_scan_stocks_btn", type="primary"):
        st.session_state["ideas_scan_stocks"] = True
    if c2.button("🔄 Recalculează (golește cache-ul acțiunilor)", key="ideas_scan_stocks_reset"):
        get_ideas_stock_detail.clear()
        st.session_state["ideas_scan_stocks"] = True
    if not st.session_state.get("ideas_scan_stocks"):
        return
    us, eu = universes.symbols("us"), universes.symbols("eu")
    try:
        with st.spinner("Descarc prețurile universului (SUA + Europa)..."):
            px = get_ideas_prices(tuple(us + eu + ["SPY", "EXSA.DE"]), "2y")
    except Exception as exc:  # yfinance
        st.warning(_ideas_fetch_error("prețurile acțiunilor", exc))
        return
    closes = px["Close"]
    pre = pd.concat([ideas_calc.stock_prefilter(closes[[s for s in us if s in closes.columns]], ideas_calc.PREFILTER_US).assign(piata="SUA"),
                     ideas_calc.stock_prefilter(closes[[s for s in eu if s in closes.columns]], ideas_calc.PREFILTER_EU).assign(piata="Europa")],
                    ignore_index=True)
    if pre.empty:
        st.info("Nicio acțiune din universuri nu trece preselecția (preț > SMA 200 și momentum 12-1 > 0).")
        return
    today = pd.Timestamp(now_ro().date())
    rows, found = [], []
    prog = st.progress(0.0, text="Analizez candidații...")
    for i, r in pre.iterrows():
        sym = r["symbol"]
        prog.progress((i + 1) / len(pre), text=f"Analizez {sym} ({i + 1}/{len(pre)})")
        try:
            d = get_ideas_stock_detail(sym)
        except Exception as exc:  # yfinance: rețea, HTTP, format
            logger.warning("idei, detalii %s: %s", sym, exc)
            rows.append({"Simbol": sym, "Piață": r["piata"], "Nume": "N/A", "Momentum 12-1": _pct_txt(r["mom"]),
                         "Față de maxim": _pct_txt(r["from_high"]), "Catalizatori": "N/A", "Calitate": "N/A",
                         "Idee": "date indisponibile"})
            continue
        bench_sym = "SPY" if r["piata"] == "SUA" else "EXSA.DE"
        reaction = ideas_calc.earnings_reaction(_ideas_close(px, sym), _ideas_close(px, bench_sym), d["ann_date"])
        checks = [ideas_calc.pead_catalyst(d["surprise"], reaction, d["ann_date"], today),
                  ideas_calc.revision_catalyst(d["revision"], d["rev_period"]),
                  ideas_calc.high_catalyst(r["from_high"])]
        cat = ideas_calc.catalyst_summary(checks)
        qual = ideas_calc.quality_filter(d["altman_zone"], d["pio_passed"], d["pio_evaluable"], d["nd_ebitda"], d["financial"])
        is_idea = cat["is_idea"] and not qual["excluded"]
        rows.append({"Simbol": sym, "Piață": r["piata"], "Nume": d["name"], "Momentum 12-1": _pct_txt(r["mom"]),
                     "Față de maxim": _pct_txt(r["from_high"]),
                     "Catalizatori": f"{cat['passed']} din {cat['evaluable']} evaluabili",
                     "Calitate": ("exclus: " + ", ".join(c["name"] for c in qual["checks"] if c["passed"] is False))
                                 if qual["excluded"] else "trece",
                     "Idee": "✅" if is_idea else ""})
        if not is_idea:
            continue
        close = _ideas_close(px, sym)
        score = technical_subscore(close)
        m_lab, _, m_note = moment_label(score, False)
        rsi_last = rsi_wilder(close).iloc[-1] if len(close) > 30 else None
        fmt = lambda c: f"{'✅' if c['passed'] else '❌' if c['passed'] is False else '➖'} {c['name']}: {c['detail']}"
        steps = [("Fundamental (filtru de calitate)", "; ".join(fmt(c) for c in qual["checks"])),
                 ("Catalizatori", "; ".join(fmt(c) for c in checks)),
                 ("Tehnic", f"momentum 12-1 {_pct_txt(r['mom'])}; peste SMA 200; moment {score_ro(score)} ({m_lab}); "
                            f"RSI 14 {rsi_description(rsi_last)} (descriptiv)"),
                 ("Regim", f"sector {d['sector'] or 'N/A'}; cadran {ctx['quad']['key'] or 'N/A'}")]
        thesis = (f"{d['name']} trece {cat['passed']} din 3 catalizatori ("
                  + ", ".join(c["name"].lower() for c in checks if c["passed"]) + ") și nu e exclusă de filtrul de calitate.")
        news = []
        for n in get_company_news_rss(sym, names=(d["name"],), limit=3):
            news.append((html.escape(plain_text(n.get("title"))).replace("[", "(").replace("]", ")"), safe_link(n.get("link"))))
        found.append({"strategy": "stock", "symbol": sym, "name": d["name"], "thesis": thesis, "steps": steps,
                      "entry": _ideas_last(px, sym), "stop": ideas_atr_stop(px, sym), "exec_symbol": sym,
                      "exec_desc": "acțiunea direct (XTB / IBKR)", "exec_price": _ideas_last(px, sym),
                      "exec_currency": d["currency"], "exec_check": None, "alignment": None,
                      "held": sym in ctx["holdings"], "benchmark": bench_sym,
                      "earnings_days": ideas_calc.earnings_in_days(d["next_date"], today), "news": news,
                      "signals": " | ".join(c["detail"] for c in checks if c["passed"])})
    prog.empty()
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    if not found:
        st.info("Niciun candidat nu are cel puțin 2 catalizatori fără excludere de calitate.")
    for idea in found:
        render_idea(idea, ctx)


def render_ideas_bvb(ctx):
    st.caption("Singura parte cu bază testată în aplicație: treapta fundamentală (fișierele „Analiză companii”) × "
               "momentul (trend + momentum 12-1, prag ≥ 80). " + bvb_calc.TEST_NOTE)
    if st.button("🇷🇴 Calculează ideile BVB", key="ideas_bvb_btn", type="primary"):
        st.session_state["ideas_bvb"] = True
    if not st.session_state.get("ideas_bvb"):
        return
    files, err = list_company_files()
    if err:
        st.info(err)
        return
    with st.spinner("Citesc fișierele din „Analiză companii” (prima dată durează până la ~30 de secunde)..."):
        data = build_bvb_dataset(tuple((f["id"], f["name"], f["modified"]) for f in files))
        prices = get_bvb_file_prices(tuple(sorted(data["companii"])))
        cs = cross_section({t: current_indicators(df, prices.get(t)) for t, df in data["companii"].items()})
    if cs.empty or "treapta" not in cs.columns:
        st.info("Prea puține companii cu date pentru scorul transversal.")
        return
    syms = tuple(sorted(f"{t}.RO" for t in cs.index))
    try:
        px = get_ideas_prices(syms, "2y")
    except Exception as exc:  # yfinance
        st.warning(_ideas_fetch_error("prețurile BVB", exc))
        px = None
    rows, found = [], []
    for tk in cs.index:
        sym = f"{tk}.RO"
        close = _ideas_close(px, sym) if px is not None else pd.Series(dtype=float)
        moment = technical_subscore(close) if len(close) else None
        tier = cs.loc[tk, "treapta"]
        label, _color, hist_txt = bvb_calc.final_verdict(tier, moment)
        band = bvb_calc.moment_band(moment)
        rows.append({"Simbol": tk, "Scor fundamental": score_ro(cs.loc[tk, "scor"], 0) if pd.notna(cs.loc[tk, "scor"]) else "N/A",
                     "Treaptă": {2.0: "FAVORABIL", 1.0: "NEUTRU", 0.0: "NEFAVORABIL"}.get(tier, "N/A"),
                     "Moment": score_ro(moment), "Combinația": label, "Istoric": hist_txt})
        if pd.notna(tier) and int(tier) == 2 and band == "high":
            m, p, n = bvb_calc.TWO_STEP[(2, "high")][2]
            thesis = (f"Treapta fundamentală de sus (scor {score_ro(cs.loc[tk, 'scor'], 0)}/100, percentile față de "
                      f"{int(cs.loc[tk, 'n_comparate'])} companii) și moment {score_ro(moment)} ≥ 80. Istoric, combinația: "
                      f"{_pct_txt(m)} față de media BVB la 6 luni, {_pct_txt(p, 0, sign=False)} cazuri pozitive (N = {n}).")
            steps = [("Fundamental", f"scor {score_ro(cs.loc[tk, 'scor'], 0)}/100, treapta FAVORABIL (prag testat)"),
                     ("Tehnic", f"moment {score_ro(moment)} (FAVORABIL, prag testat ≥ 80)"),
                     ("Regim", "BVB: verdictul macro e pe SUA; vezi regimul BET în tabul Regim")]
            found.append({"strategy": "bvb", "symbol": sym, "name": tk, "thesis": thesis, "steps": steps,
                          "entry": _ideas_last(px, sym) if px is not None else None,
                          "stop": ideas_atr_stop(px, sym) if px is not None else None, "exec_symbol": sym,
                          "exec_desc": "acțiunea direct (Tradeville), RON",
                          "exec_price": _ideas_last(px, sym) if px is not None else None, "exec_currency": "RON",
                          "exec_check": None, "alignment": None, "held": sym in ctx["holdings"],
                          "benchmark": "TVBETETF.RO", "signals": f"scor {score_ro(cs.loc[tk, 'scor'], 0)}; moment {score_ro(moment)}"})
    n_cmp = int(cs["n_comparate"].max()) if "n_comparate" in cs.columns and cs["n_comparate"].notna().any() else 0
    if cs["treapta"].isna().all():
        st.caption(f"Treptele cer cel puțin 6 companii cu scor; acum au scor {n_cmp}. Fără treaptă, combinația e "
                   "„DATE INSUFICIENTE”.")
    order = {"CANDIDAT, MOMENT FAVORABIL": 0}
    st.dataframe(pd.DataFrame(rows).sort_values("Combinația", key=lambda s: s.map(lambda x: order.get(x, 1))),
                 hide_index=True, width='stretch')
    if not found:
        st.info("Niciun emitent nu are acum combinația „CANDIDAT, MOMENT FAVORABIL”.")
    for idea in found:
        render_idea(idea, ctx)


def render_ideas_journal(today):
    values = load_ideas_journal_values()
    if values is None:
        st.info(f"Foaia {ideas_calc.JOURNAL_SHEET} nu există încă: se creează la prima idee salvată (butonul din fișă).")
        return
    df, problems = ideas_calc.parse_journal(values, portfolio_sheet.cell_number)
    for p in problems:
        st.caption(p)
    if df.empty:
        st.info("Jurnalul e gol.")
        return
    syms = tuple(sorted(set(df["Symbol"]) | {b for b in df["Benchmark"] if b}))
    try:
        px = get_ideas_prices(syms, "5y")
    except Exception as exc:  # yfinance
        st.warning(_ideas_fetch_error("prețurile din jurnal", exc))
        return
    res = []
    for _, r in df.iterrows():
        e = ideas_calc.evaluate_entry(_ideas_close(px, r["Symbol"]), r["Date"], r["Entry"], r["Stop"], r["Horizon"],
                                      pd.Timestamp(today), _ideas_close(px, r["Benchmark"]) if r["Benchmark"] else None)
        res.append({**r.to_dict(), **e})
    res = pd.DataFrame(res)
    st.dataframe(pd.DataFrame({
        "Data": res["Date"].dt.strftime("%d.%m.%Y"), "Simbol": res["Symbol"], "Strategie": res["Strategie"],
        "Intrare": [price_ro(x) for x in res["Entry"]], "Stare": res["status"].fillna("N/A"),
        "Ieșire": [x.strftime("%d.%m.%Y") if pd.notna(x) else "N/A" for x in res["exit_date"]],
        "Randament": [_pct_txt(x) for x in res["ret"]], "Benchmark": res["Benchmark"],
        "Exces": [_pct_txt(x) for x in res["excess"]]}), hide_index=True, width='stretch')
    summ = ideas_calc.journal_summary(res)
    st.markdown("**Pe strategie (doar ideile închise)**")
    st.dataframe(pd.DataFrame({
        "Strategie": summ["Strategie"], "Închise": summ["Închise"], "Deschise": summ["Deschise"],
        "Randament mediu": [_pct_txt(x) for x in summ["Randament mediu"]],
        "Exces mediu": [_pct_txt(x) for x in summ["Exces mediu"]],
        "Pozitive": [_pct_txt(x, 0, sign=False) for x in summ["Pozitive"]],
        "Verdict": ["eșantion mic (< 20), fără verdict" if s else "—" for s in summ["small"]]}),
        hide_index=True, width='stretch')
    st.caption("Ieșire: prima închidere la sau sub stopul de la intrare, altfel la orizont; până atunci, poziția e "
               "„deschisă” și evaluată la ultima închidere. Prețuri ajustate cu dividendele (intrarea și stopul se "
               "rescalează). Fără costuri și fără efectul valutar.")


def render_ideas_page():
    st.title("💡 Idei și strategii")
    st.caption("Idei generate de reguli cu bază în literatura de specialitate, pe datele de azi. Pragurile sunt "
               "netestate în aplicație (excepție: BVB). Fiecare idee își arată datele, regula, stopul și limita; "
               "nu sunt îndemnuri de cumpărare.")
    today = now_ro().date()
    c1, c2 = st.columns([3, 1])
    capital = c1.number_input("Capitalul de referință pentru dimensionare (lei)", min_value=0.0, value=0.0,
                              step=1000.0, key="ideas_capital", help=IDEAS_HELP["capital"])
    c2.metric("Risc pe idee", "1%", help=IDEAS_HELP["risk"])
    with st.spinner("Calculez regimul și semnalele..."):
        quad, quad_msg = ideas_quadrant()
        try:
            px, px_err = get_ideas_prices(IDEAS_ETF_SYMBOLS, "3y"), None
        except Exception as exc:  # yfinance
            px, px_err = None, _ideas_fetch_error("prețurile ETF", exc)
        try:
            fx = dict(get_fx_live())
        except Exception as exc:  # yfinance
            logger.warning("idei, curs live: %s", exc)
            fx = {}
        for code in ("USD", "EUR", "GBP"):   # rezervă: ultima închidere zilnică Yahoo (aceeași descărcare ca ETF-urile)
            if not fx.get(code) and px is not None:
                fx[code] = _ideas_last(px, f"{code}RON=X")
        events, ev_msg = ideas_events(today)
        ctx = {"capital": capital or None, "fx": fx, "quad": quad, "holdings": ideas_holdings(),
               "cash": {"USD": ideas_cash("USD"), "EUR": ideas_cash("EUR")},
               "events_txt": [f"{r['label']} {r['date']:%d.%m}" for r in events]}
    t_reg, t_etf, t_stk, t_bvb, t_jur = st.tabs([":material/explore: Regim", ":material/stacked_line_chart: ETF-uri",
                                                 ":material/business: Acțiuni", "🇷🇴 BVB", ":material/book: Jurnal"])
    with t_reg:
        render_ideas_regime(quad, quad_msg, events, ev_msg, ctx)
    with t_etf:
        render_ideas_etf(px, px_err, ctx)
    with t_stk:
        render_ideas_stocks(ctx)
    with t_bvb:
        render_ideas_bvb(ctx)
    with t_jur:
        render_ideas_journal(today)
    render_event_radar()


# --- Radar evenimente (analytics/events.py, data/entities.py). Propunerea: claude/propunere_radar_evenimente_2026-11-14.md.
# Decizii Daniel (14.11.2026): doar reguli (fără AI), jurnal `Evenimente_jurnal` da, SEC 8-K mai târziu.
RADAR_BENCH = {"us": "SPY", "eu": "EXSA.DE", "bvb": "TVBETETF.RO"}
RADAR_MARKETS = (("us", "🇺🇸 SUA"), ("eu", "🇪🇺 Europa"), ("bvb", "🇷🇴 România"))
RADAR_YAHOO_BATCH = 10
RADAR_HELP = {
    "reactie": "Randamentul anormal = randamentul acțiunii minus al benchmarkului pieței (SPY / EXSA.DE / TVBETETF.RO), "
               "de la ultima închidere dinaintea știrii până la ultima închidere. z = anormal ÷ (volatilitatea zilnică a "
               "randamentului anormal pe 60 de ședințe × √ședințe). „Deja în preț”: z ≥ 2 în direcția așteptată; "
               "„contrazis”: z ≥ 2 invers; „încă nereflectat”: sub 2. Volum relativ: volumul ultimei ședințe ÷ media "
               "celor 20 dinainte. Praguri netestate.",
}


def _parse_yahoo_batch(symbols):
    """Titlurile din fluxul Yahoo RSS pentru un grup de simboluri (fără cache; apelat din get_radar_headlines)."""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={','.join(symbols)}&region=US&lang=en-US"
    try:
        feed = feedparser.parse(url)
    except Exception as exc:  # feedparser: rețea, format
        logger.warning("radar, Yahoo RSS %s: %s", symbols[0], exc)
        return []
    out = []
    for entry in getattr(feed, "entries", [])[:40]:
        dt = parse_date(entry)
        out.append({"title": getattr(entry, "title", ""), "link": getattr(entry, "link", ""),
                    "source": "Yahoo Finance", "date_obj": dt.replace(tzinfo=None)})
    return out


@st.cache_data(ttl=900, show_spinner=False)
def get_radar_headlines(batches):
    """Titlurile Yahoo RSS pe grupuri de câte 10 simboluri (tuplu de tupluri), descărcate în paralel; cache 15 min."""
    items = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for part in pool.map(_parse_yahoo_batch, batches):
            items.extend(part)
    return items


@st.cache_data(ttl=300, show_spinner=False)
def load_event_journal_values():
    """Conținutul brut al foii `Evenimente_jurnal`; None dacă foaia nu există încă sau nu poate fi citită."""
    ws = connect_to_gsheets(events_calc.JOURNAL_SHEET)
    if ws is None:
        return None
    try:
        return ws.get_all_values()
    except gspread.exceptions.GSpreadException as e:
        logger.warning(f"foaia {events_calc.JOURNAL_SHEET} nu a putut fi citită: {e}")
        return None


def append_event_journal(rows):
    """Adaugă în `Evenimente_jurnal` rândurile cu chei noi (foaia se creează la prima scriere, cu antetul
    JOURNAL_COLUMNS; aprobat de Daniel, 14.11.2026). Întoarce (câte rânduri noi, mesaj de eroare sau None)."""
    if not rows:
        return 0, None
    values = load_event_journal_values()
    df, _ = events_calc.parse_journal(values)
    known = set(df["Cheie"]) if not df.empty else set()
    new = [r for r in rows if r[-1] not in known]
    if not new:
        return 0, None
    try:
        ws = connect_to_gsheets(events_calc.JOURNAL_SHEET)
        if ws is None:
            book = _get_spreadsheet()
            if book is None:
                return 0, "portofoliu_db nu e accesibil"
            ws = book.add_worksheet(title=events_calc.JOURNAL_SHEET, rows=2000, cols=len(events_calc.JOURNAL_COLUMNS))
            ws.append_row(events_calc.JOURNAL_COLUMNS, value_input_option="RAW")
        ws.append_rows(new, value_input_option="RAW")
    except (gspread.exceptions.GSpreadException, requests.RequestException, AttributeError) as exc:
        logger.warning("Evenimente_jurnal, scriere eșuată: %s", exc)
        return 0, "scrierea în Evenimente_jurnal a eșuat"
    load_event_journal_values.clear()   # doar cache-ul jurnalului
    return len(new), None


def _radar_name(sym):
    return entities_data.NAMES.get(sym, (sym,))[0]


def _radar_rows(pairs, reactions, ev_key, holdings, groups=None):
    """Tabelul companiilor dintr-un eveniment: simbol, direcție, reacție, „deții”."""
    rows = []
    for sym, d in pairs:
        r = reactions.get((ev_key, sym), {})
        rows.append({"Simbol": sym, "Companie": _radar_name(sym),
                     "Așteptat": {1: "↑ favorabil", -1: "↓ nefavorabil"}.get(d, "· menționat / ambiguu"),
                     "Anormal": _pct_txt(r.get("abnormal")), "z": format_ro(r.get("z"), 1),
                     "Volum relativ": format_ro(r.get("rvol"), 1, "×"), "Reacția pieței": r.get("label", "N/A"),
                     **({"Grup": ", ".join(groups.get(sym, []))} if groups is not None else {}),
                     "Deții": "da" if sym in holdings else ""})
    return pd.DataFrame(rows)


def render_event_radar():
    st.divider()
    st.header("📰 Radar evenimente")
    st.caption("Titlurile din ultimele 48 de ore (fluxurile RSS din Știri + Yahoo pe fiecare acțiune din universuri) "
               "legate de companii prin nume și clasificate prin reguli (fără AI): cine e afectat, ce concurenți pot "
               "câștiga cotă, cum a reacționat deja piața. Titlurile ajung cu întârziere: nu e semnal de tranzacționare "
               "rapidă; valoarea e alerta pe ce deții și efectele de ordinul 2. " + events_calc.UNTESTED.capitalize() + ".")
    c1, c2 = st.columns(2)
    if c1.button("📰 Scanează știrile", key="radar_btn", type="primary"):
        st.session_state["radar_on"] = True
    if c2.button("🔄 Reîmprospătează știrile", key="radar_refresh"):
        get_radar_headlines.clear()
        fetch_news_data.clear()
        st.session_state["radar_on"] = True
    if not st.session_state.get("radar_on"):
        return
    now = now_ro().replace(tzinfo=None)
    us, eu, bvb = universes.symbols("us"), universes.symbols("eu"), universes.symbols("bvb")
    holdings = ideas_holdings()
    tracked = list(dict.fromkeys(us + eu + bvb + sorted(holdings)))
    yahoo_syms = [s for s in tracked if not s.endswith(".RO")]
    batches = tuple(tuple(yahoo_syms[i:i + RADAR_YAHOO_BATCH]) for i in range(0, len(yahoo_syms), RADAR_YAHOO_BATCH))
    with st.spinner("Citesc titlurile și le leg de companii..."):
        items = list(fetch_news_data()) + list(get_radar_headlines(batches))
        matcher = events_calc.EntityMatcher(tracked, entities_data.NAMES, entities_data.COMMON_TICKER_WORDS,
                                            entities_data.PRODUCT_ALIASES)
        evs = events_calc.detect_events(items, matcher, entities_data.COMPETITOR_GROUPS, now, universes.market_of)
        mentioned = {s for it in events_calc.clean_items(items, now) for s in matcher.find(it.get("title"))}
    involved = sorted({s for e in evs for s, _ in e["affected"]} | {s for e in evs for s in e["beneficiaries"]})
    try:
        with st.spinner("Descarc prețurile universului (reacția pieței și mișcările fără știre)..."):
            px = get_ideas_prices(tuple(sorted(set(tracked) | set(involved) | set(RADAR_BENCH.values()))), "6mo")
    except Exception as exc:  # yfinance
        px = None
        st.info(_ideas_fetch_error("prețurile pentru radar", exc))
    reactions = {}
    if px is not None:
        for e in evs:
            pairs = list(e["affected"]) + [(s, 1) for s in e["beneficiaries"]]
            for sym, d in pairs:
                mk = universes.market_of(sym)
                reactions[(e["key"], sym)] = events_calc.reaction(
                    _ideas_close(px, sym), _ideas_close(px, RADAR_BENCH[mk]),
                    px["Volume"][sym] if "Volume" in px and sym in px["Volume"].columns else None, e["time"], mk, d)
    moves = []
    if px is not None:
        bench_of = {s: _ideas_close(px, RADAR_BENCH[universes.market_of(s)]) for s in px["Close"].columns}
        closes = px["Close"][[s for s in px["Close"].columns if s in set(tracked) and s not in RADAR_BENCH.values()]]
        moves = events_calc.big_moves(closes, bench_of, exclude=mentioned)
    n_new, j_err = append_event_journal(events_calc.journal_rows(evs, reactions, RADAR_BENCH, universes.market_of, moves))

    n_items = len(events_calc.clean_items(items, now))
    st.caption(f"{n_items} {'titlu' if n_items == 1 else 'titluri'} în ultimele 48 de ore, {len(evs)} "
               f"{'eveniment' if len(evs) == 1 else 'evenimente'}. Jurnal: {n_new} "
               f"{'rând nou' if n_new == 1 else 'rânduri noi'} în {events_calc.JOURNAL_SHEET}"
               + (f" ({j_err})." if j_err else "."))
    hit = [(e, events_calc.touches(e, holdings)) for e in evs]
    hit = [(e, t) for e, t in hit if t]
    if hit:
        subj = lambda e: e["label"] if e["kind"] == "sector" else _radar_name(e["affected"][0][0])
        st.info("⚠️ Evenimente care ating pozițiile tale: " + "; ".join(
            f"{', '.join(f'{s} ({r})' for s, r in t)} — {subj(e)}"
            + (f": {e['label'].lower()}" if e["kind"] == "companie" else "") for e, t in hit[:8]))
    icon = {3: "🔴", 2: "🟠", 1: "🟡"}
    tabs = st.tabs([lbl for _, lbl in RADAR_MARKETS])
    for (mk, _lbl), tab in zip(RADAR_MARKETS, tabs):
        with tab:
            m_evs = [e for e in evs if mk in e["markets"]]
            if not m_evs:
                st.caption("Niciun eveniment detectat pe această piață în ultimele 48 de ore.")
            for e in m_evs:
                aff = [(s, d) for s, d in e["affected"] if universes.market_of(s) == mk] if e["kind"] == "sector" else e["affected"]
                subj = e["label"] if e["kind"] == "sector" else f"{_radar_name(aff[0][0])} ({aff[0][0]})"
                kind = {1: "↑", -1: "↓", 0: "?"}[e["direction"]] if e["kind"] == "sector" else ""
                sev = icon[min(3, e["severity_total"])]
                touched = events_calc.touches(e, holdings)
                with st.expander(f"{sev} {subj}{' ' + kind if kind else ''} — "
                                 f"{e['label'] if e['kind'] == 'companie' else 'eveniment sectorial'} · "
                                 f"{e['time']:%d.%m %H:%M}" + (" · deții" if touched else ""), expanded=bool(touched)):
                    for h in e["headlines"][:5]:
                        title = html.escape(plain_text(h["title"])).replace("[", "(").replace("]", ")")
                        link = safe_link(h["link"])
                        st.markdown(f"- {'[' + title + '](' + link + ')' if link else title} — "
                                    f"{html.escape(str(h['source'] or ''))}, {h['time']:%d.%m %H:%M}")
                    st.markdown("**Afectate**" if e["kind"] == "companie" else
                                f"**Companii expuse** (direcția temei: {kind}; grupuri: {', '.join(e.get('groups', []))})",
                                help=RADAR_HELP["reactie"])
                    st.dataframe(_radar_rows(aff, reactions, e["key"], holdings), hide_index=True, width='stretch')
                    if e["beneficiaries"]:
                        st.markdown("**Pot câștiga cotă (concurenți din același grup)**", help=RADAR_HELP["reactie"])
                        st.dataframe(_radar_rows([(s, 1) for s in e["beneficiaries"]], reactions, e["key"], holdings,
                                                 groups=e["beneficiaries"]), hide_index=True, width='stretch')
                    extra = f" Alte tipuri în titlu: {', '.join(events_calc.EVENT_TYPES[t]['label'].lower() for t in e['other_types'])}." if e["other_types"] else ""
                    st.caption("Direcția vine din tipul evenimentului (reguli pe cuvinte, doar titlul); beneficiarii, "
                               "doar la evenimentele care pot muta cotă de piață. Regulile nu înțeleg negații sau ironie: "
                               "citește titlul." + extra)
            m_moves = [m for m in moves if universes.market_of(m["symbol"]) == mk]
            st.markdown("**Mișcări mari fără știre potrivită** (|z| ≥ 3 în ultima ședință)",
                        help="Randamentul anormal al ultimei ședințe ÷ volatilitatea lui pe 60 de ședințe, pentru acțiunile "
                             "din univers fără niciun titlu care să le numească în 48 de ore. Chan (2003): după mișcări "
                             "mari fără știre publică prețul tinde să revină parțial; după cele cu știre tinde să continue. "
                             "Netestat în aplicație.")
            if m_moves:
                st.dataframe(pd.DataFrame([{"Simbol": m["symbol"], "Companie": _radar_name(m["symbol"]),
                                            "Data": f"{m['date']:%d.%m}", "Randament": _pct_txt(m["ret"]),
                                            "Anormal": _pct_txt(m["abnormal"]), "z": format_ro(m["z"], 1),
                                            "Deții": "da" if m["symbol"] in holdings else ""} for m in m_moves[:15]]),
                             hide_index=True, width='stretch')
            else:
                st.caption("Niciuna.")

    with st.expander("📓 Jurnalul radarului: s-a confirmat direcția?"):
        values = load_event_journal_values()
        jdf, probs = events_calc.parse_journal(values)
        for p in probs:
            st.caption(p)
        if jdf.empty:
            st.caption("Jurnalul e gol: se completează automat la fiecare scanare.")
            return
        syms = tuple(sorted(set(jdf["Symbol"]) | {b for b in jdf["Benchmark"] if b}))
        try:
            jpx = get_ideas_prices(syms, "2y")
        except Exception as exc:  # yfinance
            st.caption(_ideas_fetch_error("prețurile jurnalului", exc))
            return
        res = jdf.copy()
        evals = [events_calc.evaluate_event(_ideas_close(jpx, r["Symbol"]), _ideas_close(jpx, r["Benchmark"]), r["Data_ref"])
                 for _, r in jdf.iterrows()]
        for k in events_calc.EVAL_SESSIONS:
            res[f"ab{k}"] = [x[k] for x in evals]
        summ = events_calc.journal_summary(res)
        st.dataframe(pd.DataFrame({
            "Rol": summ["Rol"],
            **{f"N la {k} ședințe": summ[f"N {k}"] for k in events_calc.EVAL_SESSIONS},
            **{f"Direcție corectă la {k}": [_pct_txt(x, 0, sign=False) for x in summ[f"Nimerite {k}"]] for k in events_calc.EVAL_SESSIONS},
            **{f"Anormal mediu în direcție, {k}": [_pct_txt(x) for x in summ[f"Medie {k}"]] for k in events_calc.EVAL_SESSIONS},
            "Verdict": ["eșantion mic (< 20 la 20 de ședințe), fără verdict" if s else "—" for s in summ["small"]]}),
            hide_index=True, width='stretch')
        st.caption(f"{len(jdf)} rânduri. Afectat: direcția din tipul evenimentului; beneficiar: ↑; fără știre: invers "
                   "mișcării (revenire, Chan 2003). Anormal = față de benchmarkul pieței, pe prețuri ajustate.")


# --- FONDURI DE PENSII PILON II (foaia Pilon2_detineri, scrisă de jobul de pe server) ---

PILON2_NOTE = ("Sursa: rapoartele ASF „Anexa 12” (structura detaliată a portofoliului) ale celor 7 fonduri Pilon II, "
               "publicate lunar cu **~3,5 luni întârziere**: descriu ce au făcut fondurile atunci, nu acum. "
               "Numărul de acțiuni nu apare în rapoarte: e **derivat** (valoare / prețul de închidere de la data "
               "raportului, din seria Yahoo de azi, ajustată pentru acțiunile gratuite, deci în unitățile de azi) — proxy. "
               "Fluxul în lei = Δ acțiuni × prețul de la data raportului (fără efectul de preț). "
               "Informativ și **netestat**: nu e semnal de moment.")

PILON2_HELP = {
    "detinere": ("**Ce e:** câte acțiuni dețin împreună fondurile Pilon II care au raportat la data afișată.\n\n"
                 "**Calcul:** suma pe fonduri a cantității derivate = valoarea din raport / prețul de închidere "
                 "neajustat din ultima ședință ≤ data raportului.\n\n**Atenție:** proxy; fondurile raportează la date "
                 "diferite, deci totalul cuprinde doar fondurile cu raport la acea dată."),
    "capital": ("**Ce e:** ce parte din companie e la fondurile Pilon II.\n\n**Calcul:** deținerea totală / numărul de "
                "acțiuni al companiei (Yahoo: `sharesOutstanding`; dacă lipsește, `impliedSharesOutstanding`, apoi "
                "capitalizare / preț, marcat proxy).\n\n**Atenție:** la BVB numărul de acțiuni de la Yahoo "
                "poate lipsi sau fi vechi → „N/A”. O pondere mare = free-float real mic și risc de vânzare masivă dacă "
                "se schimbă legislația Pilonului II."),
    "delta": ("**Ce e:** câte acțiuni au cumpărat (+) sau vândut (−) net fondurile față de raportul anterior al fiecăruia.\n\n"
              "**Calcul:** Σ (cantitatea acum − cantitatea la raportul anterior al fondului); în lei: × prețul de la data "
              "raportului.\n\n**Atenție:** descrie o lună de acum ~3,5 luni."),
    "fonduri": ("**Ce e:** câte dintre fondurile care au raportat la data afișată dețin acțiunea.\n\n"
                "**Calcul:** fondurile cu cantitate > 0 / fondurile cu raport la acea dată."),
    "cost": ("**Ce e:** prețul mediu la care au cumpărat net fondurile în ultimele 12 luni (proxy).\n\n"
             "**Calcul:** Σ(acțiuni cumpărate × prețul de la data raportului) / Σ acțiuni cumpărate, pe fonduri și luni.\n\n"
             "**Atenție:** prețul de sfârșit de lună, nu prețul real al tranzacțiilor; sub 20 de observații e doar orientativ."),
    "lichidare": ("**Ce e:** câte ședințe ar dura vânzarea întregii dețineri Pilon II la volumul mediu zilnic (proxy de risc).\n\n"
                  "**Calcul:** deținerea totală / volumul mediu zilnic din ultimele 3 luni (Yahoo).\n\n"
                  "**Atenție:** volumele BVB din Yahoo au goluri; rezultatul e un ordin de mărime."),
}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_pilon2():
    """Foaia `Pilon2_detineri` → (DataFrame, probleme, eroare sau None). Doar citire."""
    empty = pilon2_data.parse_holdings([])[0]
    try:
        ws = connect_to_gsheets(pilon2_data.HOLDINGS_SHEET)
        if ws is None:
            return empty, [], ("Foaia Pilon2_detineri nu există încă: o creează jobul de pe server la prima rulare "
                               "(deploy/README.md, „Jobul Pilon II”).")
        df, problems = pilon2_data.parse_holdings(ws.get_all_values())
        return df, problems, None
    except Exception as e:  # rețea / Sheets: mesaj în interfață, aplicația continuă
        logger.warning(f"Pilon2_detineri nu a putut fi citită: {e}")
        return empty, [], f"Foaia Pilon2_detineri nu a putut fi citită: {e}"


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def pilon2_closes(symbols, start):
    """Închiderile zilnice ale simbolurilor BVB din foaie, de la `start` (auto_adjust=False; Yahoo ajustează totuși
    pentru acțiunile gratuite). Întoarce (DataFrame sau None, eroare sau None)."""
    try:
        raw = yf.download(list(symbols), start=start, auto_adjust=False, progress=False, threads=False)
        close = close_frame(raw, list(symbols))
        if close is not None and not close.empty and getattr(close.index, "tz", None) is not None:
            close.index = close.index.tz_localize(None)
        return close, None
    except Exception as e:  # rețea / Yahoo: cantitățile rămân cele din foaie, cu notă în interfață
        logger.warning(f"Pilon II: prețurile pentru recalcularea cantităților nu au venit: {e}")
        return None, str(e)


def pilon2_holdings():
    """Foaia Pilon II cu cantitățile BVB recalculate la prețurile de azi. Întoarce (df, probleme, eroare, notă)."""
    df, problems, err = load_pilon2()
    if err or df.empty:
        return df, problems, err, None
    bvb = df[df["Symbol"].fillna("").str.endswith(".RO")]
    if bvb.empty:
        return df, problems, err, None
    start = (bvb["Date"].min() - pd.Timedelta(days=p2_calc.MAX_PRICE_LAG_DAYS + 3)).strftime("%Y-%m-%d")
    close, cerr = pilon2_closes(tuple(sorted(bvb["Symbol"].unique())), start)
    out, kept = p2_calc.rebase_quantities(df, close)
    if cerr or close is None or close.empty:
        note = "Prețurile Yahoo nu au venit: cantitățile sunt cele scrise în foaie (unități de la data scrierii)."
    elif kept:
        note = ("Cantități recalculate la prețurile de azi, cu excepția: " + ", ".join(k[:-3] for k in kept)
                + " (fără date Yahoo; valorile din foaie).")
    else:
        note = None
    return out, problems, err, note


def _pilon2_metrics(cp, shares_out):
    """Rândul de carduri comun (Analiză companie și pagina Fonduri de pensii). Întoarce rândul de totaluri."""
    tot = p2_calc.totals(cp)
    # luna raportată de cele mai multe fonduri: ultima dată poate avea un singur fond (decalaje diferite)
    ref = p2_calc.reference_date(cp)
    last = tot[tot["Date"] == ref].iloc[-1] if ref is not None else tot.iloc[-1]
    pct = (last["Acțiuni"] / shares_out * 100) if (shares_out and pd.notna(last["Acțiuni"])) else None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Deținere Pilon II", f"{ro_spec(',.0f')(last['Acțiuni'])} acț.", help=PILON2_HELP["detinere"])
    c2.metric("% din capital", ro_spec(".2f", "%")(pct), help=PILON2_HELP["capital"])
    dq_txt = ro_spec("+,.0f")(last["dQ"])
    flux_txt = ro_spec("+,.0f", " lei")(last["Flux_RON"])
    c3.metric("Δ față de raportul anterior", f"{dq_txt} acț.", flux_txt if flux_txt != "N/A" else None,
              delta_color="normal" if flux_txt != "N/A" else "off", help=PILON2_HELP["delta"])
    c4.metric("Fonduri care dețin", f"{int(last['Fonduri care dețin'])} din {int(last['Fonduri raportate'])}",
              help=PILON2_HELP["fonduri"])
    st.caption(f"Raportul din {last['Date']:%d.%m.%Y} (publicat cu ~3,5 luni întârziere). Cantități derivate (proxy).")
    return tot


def render_pilon2_card(real_sym, info):
    """Card scurt în Analiză companie, doar pentru .RO. Detaliile sunt în pagina „Fonduri de pensii”."""
    sym = str(real_sym).upper()
    if not sym.endswith(".RO"):
        return
    st.subheader("🏦 Fonduri de pensii Pilon II")
    df, _problems, err, note = pilon2_holdings()
    if err:
        st.caption(err)
        return
    if note:
        st.caption(note)
    cp = p2_calc.company_panel(p2_calc.build_panel(df), sym)
    if cp.empty:
        st.caption(f"{sym[:-3]} nu apare în rapoartele fondurilor Pilon II din foaie "
                   "(sau ISIN-ul nu are încă simbol în ISIN_map).")
        return
    shares_out, src = p2_calc.shares_outstanding(info)
    _pilon2_metrics(cp, shares_out)
    if src and src != "sharesOutstanding":
        st.caption(f"Numărul de acțiuni pentru „% din capital”: Yahoo, {src}.")
    st.caption("Tabelul pe fonduri, graficele și fluxurile pe piață: meniul „Fonduri de pensii”.")


def _pilon2_company(panel, df):
    last_date = p2_calc.reference_date(panel)
    size = (panel[panel["Date"] == last_date].groupby("Symbol")["Valoare"].sum().sort_values(ascending=False))
    syms = list(size.index) + sorted(set(panel["Symbol"]) - set(size.index))
    names = (df.dropna(subset=["Symbol"]).drop_duplicates("Symbol", keep="last").set_index("Symbol")["Denumire"].to_dict())
    sym = st.selectbox("Companie (BVB), în ordinea valorii deținute", syms, key="p2_sym",
                       format_func=lambda s: f"{s[:-3]} — {names.get(s) or ''}".rstrip(" —"))
    cp = p2_calc.company_panel(panel, sym)
    hist, info, _earn, _real = get_stock_data(sym)
    shares_out, shares_src = p2_calc.shares_outstanding(info)
    tot = _pilon2_metrics(cp, shares_out)
    if shares_src and shares_src != "sharesOutstanding":
        st.caption(f"Numărul de acțiuni pentru „% din capital”: Yahoo, {shares_src}.")

    adv = None
    if hist is not None and not hist.empty and "Volume" in hist.columns:
        vol = slice_window(hist["Volume"], "3L").replace(0, np.nan).dropna()
        adv = float(vol.mean()) if len(vol) else None
    cost, n_cost = p2_calc.implied_cost(cp, since=tot["Date"].max() - pd.DateOffset(months=12))
    dtl = p2_calc.days_to_liquidate(tot["Acțiuni"].iloc[-1], adv)
    c1, c2, _c3 = st.columns(3)
    c1.metric("Preț mediu implicit al cumpărărilor (12 luni)",
              ro_spec(",.2f", " lei")(cost) + (f" · {n_cost} obs." if n_cost else ""), help=PILON2_HELP["cost"])
    c2.metric("Zile de lichidare", ro_spec(",.0f")(dtl), help=PILON2_HELP["lichidare"])
    if n_cost and n_cost < p2_calc.MIN_OBS:
        st.caption(f"Prețul mediu implicit are doar {n_cost} observații (sub {p2_calc.MIN_OBS}): orientativ, fără verdict.")

    table = p2_calc.fund_table(cp, shares_out)
    if not table.empty:
        st.dataframe(table.style.format({
            "Data raport": lambda d: f"{d:%d.%m.%Y}", "Acțiuni": ro_spec(",.0f"), "% din capital": ro_spec(".2f", "%"),
            "Valoare (lei)": ro_spec(",.0f"), "Pondere în fond (%)": ro_spec(".2f", "%"),
            "Δ față de raportul anterior": ro_spec("+,.0f"), "Δ 3 luni": ro_spec("+,.0f"), "Δ 12 luni": ro_spec("+,.0f"),
        }, na_rep="N/A"), hide_index=True, width='stretch', column_config={
            "Etichetă": st.column_config.Column(help="Față de raportul anterior al fondului: nou / intrare / neschimbat / "
                                                     "ieșire / ieșit complet."),
            "Data raport": st.column_config.Column(help="Data ultimului raport al fondului (fondurile publică la date diferite)."),
        })

    # Deținerea pe fonduri (bare stivuite, culoarea urmează fondul)
    fig = go.Figure()
    for fond in sorted(cp["Fond"].unique(), key=p2_calc.fund_sort_key):
        g = cp[cp["Fond"] == fond].sort_values("Date")
        fig.add_trace(go.Bar(x=g["Date"], y=g["Cantitate"], name=fond, marker_color=p2_calc.fund_color(fond),
                             marker_line_width=0, hovertemplate="%{x|%m.%Y}<br>" + fond + ": %{y:,.0f} acț.<extra></extra>"))
    fig.update_layout(barmode="stack", height=380, title="Acțiuni deținute, pe fonduri (cantități derivate)",
                      template="plotly_dark", paper_bgcolor='#0E1117', plot_bgcolor='#0E1117', bargap=0.25,
                      legend=dict(orientation="h", y=-0.15), margin=dict(t=50, b=10))
    plot_ro(fig, width='stretch')

    # Fluxul net lunar (bare) și prețul de la data raportului (grafic separat, fără a doua axă)
    flows = tot.dropna(subset=["Flux_RON"])
    price = cp.groupby("Date")["Pret_data"].max().dropna()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.62, 0.38],
                        subplot_titles=("Flux net al fondurilor (lei, Δ acțiuni × prețul de la data raportului)",
                                        "Prețul de închidere la data raportului (lei)"))
    fig.add_trace(go.Bar(x=flows["Date"], y=flows["Flux_RON"], name="Flux net", showlegend=False,
                         marker_color=[DF_UP if v >= 0 else DF_DOWN for v in flows["Flux_RON"]], marker_line_width=0,
                         hovertemplate="%{x|%m.%Y}: %{y:+,.0f} lei<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=price.index, y=price.values, mode="lines+markers", name="Preț", showlegend=False,
                             line=dict(width=2, color="#C9D1D9"), marker=dict(size=8),
                             hovertemplate="%{x|%d.%m.%Y}: %{y:,.2f} lei<extra></extra>"), row=2, col=1)
    fig.update_layout(height=480, template="plotly_dark", paper_bgcolor='#0E1117', plot_bgcolor='#0E1117',
                      margin=dict(t=50, b=10), bargap=0.3)
    plot_ro(fig, width='stretch')

    if shares_out:
        pct = tot.dropna(subset=["Acțiuni"])
        fig = go.Figure(go.Scatter(x=pct["Date"], y=pct["Acțiuni"] / shares_out * 100, mode="lines+markers",
                                   line=dict(width=2, color=p2_calc.FUND_COLORS["NN"]), marker=dict(size=8),
                                   hovertemplate="%{x|%m.%Y}: %{y:.2f}% din capital<extra></extra>"))
        fig.update_layout(height=300, title="Partea din capital la fondurile Pilon II (%)", template="plotly_dark",
                          paper_bgcolor='#0E1117', plot_bgcolor='#0E1117', margin=dict(t=50, b=10))
        plot_ro(fig, width='stretch')
    st.caption("Totalul unei luni cuprinde doar fondurile care au raport pentru acea lună; o lună în care lipsește "
               "un fond mare arată o scădere falsă. Coloana „Data raport” din tabel arată ultimul raport al fiecăruia.")


def _pilon2_market(panel, df):
    ref = p2_calc.reference_date(panel)
    n_funds = panel[panel["Date"] == ref]["Fond"].nunique()
    st.caption(f"Luna de referință: {ref:%m.%Y} ({n_funds} fonduri cu raport). Flux = Δ acțiuni × prețul de la data raportului.")
    no_prev = p2_calc.funds_without_previous(panel, ref)
    if no_prev:
        st.caption("Fără raport anterior (incluse în „Deținere”, nu și în flux): " + ", ".join(no_prev) + ".")
    m = p2_calc.market_flows(panel, ref)
    m = m.assign(Symbol=m["Symbol"].str[:-3]).rename(columns={
        "Symbol": "Simbol", "Flux_RON": "Flux net (lei)", "dQ": "Δ acțiuni", "Valoare": "Deținere (lei)"})
    fmt = {"Flux net (lei)": ro_spec("+,.0f"), "Δ acțiuni": ro_spec("+,.0f"), "Deținere (lei)": ro_spec(",.0f")}
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Cele mai mari cumpărări nete**")
        buys = m[m["Flux net (lei)"] > 0].head(10)
        if buys.empty:
            st.caption("Nicio cumpărare netă.")
        else:
            st.dataframe(buys.style.format(fmt, na_rep="N/A"), hide_index=True, width='stretch')
    with c2:
        st.markdown("**Cele mai mari vânzări nete**")
        sells = m[m["Flux net (lei)"] < 0].sort_values("Flux net (lei)").head(10)
        if sells.empty:
            st.caption("Nicio vânzare netă.")
        else:
            st.dataframe(sells.style.format(fmt, na_rep="N/A"), hide_index=True, width='stretch')

    hm = p2_calc.flow_heatmap(panel, months=12, top=25)
    if not hm.empty:
        z = hm.to_numpy(dtype=float)
        zmax = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 1.0
        fig = go.Figure(go.Heatmap(
            z=z, x=[f"{d:%m.%Y}" for d in hm.columns], y=[s[:-3] for s in hm.index], zmid=0, zmin=-zmax, zmax=zmax,
            colorscale=[[0, DF_DOWN], [0.5, "#6E7681"], [1, DF_UP]], xgap=2, ygap=2,
            colorbar=dict(title="lei"), hovertemplate="%{y} · %{x}: %{z:+,.0f} lei<extra></extra>"))
        fig.update_layout(height=max(360, 22 * len(hm)), title="Flux net lunar pe emitent (cele mai mari 25 de dețineri)",
                          template="plotly_dark", paper_bgcolor='#0E1117', plot_bgcolor='#0E1117',
                          margin=dict(t=50, b=10), yaxis=dict(autorange="reversed"))
        plot_ro(fig, width='stretch')
        st.caption("Celulă goală = fluxul nu se poate calcula (lipsește prețul sau raportul anterior al unui fond).")

    eq = p2_calc.fund_equity_totals(df)
    if not eq.empty:
        fig = go.Figure()
        for fond in sorted(eq["Fond"].unique(), key=p2_calc.fund_sort_key):
            g = eq[eq["Fond"] == fond].sort_values("Date")
            fig.add_trace(go.Scatter(x=g["Date"], y=g["Valoare"], name=fond, mode="lines+markers",
                                     line=dict(width=2, color=p2_calc.fund_color(fond)), marker=dict(size=8),
                                     hovertemplate="%{x|%m.%Y}<br>" + fond + ": %{y:,.0f} lei<extra></extra>"))
        fig.update_layout(height=380, title="Valoarea acțiunilor din portofoliul fiecărui fond (RO + străine, lei)",
                          template="plotly_dark", paper_bgcolor='#0E1117', plot_bgcolor='#0E1117',
                          legend=dict(orientation="h", y=-0.15), margin=dict(t=50, b=10))
        plot_ro(fig, width='stretch')


def render_pilon2_page():
    st.title("🏦 Fonduri de pensii Pilon II")
    st.caption(PILON2_NOTE)
    c_reload, _ = st.columns([1, 3])
    if c_reload.button("🔄 Recitește foaia", key="p2_reload"):
        load_pilon2.clear()
        st.rerun()
    df, problems, err, note = pilon2_holdings()
    if err:
        st.info(err)
        return
    if note:
        st.caption(note)
    for p in problems:
        st.warning(p)
    panel = p2_calc.build_panel(df)
    if panel.empty:
        st.info("Foaia nu are încă rânduri BVB cu simbol (ISIN_map). Rulează jobul de pe server.")
        return
    tab_co, tab_mk = st.tabs([":material/domain: Pe companie", ":material/public: Pe piață"])
    with tab_co:
        _pilon2_company(panel, df)
    with tab_mk:
        _pilon2_market(panel, df)


# ==================================================
# FONDURI SUA (13F) — propunerea din 25.11.2026 (claude/propunere_date_sec_2026-11-25.md)
# ==================================================
SEC_NOTE = ("Rapoartele 13F ale administratorilor americani (SEC EDGAR): pozițiile long în acțiuni SUA la sfârșitul "
            "trimestrului, depuse în cel mult 45 de zile. Nu arată short-urile, acoperirile, pozițiile din afara SUA "
            "sau pe cele confidențiale. Descriu ce dețineau fondurile atunci, nu acum: context, nu semnal de moment. "
            "Netestat.")

SEC_HELP = {
    "pozitii": ("**Ce e:** numărul de titluri distincte (CUSIP) din portofoliul pe acțiuni.\n\n"
                "**Calcul:** rândurile SH fără opțiuni, adunate pe CUSIP. Opțiunile și obligațiunile (PRN) sunt separat."),
    "valoare": ("**Ce e:** valoarea portofoliului pe acțiuni declarată în raport, la prețurile de la sfârșitul trimestrului.\n\n"
                "**Atenție:** fără opțiuni și fără pozițiile din afara SUA."),
    "top": "**Ce e:** cea mai mare poziție și ponderea ei în portofoliul pe acțiuni al fondului.",
    "perioada": ("**Perioada:** sfârșitul trimestrului raportat. **Depus:** data ultimei depuneri (inclusiv amendamente). "
                 "**Decalaj:** zilele până la raportul de bază (legal: 45; dacă termenul cade în weekend sau sărbătoare, "
                 "prima zi lucrătoare)."),
    "clasa": ("**Concentrat:** ≤ 100 de poziții sau primele 10 ≥ 50% din valoare (calculat din raport). "
              "**Larg:** fonduri cantitative / multi-strategie, cu mii de poziții: nu reflectă convingere și sunt excluse "
              "din consens."),
    "convingere": ("**Poziție de convingere:** pondere ≥ max(1/N, 1%), unde N = numărul de poziții pe acțiuni ale "
                   "fondului: o poziție peste media propriului portofoliu (8 poziții → 12,5%; 30 → 3,33%; 100 sau mai "
                   "multe → 1%). Clasele de acțiuni ale aceluiași emitent se adună.\n\n**Limită:** nu corectează "
                   "mărimea companiei (AAPL apare la mulți doar pentru că e mare). Netestat."),
    "miscari": ("**Mișcarea** se judecă pe numărul de acțiuni, ajustat cu split-urile din trimestru (Yahoo): Nouă, "
                "Închisă, Mărită (> +5%), Redusă (< −5%), Păstrată.\n\n"
                "**Valoare tranzacționată (proxy):** Δ acțiuni × media închiderilor din trimestru (nu prețul real). "
                "**Impact:** |valoarea tranzacționată| / valoarea portofoliului — cât a contat mișcarea pentru fond."),
    "consens": ("**Dețin:** administratori concentrați care au emitentul (clasele de acțiuni adunate: GOOG + GOOGL). "
                "**Convingere:** câți îl au ca poziție de convingere. **Cumpărări:** câți l-au cumpărat nou sau au mărit "
                "cu ≥ +50% în ultimul trimestru, cu pondere finală ≥ 1%.\n\n**Netestat:** context, nu semnal."),
}


def sec_secret(key):
    """Un secret pentru datele SEC (`sec_user_agent`, `openfigi_api_key`) sau None. Nu se afișează."""
    try:
        val = st.secrets.get(key)
    except (FileNotFoundError, KeyError) as exc:   # fără secrets.toml (rulare locală)
        logger.info("%s indisponibil: %s", key, type(exc).__name__)
        return None
    val = str(val).strip() if val else ""
    return val or None


@st.cache_resource(show_spinner=False)
def sec_client(user_agent):
    """Un singur client EDGAR pe proces: limita de 8 cereri/s e comună tuturor încărcărilor."""
    return sec_edgar.EdgarClient(user_agent)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def sec_load_manager(cik, user_agent, n=2):
    """Ultimele `n` trimestre 13F ale unui administrator. Întoarce (nume SEC, trimestre, probleme, eroare sau None)."""
    try:
        name, quarters, problems = sec_funds.load_manager(sec_client(user_agent), cik, n)
        return name, quarters, problems, None
    except (sec_edgar.EdgarError, ValueError) as e:   # rețea / XML: mesaj în pagină, restul fondurilor continuă
        logger.warning(f"13F {cik} nu a putut fi încărcat: {e}")
        return None, [], [], str(e)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def sec_load_cusip_map():
    """Foaia `CUSIP_map` → (dict CUSIP → simbol, eroare sau None). Foaia lipsă = hartă goală."""
    try:
        ws = connect_to_gsheets(cusip_map.MAP_SHEET)
        if ws is None:
            return {}, None
        return cusip_map.parse_map(ws.get_all_values()), None
    except gspread.exceptions.GSpreadException as e:
        logger.warning(f"CUSIP_map nu a putut fi citită: {e}")
        return {}, f"CUSIP_map nu a putut fi citită: {e}"


def sec_resolve_cusips(cusips):
    """Harta CUSIP → simbol pentru codurile date: foaia `CUSIP_map`, iar codurile lipsă de la OpenFIGI, scrise apoi în
    foaie (creată la prima scriere, aprobat 25.11.2026). Codurile negăsite nu se reîncearcă în aceeași sesiune.
    Întoarce (hartă, coduri netraduse, eroare sau None)."""
    cmap, err = sec_load_cusip_map()
    failed = st.session_state.setdefault("sec_figi_failed", set())
    missing = sorted(c for c in set(cusips) - set(cmap) if c not in failed)
    if missing:
        found, bad = cusip_map.openfigi_lookup(missing, api_key=sec_secret("openfigi_api_key"))
        failed.update(bad)
        if found:
            try:
                ws = connect_to_gsheets(cusip_map.MAP_SHEET)
                if ws is None:
                    ws = _get_spreadsheet().add_worksheet(cusip_map.MAP_SHEET, rows=2000, cols=len(cusip_map.MAP_COLUMNS))
                    ws.append_row(cusip_map.MAP_COLUMNS)
                ws.append_rows(cusip_map.map_rows(found), value_input_option="RAW")
                sec_load_cusip_map.clear()
            except (gspread.exceptions.GSpreadException, AttributeError) as e:
                logger.warning(f"CUSIP_map nu a putut fi scrisă: {e}")
                err = f"CUSIP_map nu a putut fi scrisă (traducerile rămân doar în sesiune): {e}"
            cmap = {**cmap, **{c: sym for c, (sym, _name) in found.items()}}
    return cmap, sorted(failed & set(cusips)), err


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def sec_prices(symbols, start):
    """Închiderile (auto_adjust=False; Yahoo le ajustează totuși pentru split) și split-urile de la `start` până azi.
    Întoarce (close, splits, eroare sau None)."""
    syms = list(symbols)
    try:
        raw = yf.download(syms, start=start, auto_adjust=False, actions=True, progress=False, threads=False)
    except Exception as e:   # rețea / Yahoo: mișcările rămân fără valoare tranzacționată
        logger.warning(f"13F: prețurile nu au venit: {e}")
        return None, None, str(e)
    close, splits = close_frame(raw, syms), field_frame(raw, "Stock Splits", syms)
    for fr in (close, splits):
        if fr is not None and not fr.empty and getattr(fr.index, "tz", None) is not None:
            fr.index = fr.index.tz_localize(None)
    return close, splits, None


def _sec_books(quarters):
    """(portofoliul curent, cel anterior, opțiunile curente) din trimestrele încărcate; lipsă = tabel gol."""
    empty = pd.DataFrame(columns=sec_13f.BOOK_COLUMNS)
    curr = sec_13f.equity_book(quarters[0]["info"], 1.0) if quarters else empty
    prev = sec_13f.equity_book(quarters[1]["info"], 1.0) if len(quarters) > 1 else empty
    opts = sec_13f.options_book(quarters[0]["info"], 1.0) if quarters else pd.DataFrame(columns=sec_13f.OPTION_COLUMNS)
    return curr, prev, opts


def _sec_moves(curr, prev, quarters, cmap, with_prices=True):
    """Mișcările între ultimele două trimestre, cu split-uri și prețuri medii din Yahoo când `with_prices`."""
    if len(quarters) < 2 or prev.empty:
        return None, None
    split, avg, perr = {}, {}, None
    if with_prices:
        c2s = {c: cmap.get(c) for c in set(curr["CUSIP"]) | set(prev["CUSIP"]) if cmap.get(c)}
        if c2s:
            start = (pd.Timestamp(quarters[1]["period"]) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            close, splits, perr = sec_prices(tuple(sorted(set(c2s.values()))), start)
            split, avg = sec_13f.period_inputs(close, splits, quarters[1]["period"], quarters[0]["period"], c2s)
    return sec_13f.quarter_moves(prev, curr, split_factor=split, avg_price=avg), perr


def _pct(v, dec=1):
    return format_ro(v * 100, dec, "%") if v is not None and pd.notna(v) else "N/A"


def _usd_big(v):
    if v is None or pd.isna(v):
        return "N/A"
    return format_ro(v / 1e9, 1, " mld. $") if abs(v) >= 1e9 else format_ro(v / 1e6, 1, " mil. $")


def sec_snapshot(ua):
    """Datele comune ale paginii Fonduri SUA, ale cardului din Analiză companie și ale semnalelor din Portofoliu:
    (administratori, încărcate, portofolii, hartă CUSIP, coduri netraduse, eroare hartă). Fondurile concentrate se
    încarcă automat (cache 24 h); cele Larg doar dacă au fost cerute în sesiune. Marchează sesiunea ca încărcată."""
    mg = sec_funds.managers_frame()
    larg_loaded = st.session_state.setdefault("sec_larg_loaded", set())
    to_load = [c for c in mg["CIK"] if mg.set_index("CIK").at[c, "Clasa"] == sec_funds.CONCENTRAT or c in larg_loaded]
    loaded = {}
    prog = st.progress(0.0, text="Rapoartele 13F de la SEC…")
    for i, cik in enumerate(to_load):
        loaded[cik] = sec_load_manager(cik, ua)
        prog.progress((i + 1) / len(to_load), text=f"Rapoartele 13F de la SEC… {i + 1}/{len(to_load)}")
    prog.empty()
    books = {cik: _sec_books(v[1]) for cik, v in loaded.items()}
    need = set()
    for cik, (curr, prev, _o) in books.items():
        is_larg = cik in larg_loaded
        for b in (curr, prev):
            need |= set(b.nsmallest(100, "Rang")["CUSIP"] if is_larg else b["CUSIP"])
    with st.spinner("Traduc codurile CUSIP în simboluri (OpenFIGI; prima dată poate dura câteva minute)…"):
        cmap, unmapped, cerr = sec_resolve_cusips(tuple(sorted(need)))
    st.session_state["sec_loaded"] = True
    return mg, loaded, books, cmap, unmapped, cerr


def _sec_conc_panels(mg, loaded, books, cmap):
    """(panoul deținerilor tuturor fondurilor încărcate, panoul mișcărilor concentraților — cu split-urile verificate,
    mulțimea CIK-urilor concentrate)."""
    holdings = sec_view.holdings_panel({c: b[0] for c, b in books.items()}, mg)
    moves = sec_view.moves_panel(_sec_conc_moves(mg, loaded, books, cmap))
    conc = set(mg.loc[mg["Clasa"] == sec_funds.CONCENTRAT, "CIK"]) & set(loaded)
    return holdings, moves, conc


SEC_SIGNAL_HELP = ("**Dețin:** fondurile concentrate urmărite care au acțiunea în ultimul raport 13F (clasele de acțiuni "
                   "adunate). **Convingere:** câte o au ca poziție de convingere (pondere ≥ max(1/N, 1%)). **Intrări / Măriri / "
                   "Reduceri / Ieșiri:** mișcările față de trimestrul anterior, pe acțiuni (> ±5%), cu split-urile "
                   "verificate la creșterile ≥ +50%.\n\n**Limite:** 13F e depus cu până la 45 de zile întârziere, doar "
                   "pozițiile long din SUA. Context, nu semnal de moment. Netestat.")


def _sec_signal_table(sig):
    st.dataframe(pd.DataFrame({
        "Simbol": sig["Simbol"], "Dețin": sig["Detin"], "Convingere": sig["Convingere"], "Intrări": sig["Intrari"],
        "Măriri": sig["Mariri"], "Reduceri": sig["Reduceri"], "Ieșiri": sig["Iesiri"], "Mișcări": sig["Miscari"],
        "Fonduri": sig["Fonduri"]}), hide_index=True, width='stretch',
        column_config={"Dețin": st.column_config.NumberColumn(help=SEC_SIGNAL_HELP, format="%d")})


def sec_my_us_symbols():
    """Simbolurile americane deținute acum, din registrul `Tranzactii` (moneda USD, fără sufix de bursă, fără cripto)."""
    values = load_ledger_values()
    if values is None:
        return []
    ledger_df, _ = ledger_calc.parse_ledger(values)
    if ledger_df.empty:
        return []
    positions, _ = ledger_calc.build_positions(ledger_df)
    held = positions[(positions["Quantity"] > ledger_calc.QTY_EPS) & positions["Currency"].astype(str).str.upper().eq("USD")]
    return sorted({str(x).upper() for x in held["Symbol"]
                   if "." not in str(x) and "=" not in str(x) and not str(x).upper().endswith("-USD")})


def render_sec_company_card(real_sym):
    """Card scurt „Fonduri SUA (13F)” în Analiză companie, doar pentru acțiunile americane (fără sufix de bursă).
    Datele se încarcă automat dacă au fost deja încărcate în sesiune; altfel la buton (prima încărcare durează)."""
    sym = str(real_sym).upper()
    if "." in sym or "=" in sym or "^" in sym or sym.endswith("-USD"):
        return
    ua = sec_secret("sec_user_agent")
    if not ua or "@" not in ua:
        return
    st.subheader("🏛️ Fonduri SUA (13F)", help=SEC_SIGNAL_HELP)
    if not st.session_state.get("sec_loaded"):
        st.caption("Rapoartele 13F nu sunt încărcate în această sesiune (prima încărcare durează ~20–60 s).")
        if not st.button("Încarcă fondurile care dețin acțiunea", key="sec_card_load"):
            return
        sec_snapshot(ua)          # încarcă și marchează sesiunea, apoi redesenează fără buton
        st.rerun()
    mg, loaded, books, cmap, _unmapped, _cerr = sec_snapshot(ua)
    holdings, moves, conc = _sec_conc_panels(mg, loaded, books, cmap)
    sig = sec_view.holdings_signals([sym], cmap, holdings, moves, conc).iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Dețin (concentrați)", f"{sig['Detin']} din {len(conc)}", help=SEC_SIGNAL_HELP)
    c2.metric("Cu convingere", f"{sig['Convingere']}", help=SEC_HELP["convingere"])
    c3.metric("Ultimul trimestru", f"+{sig['Intrari']} / ↑{sig['Mariri']} / ↓{sig['Reduceri']} / −{sig['Iesiri']}",
              help="Intrări (poziții noi) / măriri / reduceri / ieșiri (poziții închise) ale fondurilor concentrate. "
                   + SEC_HELP["miscari"])
    keys = sec_view.keys_for_symbol(sym, cmap, holdings, moves)
    if sig["Detin"]:
        w = pd.concat([sec_13f.who_holds(holdings[holdings["CIK"].isin(conc)], k) for k in keys], ignore_index=True)
        mv = sec_view.issuer_moves(moves[moves["CUSIP6"].isin(keys)]).groupby("CIK").agg(Mișcare=("Actiune", "first"), Delta=("Delta_pct", "first"))
        w = w.join(mv, on="CIK")
        tbl, fmt = na_safe_frame(pd.DataFrame({
            "Fond": w["Fond"], "Pondere %": w["Pondere"] * 100, "Rang": w["Rang"], "Convingere": w["Convingere"],
            "Mișcare": w["Mișcare"], "Δ acțiuni %": w["Delta"] * 100}),
            {"Pondere %": ro_spec(",.2f"), "Δ acțiuni %": ro_spec("+,.1f")})
        st.dataframe(tbl.style.format(fmt),
            hide_index=True, width='stretch',
            column_config={"Convingere": st.column_config.CheckboxColumn(help=SEC_HELP["convingere"])})
    elif sig["Iesiri"]:
        st.caption(f"Niciun fond concentrat nu mai deține {sym}; ieșiri în ultimul trimestru: {sig['Miscari']}.")
    else:
        st.caption(f"{sym} nu apare în rapoartele fondurilor concentrate urmărite (sau codul CUSIP nu a fost tradus).")
    periods = sorted({q[0]["period"] for _n, q, _p, _e in loaded.values() if q})
    st.caption(f"Rapoarte 13F pentru {', '.join(periods[-2:]) if periods else 'N/A'} · detalii: meniul „Fonduri SUA (13F)”.")


INSIDER_HELP = {
    "card": ("Tranzacțiile insiderilor (directori, ofițeri, acționari > 10%) din formularele SEC Form 4, depuse în "
             "2 zile lucrătoare. Fereastra: ultimele 90 de zile, după data tranzacției.\n\n"
             "**În semnal intră doar** cumpărările pe piață (cod P) și vânzările (cod S). Vânzările cu bifa 10b5-1 "
             "(sau cu nota „10b5-1”) sunt **prin plan**, stabilite dinainte, deci aproape fără informație. "
             "Exercitările (M), impozitele reținute (F), acordările (A) și cadourile (G) apar în tabel, nu în semnal.\n\n"
             "Un 4/A înlocuiește raportul inițial al aceluiași insider pentru aceeași dată (proxy). "
             "**Netestat:** context, nu verdict. Literatura: cumpărările pe piață, mai ales în grup, au avut putere "
             "de predicție; vânzările mult mai puțin."),
    "p": ("Cumpărări pe piață (cod P) în 90 de zile: numărul de tranzacții, câți insideri distincți (un raport "
          "depus în comun = un insider) și valoarea = acțiuni × prețul raportat. Tranzacțiile fără preț nu intră în "
          "valoare."),
    "s": ("Vânzări (cod S) fără bifa sau nota 10b5-1: decise de insider la momentul vânzării. Pot avea motive "
          "personale (impozite, diversificare), deci sunt un semnal mai slab decât cumpărările."),
    "plan": ("Vânzări prin plan 10b5-1: programate dinainte (ex. aceeași cantitate în fiecare săptămână). "
             "Nu spun nimic despre ce crede insiderul acum."),
    "grup": ("Grup = cel puțin 3 insideri distincți au cumpărat pe piață în 90 de zile. % din deținere = mediana "
             "cumpărării raportate la deținerea dinaintea ei (doar cumpărările cu deținere anterioară > 0)."),
}


def _usd_small(v):
    if v is None or pd.isna(v):
        return "N/A"
    return _usd_big(v) if abs(v) >= 1e6 else format_ro(v / 1e3, 0, " mii $")


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def sec_load_insiders(symbol, user_agent, today_iso):
    """Form 4 pentru simbolul american, depuse în ultimele 95 de zile. Cache 12 h (cheia include ziua).
    Întoarce (tabel tranzacții, nr. rapoarte, probleme, trunchiat, eroare sau None)."""
    empty = insiders.transactions_frame([])
    try:
        client = sec_client(user_agent)
        cik = sec_insiders.ticker_cik(client, symbol)
        if cik is None:
            return empty, 0, [], False, f"{symbol} nu apare în lista SEC de simboluri"
        docs, problems, truncated = sec_insiders.load_form4(client, cik, today=pd.Timestamp(today_iso).date())
    except (sec_edgar.EdgarError, ValueError) as exc:
        logger.warning("Form 4 %s: %s", symbol, exc)
        return empty, 0, [], False, str(exc)
    return insiders.transactions_frame(docs), len(docs), problems, truncated, None


def render_sec_insiders_card(real_sym):
    """Card „Insideri (Form 4, 90 de zile)” în Analiză companie, doar pentru acțiunile americane.
    Se încarcă la buton (prima dată ~5–15 s); după aceea simbolul rămâne încărcat în sesiune (cache 12 h)."""
    sym = str(real_sym).upper()
    if "." in sym or "=" in sym or "^" in sym or sym.endswith("-USD"):
        return
    ua = sec_secret("sec_user_agent")
    if not ua or "@" not in ua:
        return
    st.subheader("🧑‍💼 Insideri (Form 4, 90 de zile)", help=INSIDER_HELP["card"])
    loaded = st.session_state.setdefault("ins_loaded", set())
    if sym not in loaded:
        st.caption("Tranzacțiile insiderilor nu sunt încărcate pentru acest simbol (~5–15 s).")
        if not st.button("Încarcă tranzacțiile insiderilor", key=f"ins_load_{sym}"):
            return
        loaded.add(sym)
        st.rerun()                # redesenează fără buton; datele se încarcă la rularea următoare
    today = now_ro().date()
    with st.spinner("Citesc rapoartele Form 4 de la SEC..."):
        df, n_docs, problems, truncated, err = sec_load_insiders(sym, ua, today.isoformat())
    if err:
        st.caption(f"Form 4 N/A: {err}")
        return
    s = insiders.window_summary(df, today)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cumpărări pe piață (P)", f"{s['nr_P']} · {s['cumparatori']} insideri", _usd_small(s["valoare_P"]),
              delta_color="off", help=INSIDER_HELP["p"])
    c2.metric("Vânzări discreționare", f"{s['nr_S_discretionar']}", _usd_small(s["valoare_S_discretionar"]),
              delta_color="off", help=INSIDER_HELP["s"])
    c3.metric("Vânzări prin plan 10b5-1", f"{s['nr_S_plan']}", _usd_small(s["valoare_S_plan"]),
              delta_color="off", help=INSIDER_HELP["plan"])
    c4.metric("Grup de cumpărători", "DA" if s["grup"] else "NU",
              f"% din deținere: {_pct(s['pct_din_detinere_P'])}" if s["pct_din_detinere_P"] is not None else None,
              delta_color="off", help=INSIDER_HELP["grup"])
    st.caption(insiders.describe(s, _usd_small).replace("$", "\\$"))   # „$ … $” ar fi citit ca formulă LaTeX
    other = ", ".join(f"{k} ({insiders.CODE_LABELS.get(k, k)}): {v}" for k, v in sorted(s["alte_coduri"].items()))
    notes = [f"{s['start']:%d.%m.%Y} → {s['end']:%d.%m.%Y}", f"{n_docs} rapoarte Form 4",
             f"în afara semnalului: {other}" if other else "",
             f"tranzacții P/S fără preț: {s['fara_pret']}" if s["fara_pret"] else "",
             f"{len(problems)} rapoarte nu s-au putut citi" if problems else "",
             "doar cele mai noi 80 de rapoarte" if truncated else "", "netestat: context, nu verdict"]
    st.caption(" · ".join(x for x in notes if x))
    w = df[(df["Data_tranz"] > s["start"]) & (df["Data_tranz"] <= s["end"])] if not df.empty else df
    if not w.empty:
        with st.expander(f"Toate tranzacțiile ({len(w)})"):
            # text_frame: st.dataframe arată celulele nule ca „None” chiar cu Styler(na_rep); lipsa devine „N/A”
            st.dataframe(text_frame(pd.DataFrame({
                "Data": w["Data_tranz"].dt.strftime("%d.%m.%Y"), "Depus": w["Data_depunere"].dt.strftime("%d.%m.%Y"),
                "Insider": w["Insider"], "Funcție": w["Functie"], "Cod": w["Cod"], "Tip": w["Tip"],
                "Plan 10b5-1": w["Plan_10b5_1"], "Acțiuni": w["Actiuni"].astype(float), "Preț $": w["Pret"].astype(float),
                "Valoare $": w["Valoare_USD"].astype(float), "Deținere după": w["Detinere_dupa"].astype(float)}),
                {"Acțiuni": ro_spec(",.0f"), "Preț $": ro_spec(",.2f"), "Valoare $": ro_spec(",.0f"),
                 "Deținere după": ro_spec(",.0f")}),
                hide_index=True, width='stretch',
                column_config={"Plan 10b5-1": st.column_config.CheckboxColumn(help=INSIDER_HELP["plan"])})


HOLDERS_HELP = {
    "card": ("Cine deține peste 5% dintr-o clasă de acțiuni cu drept de vot, din formularele SEC Schedule 13D și "
             "13G despre companie, depuse în ultimii 3 ani. Pentru fiecare depunător contează **ultima** lui "
             "depunere.\n\n**13D = activist:** poate urmări influențarea companiei (consiliu, strategie, vânzare); "
             "depus în 5 zile lucrătoare de la prag. **13G = pasiv:** fonduri index, instituții, investitori fără "
             "intenție de control; termene mai lungi (45 de zile după trimestru la instituții).\n\n"
             "O depunere în grup (fondul, administratorul, persoanele care îl controlează) raportează aceleași "
             "acțiuni de mai multe ori: procentul arătat e **maximul** pe persoane, nu suma. Depunerile dinainte de "
             "18.12.2024 nu au format structurat: apar cu numele, procent N/A.\n\n**Netestat:** context, nu verdict."),
    "peste": ("Depunătorii a căror ultimă depunere arată ≥ 5% (sau procent N/A, la depunerile vechi). Cei care au "
              "raportat ieșirea sub 5% sunt numărați separat.\n\n**Procent N/A** = ultima depunere e dinainte de "
              "18.12.2024 (fără format structurat). Deținerea poate fi acum și sub 5%: din 2024, un amendament se "
              "depune doar la o schimbare semnificativă. Verifică în cardul 13F sau pe sec.gov."),
    "activisti": ("Depunătorii peste prag cu ultima depunere 13D. „13G → 13D” = un investitor pasiv a devenit "
                  "activist: în literatură, unul dintre cele mai urmărite evenimente din 13D."),
    "noi": "13D noi (nu amendamente) sau treceri 13G → 13D depuse în ultimele 90 de zile.",
}


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def sec_load_holders(symbol, user_agent, today_iso):
    """13D/13G despre simbolul american, depuse în ultimii 3 ani. Cache 12 h (cheia include ziua).
    Întoarce (tabel deținători, nr. depuneri citite, probleme, trunchiat, eroare sau None)."""
    empty = schedule13.current_holders([])
    try:
        client = sec_client(user_agent)
        cik = sec_insiders.ticker_cik(client, symbol)
        if cik is None:
            return empty, 0, [], False, f"{symbol} nu apare în lista SEC de simboluri"
        docs, problems, truncated = sec_ownership.load_schedules(client, cik, today=pd.Timestamp(today_iso).date())
    except (sec_edgar.EdgarError, ValueError) as exc:
        logger.warning("13D/G %s: %s", symbol, exc)
        return empty, 0, [], False, str(exc)
    return schedule13.current_holders(docs, issuer_cik=cik), len(docs), problems, truncated, None


def render_sec_holders_card(real_sym):
    """Card „Deținători peste 5% (13D/13G)” în Analiză companie, doar pentru acțiunile americane.
    Se încarcă la buton (prima dată ~3–10 s); după aceea simbolul rămâne încărcat în sesiune (cache 12 h)."""
    sym = str(real_sym).upper()
    if "." in sym or "=" in sym or "^" in sym or sym.endswith("-USD"):
        return
    ua = sec_secret("sec_user_agent")
    if not ua or "@" not in ua:
        return
    st.subheader("🗂️ Deținători peste 5% (13D/13G)", help=HOLDERS_HELP["card"])
    loaded = st.session_state.setdefault("own_loaded", set())
    if sym not in loaded:
        st.caption("Depunerile 13D/13G nu sunt încărcate pentru acest simbol (~3–10 s).")
        if not st.button("Încarcă deținătorii peste 5%", key=f"own_load_{sym}"):
            return
        loaded.add(sym)
        st.rerun()                # redesenează fără buton; datele se încarcă la rularea următoare
    today = now_ro().date()
    with st.spinner("Citesc depunerile 13D/13G de la SEC..."):
        h, n_docs, problems, truncated, err = sec_load_holders(sym, ua, today.isoformat())
    if err:
        st.caption(f"13D/13G N/A: {err}")
        return
    s = schedule13.holders_summary(h, pd.Timestamp(today))
    c1, c2, c3 = st.columns(3)
    c1.metric("Peste 5%", f"{s['peste_prag']}",
              f"{s['fara_procent']} cu procent N/A" if s["fara_procent"] else None, delta_color="off",
              help=HOLDERS_HELP["peste"])
    c2.metric("Activiști (13D)", f"{s['activisti']}", f"pasivi (13G): {s['pasivi']}", delta_color="off",
              help=HOLDERS_HELP["activisti"])
    c3.metric("13D nou (90 de zile)", "DA" if s["d13_noi"] else "NU",
              ", ".join(str(x) for x in s["d13_noi"])[:60] or None, delta_color="off", help=HOLDERS_HELP["noi"])
    last = s["ultima_depunere"]
    notes = [f"{n_docs} depuneri 13D/13G citite (ultimii 3 ani)",
             f"ultima: {last:%d.%m.%Y}" if last is not None else "",
             f"ieșiți sub 5%: {s['sub_prag']}" if s["sub_prag"] else "",
             f"{len(problems)} depuneri nu s-au putut citi" if problems else "",
             "doar cele mai noi 60 de depuneri" if truncated else "", "netestat: context, nu verdict"]
    st.caption(" · ".join(x for x in notes if x))
    if h.empty:
        st.caption(f"Nicio depunere 13D/13G despre {sym} în ultimii 3 ani.")
        return
    st.dataframe(text_frame(schedule13.display_frame(h),
        {"Procent": lambda v: format_ro(v, 1, "%"), "Acțiuni": lambda v: format_ro(v, 0),
         "Eveniment": lambda d: f"{d:%d.%m.%Y}", "Depus": lambda d: f"{d:%d.%m.%Y}"}),
        hide_index=True, width='stretch')
    if problems:
        with st.expander(f"Depuneri necitite ({len(problems)})"):
            for p in problems:
                st.caption(f"• {p}")


def render_sec_portfolio_signals():
    """Expander în Portofoliu: semnalele 13F pe acțiunile americane deținute (fondurile concentrate)."""
    ua = sec_secret("sec_user_agent")
    if not ua or "@" not in ua:
        return
    with st.expander("🏛️ Semnale SEC pe deținerile mele (fonduri SUA, 13F)"):
        syms = sec_my_us_symbols()
        if not syms:
            st.caption("Nicio acțiune americană deschisă în registrul `Tranzactii`.")
            return
        if not st.session_state.get("sec_loaded"):
            st.caption(f"Rapoartele 13F nu sunt încărcate în această sesiune ({len(syms)} acțiuni americane deținute).")
            if not st.button("Încarcă rapoartele 13F", key="sec_pf_load"):
                return
            sec_snapshot(ua)
            st.rerun()
        mg, loaded, books, cmap, _unmapped, _cerr = sec_snapshot(ua)
        holdings, moves, conc = _sec_conc_panels(mg, loaded, books, cmap)
        _sec_signal_table(sec_view.holdings_signals(syms, cmap, holdings, moves, conc))
        st.caption(f"Fondurile concentrate urmărite ({len(conc)}); acțiunile americane deschise din registrul `Tranzactii`. "
                   "Context, nu semnal. Detalii: meniul „Fonduri SUA (13F)”.")


def render_sec_funds_page():
    st.title("🏛️ Fonduri SUA (13F)")
    st.caption(SEC_NOTE)
    ua = sec_secret("sec_user_agent")
    if not ua or "@" not in ua:
        st.info("Lipsește `sec_user_agent` în Secrets (ex. „Terminal Investitii nume@domeniu.ro”): SEC cere un contact.")
        return
    if st.button("🔄 Reîncarcă de la SEC", key="sec_reload"):
        sec_load_manager.clear()
        sec_load_cusip_map.clear()
        st.session_state.pop("sec_figi_failed", None)
        st.rerun()

    mg, loaded, books, cmap, unmapped, cerr = sec_snapshot(ua)
    if cerr:
        st.warning(cerr)
    if unmapped:
        st.caption(f"{len(unmapped)} coduri CUSIP fără simbol american la OpenFIGI: apar cu numele emitentului.")

    t_over, t_fund, t_cons, t_who = st.tabs([":material/list: Prezentare", ":material/account_balance: Fond",
                                             ":material/groups: Consens", ":material/person_search: Cine deține"])
    with t_over:
        _sec_overview(mg, loaded, books, cmap)
    with t_fund:
        _sec_fund(mg, loaded, books, cmap, ua)
    with t_cons:
        _sec_consensus(mg, loaded, books, cmap)
    with t_who:
        _sec_who(mg, loaded, books, cmap)


def _dash(col, fmt=str):
    """Coloană text pentru afișare: valorile lipsă (fond nedescărcat) apar ca „—”, nu „None”."""
    if col is None:
        return None
    return pd.Series([fmt(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else "—" for v in col],
                     index=col.index, dtype=object)


def _sec_overview(mg, loaded, books, cmap):
    rows, warn = [], []
    for _, m in mg.iterrows():
        cik = m["CIK"]
        if cik not in loaded:
            rows.append({**m.to_dict(), "Nume_SEC": None, "Perioada": None, "Pozitii": np.nan,
                         "Valoare_USD": np.nan, "Top": "nedescărcat (Larg) — tabul Fond", "Top_pondere": np.nan})
            continue
        name, quarters, problems, err = loaded[cik]
        curr, prev, _o = books[cik]
        r = sec_view.overview_row(m, name, quarters[0] if quarters else None, prev, curr, cmap)
        rows.append({**r, "CIK": cik})
        if err:
            warn.append(f"{m['Fond']}: {err}")
        if r["Nume_ok"] is False:
            warn.append(f"{m['Fond']}: la CIK {cik} SEC are numele „{name}” — CIK-ul trebuie verificat.")
        warn += [f"{m['Fond']}: {p}" for p in problems + [p for q in quarters for p in q["problems"]]]
        warn += [f"{m['Fond']}: {p}" for q in quarters for p in q.get("notes", [])]
    df = pd.DataFrame(rows)
    stale = sec_view.stale_periods(df["Perioada"].tolist())
    for f, p in zip(df["Fond"], df["Perioada"]):
        if p in stale:
            warn.append(f"{f}: ultimul raport e pentru {p}, rămas în urmă față de ceilalți — fondul nu a depus încă "
                        "raportul nou (depunere întârziată) sau depune sub alt CIK; verifică pe EDGAR.")
    view = pd.DataFrame({
        "Fond": df["Fond"], "Administrator": df["Administrator"], "Stil": df["Stil"],
        "Clasa": [c if not isinstance(a, str) or a == c else f"{c} (auto: {a})"
                  for c, a in zip(df["Clasa"], df.get("Clasa_auto", pd.Series([None] * len(df))))],
        "Perioada": _dash(df["Perioada"]), "Depus": _dash(df.get("Depus")),
        "Decalaj (zile)": _dash(df.get("Decalaj_zile"), lambda v: format_ro(v, 0)),
        "Poziții": _dash(df["Pozitii"], lambda v: format_ro(v, 0)), "Valoare": df["Valoare_USD"].map(_usd_big),
        "Top deținere": [f"{t} ({_pct(p)})" if pd.notna(p) else t for t, p in zip(df["Top"], df["Top_pondere"])],
    })
    st.dataframe(view, hide_index=True, width='stretch',
                 column_config={"Poziții": st.column_config.TextColumn(help=SEC_HELP["pozitii"]),
                                "Clasa": st.column_config.TextColumn(help=SEC_HELP["clasa"]),
                                "Perioada": st.column_config.TextColumn(help=SEC_HELP["perioada"]),
                                "Valoare": st.column_config.TextColumn(help=SEC_HELP["valoare"]),
                                "Top deținere": st.column_config.TextColumn(help=SEC_HELP["top"])})
    if warn:
        with st.expander(f"⚠️ Probleme la încărcare ({len(warn)})"):
            for w in warn:
                st.write("• " + w)


def _sec_fund(mg, loaded, books, cmap, ua):
    labels = {r["CIK"]: f"{r['Fond']} — {r['Administrator']}" + (" (Larg)" if r["Clasa"] == sec_funds.LARG else "")
              for _, r in mg.iterrows()}
    cik = st.selectbox("Fondul", list(labels), format_func=labels.get, key="sec_fund_pick")
    meta = mg.set_index("CIK").loc[cik]
    if cik not in loaded:
        st.info("Fond „Larg” (cantitativ / multi-strategie): mii de poziții, fără convingere; nu intră în consens. "
                   "Încărcarea poate dura 1–2 minute.")
        if st.button("Încarcă raportul", key=f"sec_load_{cik}"):
            st.session_state["sec_larg_loaded"].add(cik)
            st.rerun()
        return
    name, quarters, problems, err = loaded[cik]
    if err or not quarters:
        st.info(err or "Niciun raport 13F găsit.")
        return
    q = quarters[0]
    curr, prev, opts = books[cik]
    if curr.empty:
        st.warning("Raportul curent nu a trecut validarea: " + "; ".join(q["problems"]))
        return
    top = curr.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Poziții", format_ro(len(curr), 0), help=SEC_HELP["pozitii"])
    c2.metric("Valoare (acțiuni)", _usd_big(curr["Valoare_USD"].sum()), help=SEC_HELP["valoare"])
    c3.metric("Top deținere", cmap.get(top["CUSIP"]) or str(top["Emitent"])[:18], _pct(top["Pondere"]),
              delta_color="off", help=SEC_HELP["top"])
    c4.metric("Perioada / depus", str(q["period"]), f"depus {q['last_filing_date']} · "
              f"{sec_view.delay_days(q['period'], q['filing_date'])} zile", delta_color="off", help=SEC_HELP["perioada"])
    st.caption(f"{meta['Stil']} · la SEC: {name} (CIK {cik}). Clasa calculată: {sec_13f.classify_manager(curr)}.")

    h = sec_view.with_symbols(curr, cmap)
    h["Convingere"] = sec_13f.conviction_mask(h)
    st.markdown("**Deținerile pe acțiuni**", help="Convingerea se judecă pe emitent: clasele aceluiași emitent "
                "(ex. GOOG + GOOGL) se adună și primesc aceeași bifă, ca în consens și în „Cine deține”.")
    st.dataframe(pd.DataFrame({
        "#": h["Rang"], "Simbol": h["Symbol"].fillna("—"), "Emitent": h["Emitent"], "Clasă": h["Clasa_titlu"],
        "Pondere %": h["Pondere"] * 100, "Valoare (mil. $)": h["Valoare_USD"] / 1e6, "Acțiuni": h["Actiuni"],
        "Convingere": h["Convingere"]}).style.format({
            "Pondere %": ro_spec(",.2f"), "Valoare (mil. $)": ro_spec(",.1f"), "Acțiuni": ro_spec(",.0f")}),
        hide_index=True, width='stretch', height=420,
        column_config={"Convingere": st.column_config.CheckboxColumn(help=SEC_HELP["convingere"])})
    if not opts.empty:
        st.markdown("**Opțiuni în raport** (valoarea acțiunilor din spate, cum a raportat-o fondul; nu intră în ponderi)")
        o = sec_view.with_symbols(opts, cmap)
        st.dataframe(pd.DataFrame({"Simbol": o["Symbol"].fillna("—"), "Emitent": o["Emitent"], "Tip": o["PutCall"],
                                   "Valoare (mil. $)": o["Valoare_USD"] / 1e6, "Acțiuni": o["Actiuni"]}).style.format(
            {"Valoare (mil. $)": ro_spec(",.1f"), "Acțiuni": ro_spec(",.0f")}), hide_index=True, width='stretch')

    is_larg = meta["Clasa"] == sec_funds.LARG
    moves, perr = _sec_moves(curr, prev, quarters, cmap, with_prices=not is_larg)
    st.markdown(f"**Mișcări față de trimestrul anterior** ({quarters[1]['period'] if len(quarters) > 1 else 'N/A'} → "
                f"{q['period']}; doar acțiunile, fără opțiuni)", help=SEC_HELP["miscari"])
    if moves is None:
        st.info("Nu există un raport anterior valid pentru comparație.")
        return
    cnt = sec_13f.move_counts(moves)
    ok = cnt["Nouă"] + cnt["Mărită"] + cnt["Redusă"] + cnt["Păstrată"] == len(curr)
    st.caption(f"{cnt['Nouă']} noi · {cnt['Închisă']} închise · {cnt['Mărită']} mărite · {cnt['Redusă']} reduse · "
               f"{cnt['Păstrată']} păstrate (±5% în acțiuni)" + (" · verificare: închide ✓" if ok else " · verificare: NU închide"))
    if perr:
        st.caption(f"Prețurile Yahoo nu au venit ({perr}): fără split-uri și fără valoare tranzacționată.")
    elif is_larg:
        st.caption("Fond „Larg”: fără prețuri și split-uri (prea multe titluri); mișcările nu sunt ajustate pentru split.")
    mv = sec_view.with_symbols(moves, cmap)
    tbl, fmt = na_safe_frame(pd.DataFrame({
        "Mișcare": mv["Actiune"], "Simbol": mv["Symbol"].fillna("—"), "Emitent": mv["Emitent"],
        "Acțiuni înainte": mv["Actiuni_ant"], "Acțiuni acum": mv["Actiuni"], "Δ acțiuni %": mv["Delta_pct"] * 100,
        "Pondere înainte %": mv["Pondere_ant"] * 100, "Pondere acum %": mv["Pondere"] * 100, "Δ pp": mv["Delta_pp"],
        "Preț mediu (proxy)": mv["Pret_mediu"], "Valoare tranz. (mil. $, proxy)": mv["Valoare_tranzactionata"] / 1e6,
        "Impact %": mv["Impact"] * 100}), {
            "Acțiuni înainte": ro_spec(",.0f"), "Acțiuni acum": ro_spec(",.0f"), "Δ acțiuni %": ro_spec(",.1f"),
            "Pondere înainte %": ro_spec(",.2f"), "Pondere acum %": ro_spec(",.2f"), "Δ pp": ro_spec("+,.2f"),
            "Preț mediu (proxy)": ro_spec(",.2f"), "Valoare tranz. (mil. $, proxy)": ro_spec("+,.1f"),
            "Impact %": ro_spec(",.2f")})
    st.dataframe(tbl.style.format(fmt), hide_index=True, width='stretch', height=420)


def _sec_conc_moves(mg, loaded, books, cmap):
    """Mișcările administratorilor concentrați; cei cu creșteri ≥ +50% sunt recalculați cu split-urile Yahoo."""
    conc = set(mg.loc[mg["Clasa"] == sec_funds.CONCENTRAT, "CIK"]) & set(loaded)
    raw = {cik: _sec_moves(books[cik][0], books[cik][1], loaded[cik][1], cmap, with_prices=False)[0] for cik in conc}
    cands = sec_view.split_check_cusips(sec_view.moves_panel(raw))
    c2s = {c: cmap[c] for c in cands if cmap.get(c)}
    if not c2s:
        return raw
    periods = [loaded[c][1][1]["period"] for c in conc if len(loaded[c][1]) > 1]
    start = (pd.Timestamp(min(periods)) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    close, splits, _err = sec_prices(tuple(sorted(set(c2s.values()))), start)
    out = {}
    for cik, m in raw.items():
        qs = loaded[cik][1]
        if m is None or not set(m["CUSIP"]) & set(c2s):
            out[cik] = m
            continue
        split, _avg = sec_13f.period_inputs(close, splits, qs[1]["period"], qs[0]["period"], c2s)
        out[cik] = sec_13f.quarter_moves(books[cik][1], books[cik][0], split_factor=split)
    return out


def _sec_consensus(mg, loaded, books, cmap):
    st.caption("Doar administratorii concentrați. Fondurile au perioade de raportare apropiate, dar nu neapărat identice "
               "(vezi Prezentare).")
    conc_mg = mg[mg["Clasa"] == sec_funds.CONCENTRAT]
    holdings = sec_view.holdings_panel({c: books[c][0] for c in conc_mg["CIK"] if c in books}, conc_mg)
    if holdings.empty:
        st.info("Nu există deținuri încărcate.")
        return
    moves = sec_view.moves_panel(_sec_conc_moves(mg, loaded, books, cmap))
    cons = sec_13f.consensus(holdings, moves)
    sym6 = sec_view.symbols_by_key(cmap, holdings)
    cons["Simbol"] = cons["CUSIP6"].map(sym6).fillna("—")
    cfg = {"Dețin": st.column_config.NumberColumn(help=SEC_HELP["consens"], format="%d"),
           "Convingere": st.column_config.NumberColumn(help=SEC_HELP["consens"], format="%d"),
           "Cumpărări": st.column_config.NumberColumn(help=SEC_HELP["consens"], format="%d")}

    def _show(df):
        st.dataframe(pd.DataFrame({"Simbol": df["Simbol"], "Emitent": df["Emitent"], "Dețin": df["Detinatori"],
                                   "Convingere": df["Convingere"], "Cumpărări": df["Cumparari_convingere"],
                                   "Fonduri": df["Fonduri"]}), hide_index=True, width='stretch', column_config=cfg)

    st.markdown(f"**Poziții de convingere la ≥ 2 administratori** (din {conc_mg['CIK'].isin(list(books)).sum()} concentrați)",
                help=SEC_HELP["consens"])
    _show(cons[cons["Convingere"] >= 2].head(40))
    st.markdown("**Cumpărări de convingere în ultimul trimestru** (nouă sau ≥ +50% în acțiuni, pondere ≥ 1%)",
                help=SEC_HELP["consens"])
    buys = cons[cons["Cumparari_convingere"] >= 1].sort_values(["Cumparari_convingere", "Convingere"], ascending=False)
    _show(buys.head(40))


def _sec_who(mg, loaded, books, cmap):
    sym = st.text_input("Simbol (SUA)", value="AAPL", key="sec_who_sym").strip().upper()
    holdings, moves, conc = _sec_conc_panels(mg, loaded, books, cmap)
    keys = sec_view.keys_for_symbol(sym, cmap, holdings, moves)
    if not sym:
        return
    w = pd.concat([sec_13f.who_holds(holdings, k) for k in keys], ignore_index=True) if keys else pd.DataFrame()
    if w.empty:
        st.info(f"{sym} nu apare în deținerile fondurilor încărcate (sau codul lui CUSIP nu a fost tradus).")
    else:
        mv = sec_view.issuer_moves(moves[moves["CUSIP6"].isin(keys)]).groupby("CIK").agg(Mișcare=("Actiune", "first"), Delta=("Delta_pct", "first"))
        w = w.join(mv, on="CIK")
        n_conc = int((w["Clasa"] == sec_funds.CONCENTRAT).sum())
        st.markdown(f"**{n_conc} administratori concentrați dețin {sym}** (din {len(conc)} urmăriți)")
        tbl, fmt = na_safe_frame(pd.DataFrame({
            "Fond": w["Fond"], "Clasa": w["Clasa"], "Pondere %": w["Pondere"] * 100, "Rang": w["Rang"],
            "Valoare (mil. $)": w["Valoare_USD"] / 1e6, "Acțiuni": w["Actiuni"], "Convingere": w["Convingere"],
            "Mișcare": w["Mișcare"], "Δ acțiuni %": w["Delta"] * 100}), {
                "Pondere %": ro_spec(",.2f"), "Valoare (mil. $)": ro_spec(",.1f"), "Acțiuni": ro_spec(",.0f"),
                "Δ acțiuni %": ro_spec("+,.1f")})
        st.dataframe(tbl.style.format(fmt),
            hide_index=True, width='stretch',
            column_config={"Convingere": st.column_config.CheckboxColumn(help=SEC_HELP["convingere"]),
                           "Clasa": st.column_config.TextColumn(help=SEC_HELP["clasa"])})
        st.caption("Mișcarea e doar pentru fondurile concentrate (split-uri verificate la creșterile ≥ +50%).")

    with st.expander("Deținerile mele în portofoliile fondurilor concentrate"):
        syms = sec_my_us_symbols()
        if syms:
            _sec_signal_table(sec_view.holdings_signals(syms, cmap, holdings, moves, conc))
            st.caption("Acțiunile americane deschise din registrul `Tranzactii` (moneda USD, fără sufix de bursă).")
        else:
            st.caption("Nicio acțiune americană deschisă în registrul `Tranzactii`.")


# ==================================================
# CREȘTERE ȘI SURSA RANDAMENTULUI (CAGR) — decizia lui Daniel, 25.11.2026: A + B + C
# ==================================================
GROWTH_HELP = {
    "card": ("**Ce arată:** cât a crescut afacerea (A), cât a câștigat acționarul (B) și de unde a venit câștigul (C).\n\n"
             "**CAGR** = (valoarea finală ÷ valoarea inițială)^(1/n) − 1. Un capăt lipsă sau ≤ 0 (pierdere, FCF negativ) "
             "dă N/A. Informativ: nu intră în scor (momentul e deja în pasul tehnic)."),
    "a": ("**Pe acțiune** = indicatorul ÷ numărul de acțiuni (diluate la SUA/UE; emise, din fișier, la BVB). Contează "
          "pentru acționar: diluarea îl scade, răscumpărările îl cresc. **Total** = suma pentru toată compania.\n\n"
          "Dacă numărul de acțiuni s-a schimbat mult între capete (raport în afara 0,8–1,25: split, acțiuni gratuite, "
          "majorare), valorile pe acțiune sunt N/A (nu se „corectează”).\n\n"
          "**Surse:** SUA (emitenți 10-K) — rapoartele 10-K depuse la SEC (XBRL), până la ~19 ani: 3, 5 și 10 ani; "
          "valorile pe acțiune = totalul ÷ acțiunile diluate, ajustate cu split-urile de după depunere; FCF pe medii de "
          "3 ani la capete; cifra de afaceri SEC = eticheta `Revenues` (venituri totale; la unii emitenți include și alte "
          "venituri — ex. XOM, cu 2–4% peste „Sales” de la Yahoo; la bănci = venituri nete); **Consecvență** = anii cu creștere față de anul anterior din ultimii 10; o etichetă XBRL "
          "schimbată cu valori diferite în interval → N/A. UE și emitenții străini (20-F) — situațiile anuale Yahoo "
          "(~4 ani → 3 ani). "
          "Dividendul SUA/UE — plățile Yahoo, pe ani calendaristici încheiați. BVB — fișierele „Analiză companii” "
          "(foaia anuală): dividend pe acțiune = profit distribuit ÷ acțiuni emise."),
    "b": ("**Randament total anualizat** = CAGR-ul prețului ajustat pentru dividende și split (dividendele reinvestite), "
          "pe ani calendaristici până la ultima cotație. **Benchmark** din aceeași piață și valută, tot cu dividende. "
          "**Diferența** e în puncte procentuale pe an. Randamentul trecut e context, nu semnal."),
    "c": ("**Identitate:** (1 + randament total) = (1 + creșterea EPS) × (1 + schimbarea P/E) × (1 + contribuția "
          "dividendului), între cele două date fiscale (prețul de la data situației). Creșterea EPS e partea care se "
          "poate repeta; schimbarea P/E e re-evaluarea pieței (nu se repetă la nesfârșit). Suma componentelor e o "
          "aproximare a totalului: diferența se arată."),
}


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def get_growth_history(symbol):
    """Istoricul pe 11 ani: Close (ajustat doar pentru split), Adj Close (și pentru dividende), Dividends.
    Întoarce (DataFrame, eroare sau None)."""
    if not symbol:
        return pd.DataFrame(), "fără simbol"
    try:
        h = yf.Ticker(symbol).history(period="11y", auto_adjust=False, actions=True)
    except Exception as e:   # rețea / Yahoo: cardul arată N/A
        logger.warning(f"istoricul de creștere {symbol}: {e}")
        return pd.DataFrame(), str(e)
    if h is None or h.empty or "Close" not in h:
        return pd.DataFrame(), "Yahoo nu a trimis istoricul"
    if getattr(h.index, "tz", None) is not None:
        h.index = h.index.tz_localize(None)
    if "Adj Close" not in h:
        return pd.DataFrame(), "Yahoo nu a trimis prețul ajustat pentru dividende"
    return h, None


def _g(v):
    return _pct(v) if v is not None else "N/A"


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def sec_annual_table(symbol, user_agent):
    """Tabelul anual SEC (XBRL, ultima versiune) al unui emitent american: (tabel, tip emitent, eroare sau None).
    Tip: "10-K" / "20-F" / None. Split-urile vin din Yahoo; fără ele valorile pe acțiune vechi ar fi greșite → eroare."""
    try:
        cik, cf = sec_xbrl.load_for_symbol(sec_client(user_agent), symbol)
    except sec_edgar.EdgarError as exc:
        logger.warning("XBRL %s: %s", symbol, exc)
        return pd.DataFrame(), None, str(exc)
    if cik is None:
        return pd.DataFrame(), None, "simbolul nu e în lista SEC"
    kind = xbrl_facts.filer_kind(cf)
    if kind != "10-K":
        return pd.DataFrame(), kind, None
    try:
        splits = yf.Ticker(symbol).splits
    except Exception as exc:   # Yahoo: erori variate; fără split-uri nu se calculează valorile pe acțiune
        logger.warning("split-uri %s: %s", symbol, exc)
        return pd.DataFrame(), kind, "split-urile Yahoo indisponibile"
    return xbrl_facts.annual_table(cf, splits=splits), kind, None


def _sec_growth_rows(sgr):
    return [{**{k: v for k, v in r.items() if not k.endswith(" ani")}, **{f"{n} ani": _g(r[f"{n} ani"]) for n in (3, 5, 10)}}
            for r in sgr["rows"]]


def render_growth_card(real_sym, info):
    """Cardul „📈 Creștere și sursa randamentului” (analytics/growth.py)."""
    sym = str(real_sym).upper()
    st.subheader("📈 Creștere și sursa randamentului", help=GROWTH_HELP["card"])
    st.caption("Informativ, nu intră în scor. A: cât a crescut afacerea · B: cât a câștigat acționarul · "
               "C: de unde a venit câștigul.")
    hist, herr = get_growth_history(sym)
    dec_in = None

    # ---------- A. creșterea afacerii
    rows, period_note = [], ""
    if sym.endswith(".RO"):
        files, ferr = list_company_files()
        data = None if ferr else build_bvb_dataset(tuple((f["id"], f["name"], f["modified"]) for f in files))
        fg = growth.file_growth((data or {}).get("companii", {}).get(sym[:-3])) if data else None
        if fg is None:
            st.caption(f"{sym[:-3]} nu are fișier în „{COMPANY_FOLDER_NAME}” cu date anuale: creșterea afacerii N/A.")
        else:
            rows.extend(growth.file_growth_table(fg, _g))
            period_note = (f"Fișierul „Analiză companii”, foaia anuală: ultimul an {fg['year_new']}. "
                           + " ".join(f"{n} ani: {t}." for n, t in fg["notes"].items()))
            dec_in = (fg.get("eps_new"), fg.get("eps_old"), fg.get("date_new"), fg.get("date_old"))
    else:
        fin = get_financial_statements(sym)
        sgr, sec_note, tbl = None, "", None
        ua = sec_secret("sec_user_agent")
        if "." not in sym and "=" not in sym and "^" not in sym and ua and "@" in ua:
            with st.spinner("Citesc rapoartele 10-K de la SEC..."):
                tbl, kind, serr = sec_annual_table(sym, ua)
            if not tbl.empty:
                sgr = growth.sec_growth(tbl, financial=fund.is_financial_issuer(info.get("sector"), sym))
            sec_note = (f"SEC: {serr}. " if serr else "") + ("Emitent străin (20-F): " if kind == "20-F" else "")
        if sgr is not None:
            rows.extend(_sec_growth_rows(sgr))
            diffs = growth.sec_yahoo_diffs(tbl, fin.get("income"), fin.get("cashflow"))
            period_note = (f"Sursa: rapoartele 10-K depuse la SEC (XBRL), exerciții {sgr['first']:%Y}–"
                           f"{sgr['date_new']:%Y} (ultimul încheiat {sgr['date_new']:%d.%m.%Y}); ultimul raport "
                           f"folosit depus {pd.Timestamp(sgr['filed']):%d.%m.%Y}.")
            if diffs:
                period_note += " Diferențe față de Yahoo pe ultimul an (> 2%): " + "; ".join(
                    f"{lbl} SEC {format_ro(s / 1e9, 2)} vs. Yahoo {format_ro(y / 1e9, 2)} mld."
                    + (f" ({_g(d)})" if d is not None else "") for lbl, s, y, d in diffs) + "."
            if sgr["date_old"] is not None:
                dec_in = (sgr["eps_new"], sgr["eps_old"], sgr["date_new"], sgr["date_old"])
        sg = None if sgr is not None else growth.statement_growth(fin.get("income"), fin.get("cashflow"))
        if sgr is not None:
            pass
        elif sg is None:
            st.caption(sec_note + "Situațiile anuale Yahoo nu au două capete utilizabile: creșterea afacerii N/A.")
        else:
            for name, r in sg["rows"].items():
                if name != "EPS diluat":
                    rows.append({"Indicator": name, "Măsură": "total", "3 ani": _g(r["total"]),
                                 "5 ani": "N/A (Yahoo ~4 ani)", "10 ani": "N/A (Yahoo ~4 ani)", "Notă": ""})
                rows.append({"Indicator": name, "Măsură": "pe acțiune", "3 ani": _g(r["per_share"]),
                             "5 ani": "N/A (Yahoo ~4 ani)", "10 ani": "N/A (Yahoo ~4 ani)", "Notă": r["note"]})
            period_note = (sec_note + f"Situațiile anuale Yahoo: {sg['date_old']:%d.%m.%Y} → {sg['date_new']:%d.%m.%Y} "
                           f"({sg['years']} ani fiscali).")
            dec_in = (sg.get("eps_new"), sg.get("eps_old"), sg["date_new"], sg["date_old"])
        y_ref, dg = growth.dividend_growth(hist["Dividends"] if "Dividends" in hist else None, now_ro(),
                                           history_start=hist.index.min() if not hist.empty else None)
        if y_ref is not None and sgr is None:          # cu SEC, dividendul vine din 10-K (rândul de mai sus)
            rows.append({"Indicator": "Dividend pe acțiune (Yahoo)", "Măsură": "pe acțiune",
                         **{f"{n} ani": _g(dg[n]) for n in (3, 5, 10)}, "Notă": f"ani calendaristici până în {y_ref}"})
    if rows:
        st.markdown("**A. Creșterea afacerii (CAGR)**", help=GROWTH_HELP["a"])
        st.dataframe(pd.DataFrame(rows).reindex(columns=["Indicator", "Măsură", "3 ani", "5 ani", "10 ani",
                                                         "Consecvență", "Notă"])
                     .fillna(""), hide_index=True, width='stretch')
        if period_note:
            st.caption(period_note)

    # ---------- B. randament total față de benchmark
    st.markdown("**B. Randament total anualizat (prețul + dividendele reinvestite)**", help=GROWTH_HELP["b"])
    bench_sym, bench_name = growth.tr_benchmark(sym, info.get("currency"))
    bhist, _berr = get_growth_history(bench_sym) if bench_sym else (pd.DataFrame(), None)
    if hist.empty:
        st.caption(f"Istoricul de preț nu a venit ({herr}): randamentul total N/A.")
    else:
        brows = []
        for n in growth.HORIZONS:
            own = growth.total_return(hist["Adj Close"], n)
            ben = growth.total_return(bhist["Adj Close"], n, end=own["end"]) if own and not bhist.empty else None
            diff = (own["cagr"] - ben["cagr"]) * 100 if own and ben and own["cagr"] is not None and ben["cagr"] is not None else None
            brows.append({"Perioada": f"{n} ani", "Acțiunea": _g(own["cagr"]) if own else "N/A (istoric scurt)",
                          "Benchmark": _g(ben["cagr"]) if ben else (
                              (growth.short_history_note(bhist["Adj Close"], n, end=own["end"]) if own and not bhist.empty
                               else None) or "N/A"),
                          "Diferență (pp/an)": format_ro(diff, 1, " pp") if diff is not None else "N/A",
                          "Cotații": format_ro(own["n_obs"], 0) if own else "—"})
        st.dataframe(pd.DataFrame(brows), hide_index=True, width='stretch')
        st.caption(f"Benchmark: {bench_name or 'N/A (piață fără benchmark de randament total)'}. "
                   "„Cotații” = zilele cu preț în fereastră; la BVB istoricul Yahoo are goluri.")

    # ---------- C. sursa randamentului
    st.markdown("**C. De unde a venit randamentul (perioada situațiilor financiare)**", help=GROWTH_HELP["c"])
    d = None
    if dec_in and not hist.empty and dec_in[0] is not None and dec_in[2] is not None:
        d = growth.decompose(hist["Close"], hist["Adj Close"], *dec_in)
    if d is None:
        st.caption("N/A: lipsește EPS la un capăt, EPS ≤ 0, numărul de acțiuni s-a schimbat mult sau lipsește prețul "
                   "de la data situației.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Creștere EPS", _g(d["eps"]) + "/an", help=GROWTH_HELP["c"])
    c2.metric("Schimbarea P/E", _g(d["pe"]) + "/an",
              f"P/E {format_ro(d['pe_old'], 1)} → {format_ro(d['pe_new'], 1)}", delta_color="off", help=GROWTH_HELP["c"])
    c3.metric("Dividende", _g(d["div"]) + "/an", help=GROWTH_HELP["c"])
    c4.metric("Randament total", _g(d["total"]) + "/an", f"suma componentelor {_g(d['eps'] + d['pe'] + d['div'])}",
              delta_color="off", help=GROWTH_HELP["c"])
    st.caption(f"{dec_in[3]:%d.%m.%Y} → {dec_in[2]:%d.%m.%Y} ({d['years']} ani). Verificare: (1 + EPS) × (1 + P/E) × "
               f"(1 + dividend) − 1 = {_g((1 + d['eps']) * (1 + d['pe']) * (1 + d['div']) - 1)} = randamentul total. "
               "Prețul de la datele situațiilor, nu de azi.")


# --- MAIN APP ---
def main():
    # Căutarea (doar în Analiză companie) stă deasupra meniului: locul se rezervă aici, conținutul se scrie în secțiune.
    search_slot = st.sidebar.container()
    st.sidebar.title("Navigare")
    # Valorile rămân „1. Agregator Știri” etc. (toate condițiile `sectiune == ...` le folosesc);
    # doar eticheta afișată are icon Material și nu mai are număr (DESIGN.md §6, decis 10.11).
    _MENU_LABEL = {
        "1. Agregator Știri": ":material/newspaper: Știri",
        "2. Analiză Companie": ":material/query_stats: Analiză companie",
        "11. Fonduri de Pensii": ":material/savings: Fonduri de pensii",
        "12. Fonduri SUA": ":material/account_balance: Fonduri SUA (13F)",
        "3. Portofoliu": ":material/account_balance_wallet: Portofoliu",
        "4. Piață Globală": ":material/public: Piață globală",
        "5. Import Date": ":material/upload_file: Import date",
        "6. Rezumatul Zilei": ":material/today: Rezumatul zilei",
        "7. Scanner Volum": ":material/radar: Scanner volum",
        "8. Watchlist": ":material/visibility: Watchlist",
        "10. Idei și Strategii": ":material/lightbulb: Idei și strategii",
        "9. Glosar Economic": ":material/menu_book: Glosar economic",
    }
    sectiune = st.sidebar.radio("Mergi la:", list(_MENU_LABEL), format_func=_MENU_LABEL.get)
    st.sidebar.markdown("---")

    # Versiunile instalate pe server: necesare ca requirements.txt să fixeze exact ce funcționează
    with st.sidebar.expander("ℹ️ Despre aplicație"):
        import platform
        from importlib import metadata as _md
        rows = [f"Python {platform.python_version()}"]
        for pkg in ("streamlit", "yfinance", "pandas", "numpy", "scipy", "plotly", "curl_cffi", "httpx",
                    "gspread", "google-auth", "google-auth-oauthlib", "feedparser", "pandas-datareader", "scikit-learn",
                    "prophet", "torch", "transformers", "lxml", "requests", "matplotlib"):
            try:
                rows.append(f"{pkg} {_md.version(pkg)}")
            except _md.PackageNotFoundError:
                rows.append(f"{pkg} —")
        st.code("\n".join(rows), language=None)

    # ==================================================
    # 1. AGREGATOR ȘTIRI
    # ==================================================
    if sectiune == "1. Agregator Știri":
        st.title("🌍 Agregator Știri Financiare")
        if st.button("🔄 Actualizează Flux Știri", type="primary"):
            fetch_news_data.clear()
            st.rerun()

        with st.spinner("Se încarcă știrile..."):
            raw_news = fetch_news_data()
        
        categories = list(RSS_CONFIG["Categorii"].keys())
        tabs = st.tabs(categories)

        for i, cat in enumerate(categories):
            with tabs[i]:
                items = filter_news(raw_news, cat)
                if items:
                    for item in items:
                        # Text RSS extern: curățat de tag-uri, tăiat la cuvânt și escapat (data.news.news_card_html).
                        st.markdown(news_card_html(item, future_note(item)), unsafe_allow_html=True)
                else:
                    st.info(f"Nu există știri recente pentru: {cat}.")

    # ==================================================
    # 2. ANALIZĂ COMPANIE (VERSIUNE INTEGRALĂ REPARATĂ)
    # ==================================================
    elif sectiune == "2. Analiză Companie":
        with search_slot:
            st.header("Căutare")
            sym = st.text_input("Simbol (ex: AAPL, TLV):", "AAPL").upper()
            # Indicatorii graficului, restrânși implicit (se deschid și se închid la click)
            with st.expander("📈 Indicatori grafic", expanded=False):
                show_sma20 = st.checkbox("SMA 20", value=True)
                show_sma50 = st.checkbox("SMA 50", value=True)
                show_sma200 = st.checkbox("SMA 200", value=True)
                show_rsi = st.checkbox("RSI 14", value=True)
                show_macd = st.checkbox("MACD", value=True)
            st.markdown("---")

        # Timpul rulării complete (Etapa 5): se scrie la final, ca viteza paginii să se poată măsura.
        _an_t0 = time.perf_counter()
        _an_timer = st.empty()

        with st.spinner(f"Se analizează {sym}..."):
            hist, info, earn_df, real_sym = get_stock_data(sym)
            info = info or {}
            if hist is not None and not hist.empty:
                info = apply_bvb_sheet(info, real_sym, hist)                    # BVB: foaia proprie are prioritate
                info = enrich_info_from_statements(info, real_sym, hist)     # apoi golurile, din situațiile financiare
            
            # --- VERIFICARE DE SIGURANȚĂ (OBLIGATORIE PENTRU CLOUD) ---
            if hist is not None and not hist.empty:
                # Doar dacă avem date, calculăm indicatorii
                hist = calculate_technical_indicators(hist)
                # Stopul se calculează o singură dată, pe tot istoricul; graficul doar îl afișează.
                hist_with_stop = calculate_atr_trailing_stop(hist)
                if hist_with_stop is not None:
                    hist = hist_with_stop
                
                # Extragem prețul SL în siguranță (0 = indisponibil)
                sl_price = 0
                sl_hit_note = ""
                if 'ATR_Stop' in hist.columns and pd.notna(hist['ATR_Stop'].iloc[-1]):
                    sl_price = float(hist['ATR_Stop'].iloc[-1])
                    hit_positions = np.flatnonzero(hist['ATR_Stop_Hit'].to_numpy())
                    if len(hit_positions) and (len(hist) - 1 - hit_positions[-1]) <= 5:
                        sl_hit_note = f" · stop atins pe {hist.index[hit_positions[-1]].strftime('%d.%m')}, resetat"
            # Curba randamentelor (10 ani − 3 luni) a pieței acțiunii: SUA, zona euro sau România.
            # None = indisponibilă; consumatorii (SWOT, Verdict final) nu o punctează.
            curve_market = market_of(real_sym, info.get('currency'))
            spread, spread_label = get_market_curve(curve_market)
            # Intrările „Verdict final” calculate doar în anumite ramuri: inițializare explicită.
            regime_msg, z_val_swot, s_score_val, c_news_ai = None, None, None, []
            rvol, iv_rank, iv_days = None, None, 0
            
        if hist is None or hist.empty:
            st.error("Simbol invalid sau date indisponibile.")
        else:
            # 1. Informații Generaley
            st.markdown(f"## {info.get('longName') or real_sym}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Sector", info.get('sector') or bvb_sector(real_sym) or 'N/A',
                      help="Sectorul după clasificarea Yahoo. La BVB Yahoo nu îl trimite: vine din harta sectoarelor "
                           "ținută în aplicație. Sectorul alege comparabilii din Peer Review.")
            c2.metric("Industrie", info.get('industry') or 'N/A',
                      help="Industria (subdiviziunea sectorului) după Yahoo. La BVB lipsește de obicei (N/A).")
            c3.metric("Capitalizare", format_num(info.get('marketCap')),
                      help="Capitalizarea bursieră = prețul acțiunii × numărul de acțiuni în circulație, în moneda de "
                           "tranzacționare (Yahoo, `marketCap`). Arată mărimea companiei, nu cât valorează ea fundamental.")
            from_statements = [STATEMENT_RATIO_LABELS[k] for k in info.get('_from_statements', [])]
            if not info.get('_fundamentals_available', True):
                if from_statements:
                    st.warning(f"⚠️ Yahoo nu a trimis rezumatul companiei pentru {real_sym}. Am calculat din situațiile financiare: "
                               f"{', '.join(from_statements)}. Rămân N/A și nu intră în scoruri: sectorul, estimările analiștilor "
                               "(Forward P/E, preț țintă), dividendul și acționariatul.")
                else:
                    st.warning(f"⚠️ Yahoo nu a trimis datele fundamentale pentru {real_sym} (P/E, ROE, datorii, sector, analiști, acționariat). "
                               "Indicatorii bazați pe ele apar N/A și nu intră în scoruri. Prețurile, graficele și situațiile financiare anuale sunt disponibile.")
                st.caption(f"Detaliu tehnic: {info.get('_info_error') or 'Yahoo a răspuns, dar fără indicatorii fundamentali.'} "
                           f"· yfinance {yf.__version__} · citit la {now_ro():%H:%M:%S}")
                # Un refuz temporar al Yahoo rămâne altfel în cache 15 minute.
                if st.button("🔄 Reîncearcă citirea datelor de la Yahoo", key="retry_fundamentals"):
                    get_stock_data.clear()
                    st.rerun()
        
        # --- 1. DEFINIREA PREȚULUI (VITAL PENTRU CALCULE) ---
            # Luăm ultimul preț disponibil din istoricul deja descărcat
            curr_price = hist['Close'].iloc[-1] if not hist.empty else 0

            # --- 2. EXTRAGEREA ȚINTEI DIN WATCHLIST ---
            target_p = get_watchlist_target(real_sym)
            
            st.markdown("---") # Separator vizual între info generale și țintă

            # --- 3. AFIȘARE CARD DINAMIC CONSOLIDAT (ȚINTĂ | LIVE | STOP) ---
            if curr_price > 0:
                dist_sl = ((curr_price - sl_price) / curr_price) * 100 if sl_price else 0
                sl_display = f"{ro(sl_price):.2f}" if sl_price else "N/A"
                sl_risk_display = f"Risc: -{ro(dist_sl):.1f}%{sl_hit_note}" if sl_price else "Istoric insuficient pentru ATR"
                
                # Culori pentru statusul țintei
                target_text, t_status, t_color = entry_target_view(curr_price, target_p)

                st.markdown(f"""
                    <div style="background:#161B22; padding:20px; border-radius:15px; border:1px solid #30363D; margin-bottom:20px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                            <div style="flex: 1; min-width: 100px;">
                                <p style="color:#8B949E; margin:0; font-size:12px; text-transform:uppercase; letter-spacing:1px;">Țintă Intrare</p>
                                <h2 style="color:var(--tk-text); margin:5px 0; font-size:22px;">{target_text} <span style="font-size:12px; color:#8B949E;">{info.get('currency', 'USD')}</span></h2>
                            </div>
                            <div style="flex: 1; min-width: 100px; text-align: center; border-left: 1px solid #30363D; border-right: 1px solid #30363D;">
                                <p style="color:#8B949E; margin:0; font-size:12px; text-transform:uppercase; letter-spacing:1px;">Preț Live</p>
                                <h2 style="color:{t_color}; margin:5px 0; font-size:28px;">{ro(curr_price):.2f}</h2>
                            </div>
                            <div style="flex: 1; min-width: 100px; text-align: right;">
                                <p style="color:#F85149; margin:0; font-size:12px; text-transform:uppercase; letter-spacing:1px;">Stop-Loss (Exit)</p>
                                <h2 style="color:#F85149; margin:5px 0; font-size:22px;">{sl_display}</h2>
                                <small style="color:#8B949E;">{sl_risk_display}</small>
                            </div>
                        </div>
                        <div style="background:{t_color}22; color:{t_color}; padding:6px; border-radius:8px; font-weight:bold; font-size:14px; text-align:center; margin-top:15px; border: 1px solid {t_color}44;">
                            {t_status}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown("---") # Separator vizual între info generale și țintă    
                        
            # 2. Grafic Tehnic (Păstrat exact cum era în original)
            hist = calculate_technical_indicators(hist)
            st.subheader("📉 Grafic Tehnic")
            col_sel, col_price_info = st.columns([1, 4])
            with col_sel:
                time_opt = st.selectbox("Interval", ["1 Lună", "3 Luni", "6 Luni", "1 An", "3 Ani", "5 Ani"], index=3)
            
            # Fereastră calendaristică: "1 An" înseamnă un an de date, nu 365 de rânduri (~1,45 ani)
            subset = slice_window(hist, time_opt)
            
            # Lipsă (interval gol, o singură ședință) → N/A, nu 0 (analytics.technical.price_changes)
            pc = price_changes(subset['Close'] if not subset.empty else None, hist['Close'])
            curr_price = pc['last']
            _chg = lambda v, p: None if v is None or p is None else f"{ro(v):.2f} ({ro(p):.2f}%)"

            with col_price_info:
                 m1, m2 = st.columns(2)
                 m1.metric(f"Interval ({time_opt})", "N/A" if curr_price is None else f"{ro(curr_price):.2f} {info.get('currency', '')}",
                           _chg(pc['diff'], pc['diff_pct']),
                           help="Ultima închidere și variația față de prima închidere din intervalul ales (fereastră "
                                "calendaristică). Prețuri ajustate pentru dividende și splituri, ca în grafic.")
                 m2.metric("Evoluție Azi", "N/A" if curr_price is None else f"{ro(curr_price):.2f}",
                           _chg(pc['day'], pc['day_pct']),
                           help="Ultima închidere față de închiderea ședinței precedente. În timpul ședinței, „ultima "
                                "închidere” e prețul de acum (ședință în curs).")

            rows_needed = 1 + (1 if show_rsi else 0) + (1 if show_macd else 0)
            row_heights = [0.6] + ([0.2] if show_rsi else []) + ([0.2] if show_macd else [])
            total = sum(row_heights)
            row_heights = [r/total for r in row_heights]

            fig = make_subplots(rows=rows_needed, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights)
            fig.add_trace(go.Candlestick(x=subset.index, open=subset['Open'], high=subset['High'], low=subset['Low'], close=subset['Close'], name='Preț', hovertext=subset['Volume'].apply(lambda x: f"Volum: {format_num(x)}")), row=1, col=1)
            # --- ATR STOP ---
            # `subset` este o felie din `hist`, care are deja coloana ATR_Stop calculată
            # pe tot istoricul. Nu se recalculează pe fereastră: altfel valoarea din
            # card ar diferi de cea de pe grafic și s-ar schimba odată cu intervalul ales.

            # Adăugăm linia de Stop-Loss pe grafic (Etajul 1)
            if 'ATR_Stop' in subset.columns:
                fig.add_trace(go.Scatter(
                    x=subset.index, 
                    y=subset['ATR_Stop'],
                    line=dict(color='#F85149', width=2, dash='dot'),
                    name='ATR Trailing Stop (Exit)',
                    hovertemplate="Stop-Loss: %{y:.2f}"
                ), row=1, col=1)

            if show_sma20: fig.add_trace(go.Scatter(x=subset.index, y=subset['SMA20'], line=dict(color='orange', width=1), name='SMA 20'), row=1, col=1)
            if show_sma50: fig.add_trace(go.Scatter(x=subset.index, y=subset['SMA50'], line=dict(color='cyan', width=1), name='SMA 50'), row=1, col=1)
            if show_sma200: fig.add_trace(go.Scatter(x=subset.index, y=subset['SMA200'], line=dict(color='purple', width=1.5), name='SMA 200'), row=1, col=1)

            current_row = 2
            if show_rsi:
                fig.add_trace(go.Scatter(x=subset.index, y=subset['RSI'], line=dict(color='yellow'), name='RSI 14'), row=current_row, col=1)
                fig.add_hline(y=70, line_dash="dot", row=current_row, col=1, line_color="red")
                fig.add_hline(y=30, line_dash="dot", row=current_row, col=1, line_color="green")
                current_row += 1

            if show_macd:
                fig.add_trace(go.Scatter(x=subset.index, y=subset['MACD'], line=dict(color='#00E5FF'), name='MACD'), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=subset.index, y=subset['Signal'], line=dict(color='#FFAB00'), name='Signal'), row=current_row, col=1)
                macd_hist = subset['MACD'] - subset['Signal']
                fig.add_trace(go.Bar(x=subset.index, y=macd_hist, name='Hist',   # verde peste 0, roșu sub 0, după tema activă
                                     marker_color=ui_theme.sign_colors(macd_hist, DF_UP, DF_DOWN, DF_MUTED)), row=current_row, col=1)

            fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False, hovermode="x unified", paper_bgcolor='#0E1117', plot_bgcolor='#0E1117')
            plot_ro(fig, width='stretch')

            # --- SEPARATORUL SOLICITAT (ADAUGĂ ACEASTĂ LINIE) ---
            st.markdown("---")

            # Beta și alpha (un singur calcul, folosit de scorul fundamental, DCF, audit și scoruri)
            risk_stats = resolve_beta_alpha(real_sym, info, hist)
            beta_val, alpha_val = risk_stats["beta"], risk_stats["alpha"]

            # 2b. Decizie BVB în doi pași (doar .RO cu fișier în „Analiză companii”)
            bvb_decision = render_bvb_decision(real_sym, hist)
            # 2b'. Deținerile fondurilor de pensii Pilon II (doar .RO; informativ, netestat)
            render_pilon2_card(real_sym, info)
            # 2b''. Fondurile SUA (13F) care dețin acțiunea (doar SUA; context, netestat)
            render_sec_company_card(real_sym)
            # 2b-bis. Insiderii (Form 4, 90 de zile; doar SUA; context, netestat)
            render_sec_insiders_card(real_sym)
            # 2b-ter. Deținătorii peste 5% (13D/13G; doar SUA; context, netestat)
            render_sec_holders_card(real_sym)
            # 2c. Scor fundamental pentru acțiunile din afara BVB (descriptiv: testul pe SUA nu a găsit putere de predicție)
            global_decision = render_global_decision(real_sym, info, hist, risk_stats)
            # „Verdict final” refolosește exact acest rezultat (nu recalculează)
            score_decision = bvb_decision or global_decision

            # 2d. Creștere (CAGR) și sursa randamentului (informativ, nu intră în scor)
            render_growth_card(real_sym, info)

            # 3. Indicatori Fundamentali (Cele 4 coloane originale)
            st.subheader("📊 Indicatori Fundamentali")
            if from_statements:
                st.caption("Calculat de aplicație din situațiile financiare (Yahoo nu a trimis valoarea): "
                           + ", ".join(from_statements) + ". Rentabilitățile folosesc soldul de la sfârșitul perioadei.")
            if info.get('_from_bvb_sheet'):
                bvb_note = (f"Din foaia BVB (raportare {info.get('_bvb_period') or 'N/A'}), cu prioritate față de Yahoo: "
                            + ", ".join(STATEMENT_RATIO_LABELS[k] for k in info['_from_bvb_sheet']) + ". ")
                repriced = info.get('_bvb_repriced', [])
                if 'trailingPE' in repriced and 'priceToBook' in repriced:
                    bvb_note += "P/E și P/BV sunt recalculate la prețul curent, din EPS-ul și valoarea contabilă din foaie."
                elif 'trailingPE' in repriced:
                    bvb_note += "P/E e recalculat la prețul curent din EPS-ul din foaie; P/BV rămâne cel din foaie, la prețul actualizării ei."
                else:
                    bvb_note += "P/E și P/BV nu au putut fi recalculate la prețul curent (EPS lipsă sau negativ): sunt cele din foaie."
                st.caption(bvb_note)
                bvb_unmapped = load_bvb_fundamentals().get("_unmapped") or []
                if bvb_unmapped:
                    st.warning("⚠️ Foaia BVB nu mai are rândurile așteptate pentru: "
                               + ", ".join(STATEMENT_RATIO_LABELS.get(k, k) for k in bvb_unmapped)
                               + ". Probabil rândul a fost redenumit; acești indicatori vin acum din Yahoo sau din situațiile financiare.")
            if info.get('_bvb_indicators'):
                with st.expander(f"📄 Toți indicatorii din foaia BVB pentru {bvb_symbol(real_sym)}, față de piață"):
                    st.dataframe(pd.DataFrame(
                        [(row[0].strip(), row[1] if row[2] is not None else "N/A", row[3], row[4]) for row in info['_bvb_indicators']],
                        columns=["Indicator", bvb_symbol(real_sym), "Media BVB (din foaie)", "Mediana BVB (calculată)"]),
                        hide_index=True, width='stretch')
                    st.caption("Valorile companiei sunt cele din foaie, la prețul actualizării ei. Media e coloana A a foii; mediana e calculată "
                               "din aceleași companii și nu e trasă de extreme (un singur P/E foarte mare ridică media, nu și mediana).")
            de_ratio = info.get('debtToEquity')
            de_display = f"{ro(de_ratio):.2f}%" if de_ratio is not None else "N/A"

            with st.container():
                c_eval, c_prof, c_indat, c_risc = st.columns(4)
                with c_eval:
                    st.markdown("**Evaluare & Dividende**")
                    st.metric("P/E Ratio", format_num(info.get('trailingPE')), help=fund_help(info, "trailingPE"))
                    st.metric("Forward P/E", format_num(info.get('forwardPE')), help=fund_help(info, "forwardPE"))
                    # PEG (Lynch): istoric = CAGR EPS 3 ani din situațiile raportate (la BVB din fișierul „Analiză
                    # companii”); analiști = Yahoo trailingPegRatio. Nu intră în scor (scorul e testat fără el).
                    if (bvb_decision or {}).get("is_bvb"):
                        peg_g = bvb_decision.get("eps_growth")
                        peg_src = ("fișierul „Analiză companii”, EPS = profit net / acțiuni emise"
                                   + (f", {peg_g['year_old']} → {peg_g['year_new']}" if peg_g else ""))
                    else:
                        peg_g = fund.eps_growth_statements(get_financial_statements(real_sym).get("income"))
                        peg_src = ("situațiile financiare anuale Yahoo (EPS diluat)"
                                   + (f", {peg_g['date_old']:%Y} → {peg_g['date_new']:%Y}" if peg_g else ""))
                    peg_growth = (peg_g or {}).get("growth")
                    peg_hist = fund.peg_ratio(num(info, 'trailingPE'), peg_growth)
                    peg_note = ("" if peg_g is None else
                                f" EPS {ro(peg_g['eps_old']):.4f} → {ro(peg_g['eps_new']):.4f}"
                                + ("" if peg_growth is None else f", creștere {ro(peg_growth * 100):+.1f}%/an") + ".")
                    if (peg_g or {}).get("shares_changed"):
                        peg_note += " Numărul de acțiuni s-a schimbat mult între cei doi ani: EPS-urile nu sunt comparabile (N/A)."
                    st.metric("PEG (istoric, 3 ani)", f"{ro(peg_hist):.2f}" if peg_hist is not None else "N/A",
                              help=f"{FUND_HELP['peg_hist']}\n\n**Sursă:** {peg_src}.{peg_note}")
                    peg_an = num(info, 'trailingPegRatio')
                    st.metric("PEG (estimări analiști)", f"{ro(peg_an):.2f}" if (peg_an is not None and peg_an > 0) else "N/A",
                              help=f"{FUND_HELP['peg_analysts']}\n\n**Sursă:** Yahoo Finance.")
                    div_rate = info.get('dividendRate')
                    div_display = f"{ro(div_rate):.2f} ({ro((div_rate/curr_price*100)):.2f}%)" if (div_rate and curr_price) else "N/A"
                    div_help = fund_help(info, "dividendRate")
                    bvb_dy = (bvb_decision or {}).get("dy")
                    if bvb_dy is not None and (bvb_decision or {}).get("price"):
                        # BVB: din fișierul „Analiză companii” (Yahoo nu trimite dividendul); are prioritate.
                        dps = bvb_dy * bvb_decision["price"]
                        div_display = f"{ro(dps):.4f} ({ro(bvb_dy * 100):.2f}%)" if dps < 1 else f"{ro(dps):.2f} ({ro(bvb_dy * 100):.2f}%)"
                        div_help = (f"{FUND_HELP['dividendRate']}\n\n**Sursă:** fișierul „Analiză companii” {bvb_symbol(real_sym)}: "
                                    f"dividendul total din {bvb_decision.get('dy_year') or 'ultimul an'} / capitalizarea la prețul curent "
                                    f"({ro(bvb_decision['price']):.4f} lei); pe acțiune = randament × preț.")
                    st.metric("Dividend (Randament)", div_display, help=div_help)
                    st.metric("P/BV", format_num(info.get('priceToBook')), help=fund_help(info, "priceToBook"))
                    gn_pe, gn_pb = num(info, 'trailingPE'), num(info, 'priceToBook')
                    gn_calc = gn_pe * gn_pb if (gn_pe is not None and gn_pb is not None) else None   # lipsă = N/A, nu 0
                    st.metric("GN (Graham)", f"{ro(gn_calc):.2f}" if (gn_calc is not None and gn_calc > 0) else "N/A", help=fund_help(info, "gn"))
                    st.metric("EPS", format_num(info.get('trailingEps'), small_decimals=4), help=fund_help(info, "trailingEps"))
                    st.metric("Val. Contabilă/Acțiune", format_num(info.get('bookValue'), small_decimals=4), help=fund_help(info, "bookValue"))
                with c_prof:
                    st.markdown("**Profitabilitate**")
                    st.metric("ROA", format_num(info.get('returnOnAssets'), True), help=fund_help(info, "returnOnAssets"))
                    st.metric("ROE", format_num(info.get('returnOnEquity'), True), help=fund_help(info, "returnOnEquity"))
                    st.metric("Marjă Netă", format_num(info.get('profitMargins'), True), help=fund_help(info, "profitMargins"))
                    st.metric("Marjă Operațională", format_num(info.get('operatingMargins'), True), help=fund_help(info, "operatingMargins"))
                with c_indat:
                    st.markdown("**Îndatorare**")
                    st.metric("Datorii/Capital", de_display, help=fund_help(info, "debtToEquity"))
                    st.metric("Current Ratio", format_num(info.get('currentRatio')), help=fund_help(info, "currentRatio"))
                    st.metric("Quick Ratio", format_num(info.get('quickRatio')), help=fund_help(info, "quickRatio"))
                with c_risc:
                    st.markdown("**Risc (Alpha & Beta)**")
                    st.metric("Beta", format_num(beta_val), help=f"{RISK_HELP['beta']}\n\n**Sursă:** {risk_stats['beta_label']}")
                    st.metric("Alpha (1Y)", format_num(alpha_val, True), help=f"{RISK_HELP['alpha']}\n\n**Sursă:** {risk_stats['alpha_label']}")
            
            # ==================================================
            # MODUL NOU: DATE FINANCIARE VIZUALE (STIL XTB)
            # ==================================================
            st.markdown("---")
            st.subheader("📊 Evoluție Financiară (Venit vs. Profit Net)")
            
            with st.spinner("Extragem datele contabile..."):
                try:
                    # Aceleași situații Yahoo (financials, quarterly_financials) ca scorurile, din cache-ul de 1 oră
                    # al get_financial_statements; înainte se descărcau din nou la fiecare clic din pagină.
                    _fs = get_financial_statements(real_sym)
                    
                    # Preluăm datele (yfinance le aduce cu datele pe coloane, deci facem Transpose .T)
                    df_ann = _fs["income"].T if _fs.get("income") is not None else pd.DataFrame()
                    df_qtr = _fs["q_income"].T if _fs.get("q_income") is not None else pd.DataFrame()
                    
                    # Curățăm și ordonăm cronologic (cele mai vechi primele)
                    if not df_ann.empty and not df_qtr.empty:
                        df_ann = df_ann.sort_index(ascending=True)
                        df_qtr = df_qtr.sort_index(ascending=True)
                        
                        # Creăm Tab-uri exact ca în XTB
                        tab_anual, tab_trimestrial = st.tabs([":material/calendar_month: Anual", ":material/date_range: Trimestrial"])
                        
                        def draw_financial_chart(df, title, is_quarterly=False):
                            from plotly.subplots import make_subplots # Asigură-te că e importat
                            
                            # Identificăm coloanele corecte
                            rev_col = 'Total Revenue' if 'Total Revenue' in df.columns else 'Operating Revenue'
                            net_col = 'Net Income' if 'Net Income' in df.columns else 'Net Income Common Stockholders'
                            
                            if rev_col not in df.columns or net_col not in df.columns:
                                return st.info("Date financiare incomplete pentru grafic.")
                                
                            # Păstrăm ultimele 5 perioade și folosim .copy() pentru a evita erorile de manipulare pandas
                            df_plot = df.tail(5).copy()
                            
                            # CALCULĂM MARJA NETĂ (%) LA CALD
                            df_plot['Net Margin'] = (df_plot[net_col] / df_plot[rev_col]) * 100
                            
                            # Formatăm axa X
                            if is_quarterly:
                                x_labels = [f"T{d.quarter} {d.year}" for d in df_plot.index]
                            else:
                                x_labels = [str(d.year) for d in df_plot.index]

                            # CREĂM UN GRAFIC CU AXĂ SECUNDARĂ (Y2)
                            fig = make_subplots(specs=[[{"secondary_y": True}]])
                            
                            # Bara 1: Venit (Axa Principală Stânga)
                            fig.add_trace(go.Bar(
                                x=x_labels, y=df_plot[rev_col],
                                name='Venit', marker_color='#58A6FF', opacity=0.8,
                                text=df_plot[rev_col].apply(lambda x: format_num(x)), textposition='auto'
                            ), secondary_y=False)
                            
                            # Bara 2: Profit Net (Axa Principală Stânga)
                            colors_net = ['#3FB950' if val > 0 else '#F85149' for val in df_plot[net_col]]
                            fig.add_trace(go.Bar(
                                x=x_labels, y=df_plot[net_col],
                                name='Venit Net', marker_color=colors_net, opacity=0.9,
                                text=df_plot[net_col].apply(lambda x: format_num(x)), textposition='auto'
                            ), secondary_y=False)

                            # Linia: Marja Netă (Axa Secundară Dreapta)
                            fig.add_trace(go.Scatter(
                                x=x_labels, y=df_plot['Net Margin'],
                                name='Marja Netă (%)', mode='lines+markers+text',
                                line=dict(color='#FFAB00', width=3),
                                marker=dict(size=10, symbol='diamond', color='#FFAB00', line=dict(width=1, color='white')),
                                text=df_plot['Net Margin'].apply(lambda x: f"{ro(x):.1f}%"),
                                textposition='top center',
                                textfont=dict(color='#FFAB00', weight='bold'),
                                hovertemplate="%{y:.1f}%"
                            ), secondary_y=True)

                            # Stilizare layout pentru un aspect curat și "aerisit"
                            fig.update_layout(
                                barmode='group',
                                height=450, # Am mărit un pic înălțimea pentru a acomoda textul de deasupra liniei
                                template="plotly_dark",
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
                                margin=dict(l=0, r=0, t=50, b=0),
                                hovermode="x unified"
                            )
                            
                            # Setări specifice pentru cele două axe Y
                            fig.update_yaxes(title_text="", showgrid=True, gridcolor='#30363D', secondary_y=False)
                            fig.update_yaxes(title_text="Marjă (%)", showgrid=False, ticksuffix="%", tickformat=".1f", color="#FFAB00", secondary_y=True)
                            
                            plot_ro(fig, width='stretch')

                        with tab_anual:
                            draw_financial_chart(df_ann, "Evoluție Anuală")
                            
                        with tab_trimestrial:
                            draw_financial_chart(df_qtr, "Evoluție Trimestrială", is_quarterly=True)
                            
                    else:
                        st.info("Bilanțurile detaliate nu sunt disponibile pentru acest simbol.")
                except Exception as e:
                    st.warning(f"A apărut o problemă la generarea graficului contabil: {e}")
            # ==================================================
            # MODUL PRO: PEER ANALYSIS (CARDURI SUS + TABEL JOS)
            # ==================================================
            st.markdown("---")
            st.subheader("🏁 Peer Review: Poziționarea față de Liderii de Sector")
            
            # Compania față de mediana comparabililor din același sector și aceeași regiune.
            p_region = peer_region(real_sym)
            p_is_bvb = p_region == "BVB"
            # La BVB sectorul vine din harta proprie (Yahoo nu îl trimite), iar comparabilii sunt
            # companiile BVB din același sector (din foaie) plus, unde există, cei regionali (Yahoo).
            p_sector = bvb_sector(real_sym) if p_is_bvb else info.get('sector')
            own_row = {"Simbol": real_sym, "Capitalizare": num(info, 'marketCap'), "Monedă": info.get('currency') or ""}
            for p_key, p_label, p_mult in PEER_METRICS:
                p_val = num(info, p_key)
                own_row[p_label] = None if p_val is None else p_val * p_mult

            peer_rows, peer_failed, p_excluded, p_group = [], [], [], None
            p_has_list = bool(bvb_sector_peers(real_sym) or bvb_regional_peers(real_sym)) if p_is_bvb else bool(peer_list(real_sym, p_sector))
            if p_is_bvb:
                own_row["Sursă"] = ""
                if own_row["Capitalizare"] is None:
                    own_row["Capitalizare"] = bvb_file_market_cap(real_sym, positive_or_none(hist["Close"].iloc[-1]) if hist is not None and not hist.empty else None)
                bvb_sheet_data = load_bvb_fundamentals()
                bvb_not_in_sheet = []
                bvb_peer_prices = get_bvb_peer_prices(tuple(bvb_sector_peers(real_sym))) if bvb_sector_peers(real_sym) else {}
                for b_sym in bvb_sector_peers(real_sym):
                    b_entry = bvb_sheet_data.get(b_sym)
                    b_info = dict(b_entry.get("info") or {}) if isinstance(b_entry, dict) else {}
                    # P/E și P/BV la prețul curent, ca la compania analizată; fără preț rămân cele din foaie.
                    b_repriced = reprice_bvb(b_info, bvb_peer_prices.get(b_sym))
                    b_info.update(b_repriced)
                    b_row = sheet_peer_row(b_sym, b_info)
                    if b_row is None:
                        bvb_not_in_sheet.append(f"{b_sym}.RO")
                    else:
                        b_row["Capitalizare"] = bvb_file_market_cap(b_sym, bvb_peer_prices.get(b_sym))
                        b_source = "foaie, preț curent" if "trailingPE" in b_repriced else "foaie, preț vechi"
                        peer_rows.append(dict(b_row, **{"Sursă": b_source}))
                if bvb_not_in_sheet:
                    # Fără indicatori în foaie (ex. BRD, Patria Bank): se citesc din Yahoo, cu sursa afișată.
                    y_rows, y_failed = get_bvb_yahoo_peer_rows(tuple(bvb_not_in_sheet))
                    peer_rows += [dict(r, **{"Sursă": "Yahoo (nu e în foaie)"}) for r in y_rows]
                    peer_failed += [f"{sym} (nici în foaia BVB, nici în Yahoo)" for sym in y_failed]
                if bvb_regional_peers(real_sym):
                    with st.spinner("Se citesc comparabilii regionali (o singură dată la 6 ore pe sector)..."):
                        reg_rows, reg_failed = get_peer_rows(BVB_REGION, p_sector)
                    peer_rows += [dict(r, **{"Sursă": "Yahoo"}) for r in reg_rows]
                    peer_failed += reg_failed
            elif peer_list(real_sym, p_sector):
                with st.spinner("Se citesc comparabilii (o singură dată la 6 ore pe sector)..."):
                    all_rows, peer_failed = get_peer_rows(p_region, p_sector)
                peer_rows = [r for r in all_rows if r["Simbol"].upper() != real_sym.upper()]
                # Sector financiar: bănci cu bănci, restul între ei (după industria din Yahoo).
                peer_rows, p_excluded, p_group = split_financial_peers(peer_rows, p_sector, info.get('industry'))
            medians = peer_medians(peer_rows)

            # --- PASUL 1: compania față de mediană (fără verdict: eșantionul e mic) ---
            p_cards = st.columns(4)
            for p_col, p_label, p_fmt in zip(p_cards, ("P/E", "ROE (%)", "ROA (%)", "Marjă netă (%)"),
                                             ("{:.1f}", "{:.1f}%", "{:.1f}%", "{:.1f}%")):
                p_own = own_row[p_label]
                p_med, p_n = medians[p_label]
                p_diff = versus_median(p_own, p_med)
                p_title = p_label.replace(" (%)", "")
                p_help = (f"{p_title} al companiei față de mediana comparabililor din același sector și aceeași regiune. "
                          "Diferența = (valoarea − mediana) ÷ |mediana|; n = câți comparabili au valoarea. Mediana apare de la "
                          "3 comparabili; la P/E se iau doar valorile pozitive. Fără verdict: eșantionul e mic. Culoarea arată doar sensul: "
                          "verde = P/E sub mediană, respectiv ROE/ROA/marjă peste mediană; P/E negativ rămâne gri. Sursa: foaia BVB "
                          "la emitenții BVB, altfel Yahoo, completat din situațiile financiare.")
                if p_own is None:
                    p_col.metric(p_title, "N/A",
                                  help=p_help)
                elif p_diff is None:
                    p_col.metric(p_title, p_fmt.format(ro(p_own)), "fără mediană de comparație", delta_color="off",
                                  help=p_help)
                else:
                    p_col.metric(p_title, p_fmt.format(ro(p_own)),
                                 f"{ro(p_diff * 100):+.0f}% față de mediana {p_fmt.format(ro(p_med))} (n={p_n})",
                                 delta_color=peer_delta_color(p_label, p_own),
                                  help=p_help)

            st.write("")

            # --- PASUL 2: tabelul ---
            if p_is_bvb and not p_has_list:
                st.info(("Simbolul nu este în harta de sectoare BVB (analytics/peers.py). " if not p_sector else
                         f"În sectorul „{p_sector}” nu există altă companie în foaia BVB și nici listă regională. ")
                        + "Reperul rămâne media și mediana pieței: vezi expanderul „Toți indicatorii din foaia BVB” "
                          "de sub Indicatori Fundamentali.")
            elif not p_has_list:
                st.info("Nu există o listă de comparabili pentru acest simbol: "
                        + ("Yahoo nu a trimis sectorul." if not p_sector else
                           f"sectorul „{p_sector}” sau piața simbolului nu are listă definită."))
            elif not peer_rows:
                st.warning("Nu există date pentru niciun comparabil"
                           + (": " + ", ".join(peer_failed) + "." if p_is_bvb and peer_failed else
                              " (probabil o limitare temporară a Yahoo)."))
                if st.button("🔄 Reîncearcă citirea comparabililor", key="retry_peers"):
                    get_peer_rows.clear()
                    st.rerun()
            else:
                p_where = ("BVB (din foaie) și Europa Centrală și de Est (Yahoo)" if p_is_bvb and bvb_regional_peers(real_sym)
                           else "BVB (din foaie)" if p_is_bvb else "SUA" if p_region == 'US' else "Europa")
                st.markdown(f"**🔍 Comparabili: sectorul „{p_sector}”, {p_where}**")
                median_row = {"Simbol": "Mediana comparabililor", "Capitalizare": None, "Monedă": ""}
                if p_is_bvb:
                    median_row["Sursă"] = ""
                median_row.update({label: medians[label][0] for _, label, _ in PEER_METRICS})
                df_peers = pd.DataFrame([own_row, median_row] + sorted(
                    peer_rows, key=lambda r: -(r["Capitalizare"] or 0)))
                # Mediana nu are capitalizare (celulă goală); un comparabil fără capitalizare arată N/A.
                df_peers["Capitalizare"] = [
                    "" if sym == "Mediana comparabililor" else "N/A" if pd.isna(cap) else f"{format_num(cap)} {cur}".strip()
                    for sym, cap, cur in zip(df_peers["Simbol"], df_peers["Capitalizare"], df_peers["Monedă"])]
                df_peers = df_peers.drop(columns=["Monedă", "Industrie"], errors="ignore")
                for _, p_label, _ in PEER_METRICS:
                    if medians[p_label][1] == 0:
                        df_peers = df_peers.drop(columns=[p_label])     # niciun comparabil nu are indicatorul
                    else:
                        df_peers[p_label] = [format_peer_value(p_label, v) for v in df_peers[p_label]]
                st.dataframe(df_peers.style.apply(
                    lambda row: [f"font-weight: bold; background-color: {ui_theme.DF_HIGHLIGHT_BG}" if row.name < 2 else "" for _ in row], axis=1),
                    width='stretch', hide_index=True)
                p_counts = ", ".join(f"{label.replace(' (%)', '')} n={medians[label][1]}" for _, label, _ in PEER_METRICS)
                p_note = (f"Mediana e calculată din {len(peer_rows)} comparabili, fără {real_sym}; observații pe indicator: {p_counts}. "
                          "La P/E și P/BV valorile negative sunt excluse. Eșantionul e mic și ales manual (companii mari): "
                          "arată unde se situează compania, nu dacă e scumpă sau ieftină.")
                if p_region == "EU" or p_is_bvb:
                    p_note += (" Capitalizările sunt în moneda fiecărei burse; rapoartele nu depind de monedă. La companiile BVB din foaie, "
                               "capitalizarea = prețul curent × acțiunile din ultimul raport din fișierele „Analiză companii” "
                               "(N/A dacă emitentul nu are fișier sau prețul lipsește).")
                if p_is_bvb:
                    p_note += (" ROE, ROA și marja netă se compară direct între țări. P/E și P/BV nu: rata fără risc diferă între monede "
                               "(în RON e mult peste cea în EUR), deci un multiplu mai mic la București nu înseamnă, singur, subevaluare. "
                               "Companiile BVB au cifrele din foaie, cu P/E și P/BV recalculate la prețul curent când acesta e disponibil; "
                               "cele care nu sunt în foaie vin din Yahoo (vezi coloana Sursă). „Datorii/Capital” nu există în foaie. Mediana întregii piețe BVB e în expanderul de sub Indicatori Fundamentali.")
                if p_group:
                    p_note += (f" Grup: {p_group}, după industria din Yahoo; excluse din mediană: "
                               + (", ".join(p_excluded) if p_excluded else "niciunul") + ".")
                if peer_failed:
                    p_note += (" Fără date: " if p_is_bvb else " Fără date de la Yahoo: ") + ", ".join(peer_failed) + "."
                st.caption(p_note)
            st.markdown("---")
            
            # 4. Financiar & Raportări
            st.subheader("💰 Financiar & Raportări")
            st.markdown("""<div class="fin-card"><h4>Rezultate Financiare (Ultima Raportare)</h4></div>""", unsafe_allow_html=True)
            rev = info.get('totalRevenue'); net_inc = info.get('netIncomeToCommon'); cash = info.get('totalCash')
            exp = (rev - net_inc) if (rev and net_inc) else None
            cf1, cf2, cf3, cf4 = st.columns(4)
            cf1.metric("Venituri Totale", format_num(rev),
                       help="Cifra de afaceri pe ultimele 12 luni (TTM, Yahoo `totalRevenue`); fără ea, din situațiile "
                            "financiare: TTM din 4 trimestre sau ultimul an fiscal. În moneda de raportare.")
            cf2.metric("Profit Net", format_num(net_inc),
                       help="Profitul net atribuibil acționarilor ordinari, ultimele 12 luni (Yahoo `netIncomeToCommon`; "
                            "fără el, din situațiile financiare). În moneda de raportare.")
            cf3.metric("Cheltuieli (Est.)", format_num(exp),
                       help="Proxy: venituri − profit net = toate costurile, inclusiv dobânzi, impozit și elemente "
                            "excepționale. Nu e un rând din situațiile financiare.")
            cf4.metric("Numerar Disponibil", format_num(cash),
                       help="Numerar și echivalente (plus plasamente pe termen scurt, unde Yahoo le include) din cel mai "
                            "recent bilanț (Yahoo `totalCash`; fără el, din bilanț). Nu scade datoriile.")
            
            st.markdown("<br>", unsafe_allow_html=True)
            col_an_left, col_an_right = st.columns([1, 2])
            with col_an_left:
                st.markdown("""<div class="fin-card"><h4>Analiști</h4></div>""", unsafe_allow_html=True)
                rec = (info.get('recommendationKey') or 'N/A').replace('_', ' ').upper()
                rec_mean = info.get('recommendationMean')
                target = info.get('targetMeanPrice')
                color_rec = "#3FB950" if "BUY" in rec else "#F85149" if "SELL" in rec else "#8B949E"
                st.markdown(f"Recomandare: <span style='color:{color_rec}; font-weight:bold;'>{rec}</span>", unsafe_allow_html=True)
                if rec_mean:
                    pos_p = (max(1.0, min(5.0, rec_mean)) - 1.0) / 4.0 * 100.0
                    st.markdown(f"""<div class="analyst-bar-container"><div class="analyst-bar-gradient"></div><div class="analyst-marker" style="left:{pos_p}%;"></div></div>""", unsafe_allow_html=True)
                st.metric("Preț Țintă (Mediu)", f"{ro(target):,.2f} {info.get('currency','USD')}" if target else "N/A",
                          help=RISK_HELP["target"]
                               + (f"\n\n**Față de prețul curent:** {ro((target / curr_price - 1) * 100):+.1f}%."
                                  if (target and curr_price) else ""))

            with col_an_right:
                st.markdown("""<div class="fin-card"><h4>🆚 Raportări vs Așteptări</h4></div>""", unsafe_allow_html=True)
                e_disp, e_has_est = earnings_table(earn_df)
                if not e_disp.empty:
                    def style_surprise(val):
                        if val is None or pd.isna(val):
                            return f'color: {DF_MUTED}'
                        color = DF_UP if val > 0 else DF_DOWN if val < 0 else DF_MUTED
                        return f'color: {color}; font-weight: bold'
                    if not e_has_est:
                        st.caption("Fără estimări ale analiștilor la Yahoo pentru acest simbol: se afișează doar EPS-ul raportat; Estimare, Diferență și Surpriză sunt N/A.")
                    st.dataframe(e_disp.style.map(style_surprise, subset=['Surpriză %']).format({'Estimare': ro_spec(',.2f'), 'Realizat': ro_spec(',.2f'), 'Diferență': ro_spec(',.2f'), 'Surpriză %': ro_spec('.2%')}), width='stretch')
                else: st.info("Date earnings indisponibile.")
            
            # --- MODUL NOU: ANALIZA VOLUMULUI INSTITUȚIONAL ---
            st.markdown("---")
            st.subheader("📊 Analiza Fluxului de Volum (Instituțional)")

            if not hist.empty:
                # Calculăm Volumul Relativ (RVOL)
                current_vol = hist['Volume'].iloc[-1]
                avg_vol = hist['Volume'].rolling(window=20).mean().iloc[-1]
                rvol = current_vol / avg_vol

                v_col1, v_col2 = st.columns([1, 2])

                with v_col1:
                    # Indicator vizual pentru RVOL
                    v_color = "#3FB950" if rvol > 1.5 else ("#F85149" if rvol < 0.7 else "#8B949E")
                    st.markdown(f"""
                        <div style="background:#161B22; padding:20px; border-radius:15px; border:2px solid {v_color}; text-align:center;">
                            <p style="color:#8B949E; margin:0; font-size:12px; text-transform:uppercase;">Volum Relativ (RVOL)</p>
                            <h1 style="color:{v_color}; margin:10px 0;">{ro(rvol):.2f}x</h1>
                        </div>
                    """, unsafe_allow_html=True)

                with v_col2:
                    # Interpretare profesională
                    if rvol > 2.0:
                        st.warning("⚠️ **ACTIVITATE INSTITUȚIONALĂ EXTREMĂ:** Volumul este de peste 2 ori mai mare decât media. Se fac mișcări mari de portofoliu.")
                    elif rvol > 1.3:
                        st.success("✅ **ACCUMULARE/INTERES:** Interes crescut în piață pentru acest activ.")
                    else:
                        st.info("⚖️ **VOLUM NORMAL:** Tranzacționare de retail, fără mișcări majore ale balenelor.")

                    # Afișăm și prețul pentru context
                    day_change = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100
                    st.write(f"Variație Preț: **{ro(day_change):+.2f}%**")
                    if rvol > 1.3 and day_change > 1.5:
                        st.markdown("🚀 **CONCLUZIE:** Achiziție agresivă detectată (Bullish Breakout).")
                    elif rvol > 1.3 and day_change < -1.5:
                        st.markdown("🚨 **CONCLUZIE:** Vânzare de panică sau descărcare instituțională (Bearish Distribution).")
            
            # --- 🕵️‍♂️ MODUL CONSOLIDAT: LEADERSHIP & ACȚIONARIAT ---
            st.markdown("---")
            st.subheader("👥 Structura Acționariatului & Leadership")

            from ai_engine import get_detailed_ownership_and_execs
            with st.spinner("Se decodează registrul acționarilor..."):
                df_summ, df_inst_list, ceo_name, total_inst_val = get_detailed_ownership_and_execs(real_sym)

            # FIX CRITIC PENTRU MASTER AI:
            # Această variabilă va fi folosită mai jos de algoritmul de decizie
            inst_percent = total_inst_val 

            if df_summ is not None:
                # Rândul 1: Carduri de Status
                c_ceo, c_total, c_graph = st.columns([1.2, 1.2, 1])
                
                with c_ceo:
                    st.markdown(f"""
                    <div style='background:#161B22; padding:20px; border-radius:15px; border-top: 4px solid #58A6FF; text-align:center; height:130px;'>
                        <p style='color:#8B949E; margin:0; font-size:12px; text-transform:uppercase;'>Director Executiv (CEO)</p>
                        <h3 style='margin:10px 0; color:var(--tk-text); font-size:18px;'>{ceo_name}</h3>
                    </div>
                    """, unsafe_allow_html=True)

                with c_total:
                    # Calculăm culoarea în funcție de dominanță
                    i_col = "#3FB950" if inst_percent > 50 else "#D29922"
                    st.markdown(f"""
                    <div style='background:#161B22; padding:20px; border-radius:15px; border-top: 4px solid {i_col}; text-align:center; height:130px;'>
                        <p style='color:#8B949E; margin:0; font-size:12px; text-transform:uppercase;'>Dețineri Instituționale Totale</p>
                        <h2 style='margin:10px 0; color:{i_col};'>{ro(inst_percent):.2f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)

                with c_graph:
                    # Donut chart minimalist pentru context
                    fig_mini = go.Figure(data=[go.Pie(
                        labels=df_summ['Tip Acționar'],
                        values=df_summ['Procent (%)'],
                        hole=.6,
                        marker=dict(colors=['#FFAB00', '#58A6FF', '#8B949E']),
                        textinfo='none'
                    )])
                    fig_mini.update_layout(height=130, margin=dict(t=0, b=0, l=0, r=0), showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
                    plot_ro(fig_mini, width='stretch')

                # Rândul 2: Tabelul Nominal (Balenele)
                st.markdown("#### 🏛️ Top 10 Deținători Instituționali")
                if not df_inst_list.empty:
                    st.dataframe(
                        df_inst_list.style.format({
                            'Deținere (%)': ro_spec('.2f', '%'),
                            'Valoare ($)': ro_spec(',.0f', ' USD')
                        }), 
                        hide_index=True, 
                        width='stretch'
                    )
                else:
                    st.info("Detaliile nominale nu sunt publice pentru acest simbol.")
            else:
                st.warning("Datele despre acționari sunt momentan indisponibile.")    
            st.markdown("---")

            # 5. Calculator Fair Value (REPARAT - REACTIVITATE TOTALĂ)
            st.subheader("🧮 Calculator Valoare Intrinsecă (Valoare justă)")
            
            # Preluare Date
            last_close = float(hist['Close'].iloc[-1]) if pd.notna(hist['Close'].iloc[-1]) else 0
            eps_f = num(info, 'trailingEps') or 0
            bv_f = num(info, 'bookValue') or 0
            # Yahoo nu trimite mereu currentPrice (ETF-uri, BVB): ultima închidere e rezerva.
            price_f = num(info, 'currentPrice') or num(info, 'previousClose') or last_close
            t_curr = info.get('currency') or 'USD'

            # --- Datele DCF-ului vin din situațiile financiare, nu din `info` ---
            fin = get_financial_statements(real_sym)
            dcf_in = fund.dcf_inputs(fin.get("income"), fin.get("balance"), fin.get("cashflow"),
                                     fin.get("q_cashflow"), fin.get("q_balance"))
            dcf_na = fund.dcf_not_applicable_reason(info.get('sector'), info.get('industry'), real_sym)
            fin_curr = info.get('financialCurrency')
            if (dcf_na is None and not info.get('sector') and not fin_curr
                    and not real_sym.upper().endswith(".RO")):
                # Fără sector și fără moneda de raportare nu se poate exclude o bancă sau un ADR.
                # (La BVB Yahoo nu trimite de regulă sectorul: acolo decide lista emitenților financiari.)
                dcf_na = ("Yahoo nu a trimis sectorul și moneda de raportare: nu pot verifica dacă DCF se aplică "
                          "(bancă, asigurător sau ADR). Reîncearcă citirea datelor.")
            if dcf_na is None and fin_curr and fin_curr != t_curr:
                dcf_na = (f"Situațiile financiare sunt în {fin_curr}, iar acțiunea se tranzacționează în {t_curr}: "
                          "valoarea pe acțiune cere conversie valutară, care nu e încă implementată.")
            rf_val, rf_label = get_risk_free_for_currency(t_curr)
            # Același beta ca în „Indicatori Fundamentali" și în audit (vezi resolve_beta_alpha).
            beta_used, beta_label = beta_val, risk_stats["beta_label"]
            wacc_res = fund.wacc(rf_val, beta_used, num(info, 'marketCap'), dcf_in["total_debt"],
                                 dcf_in["interest_expense"], dcf_in["tax_rate"])

            st.write("⚙️ **Ipoteze.** Creșterea se aplică ambelor modele; rata de scont și creșterea terminală, doar DCF-ului.")
            ctrl1, ctrl2, ctrl3 = st.columns(3)

            growth_help = """
            **Creșterea anuală estimată:**
            - În Graham: creșterea profitului pe acțiune (EPS).
            - În DCF: creșterea free cash flow-ului în primul an; apoi scade liniar, an de an, până la creșterea terminală din anul 5.

            Repere: 0-5% companii mature (utilități, bunuri de bază), 10-20% companii de creștere, peste 25% greu de susținut.
            """

            discount_help = """
            **Rata de Scont (Discount Rate):**
            Reprezintă randamentul minim pe care îl ceri de la această investiție pentru a justifica riscul.
            - 7-9%: Companii sigure, cu cash-flow stabil (Blue Chips).
            - 10-12%: Media pieței (S&P 500).
            - 13-15%+: Companii riscante sau cu datorii mari.
            Cu cât rata de scont e mai mare, cu atât valoarea justă calculată va fi mai mică.
            """

            gterm_help = """
            **Creșterea terminală (g):**
            Ritmul în care crește free cash flow-ul la nesfârșit după anul 5 (formula Gordon).
            Nu poate depăși creșterea economiei pe termen lung, de aceea e limitată la 3%,
            și trebuie să fie sub rata de scont. Reper uzual: 2-2,5%.
            """

            growth_val = ctrl1.slider(
                "Creștere anuală estimată (%)",
                -5, 40, 10, step=1,
                help=growth_help,
                key="v_final_g"
            )
            if dcf_in["fcf_cagr"] is not None and len(dcf_in["fcf_history"]) >= 2:
                ctrl1.caption(
                    f"Reper: FCF a crescut cu {ro(dcf_in['fcf_cagr'] * 100):.1f}% pe an între "
                    f"{dcf_in['fcf_history'].index[0]:%Y} și {dcf_in['fcf_history'].index[-1]:%Y} "
                    f"({len(dcf_in['fcf_history'])} ani raportați)."
                )

            use_wacc = False
            if wacc_res is not None:
                use_wacc = ctrl2.toggle(
                    f"Folosește WACC calculat ({ro(wacc_res['wacc'] * 100):.1f}%)",
                    value=True, key="v_final_use_wacc"
                )
            else:
                ctrl2.caption("WACC nu se poate calcula (lipsește rata fără risc, beta sau capitalizarea): alege rata manual.")
            discount_val = ctrl2.slider(
                "Rata de scont manuală (%)",
                5, 20, 9, step=1,
                help=discount_help,
                key="v_final_d",
                disabled=use_wacc
            )
            gterm_val = ctrl3.slider(
                "Creștere terminală g (%)",
                0.0, 3.0, 2.0, step=0.5,
                help=gterm_help,
                key="v_final_gt"
            )
            discount_rate = wacc_res['wacc'] if use_wacc else discount_val / 100

            # Baza de proiecție: FCF-ul curent sau media anilor fiscali (FCF normalizat)
            fcf_base, fcf_base_label = dcf_in["fcf"], dcf_in["fcf_basis"] or "N/A"
            fcf_distortion = fund.fcf_distortion_warning(dcf_in["cfo"], dcf_in["capex"], dcf_in["fcf"], dcf_in["fcf_average"])
            avg_label, fcf_on_average = None, False
            if dcf_in["fcf_average"] is not None:
                avg_label = f"Media ultimilor {dcf_in['fcf_average_years']} ani fiscali"
                fcf_choice = st.radio(
                    "FCF de pornire în DCF", ["FCF curent", avg_label], horizontal=True, key="v_final_fcf_base",
                    help="FCF curent = ultimele 4 trimestre (sau ultimul an fiscal). Media anilor fiscali netezește "
                         "un vârf de investiții sau un an atipic. Alegerea rămâne valabilă și când schimbi simbolul."
                )
                if fcf_choice == avg_label:
                    fcf_base, fcf_base_label = dcf_in["fcf_average"], avg_label.lower()
                    fcf_on_average = True
            if fcf_distortion and not dcf_na:
                st.warning(f"⚠️ DCF: {fcf_distortion} Compară cu varianta pe media anilor fiscali.")

            # --- LOGICĂ REACTIVĂ ---
            # 1. Graham, formula revizuită: V = EPS × (8,5 + 2g) × 4,4 / Y (analytics/fundamentals.py).
            #    None = nu se aplică (EPS ≤ 0 sau lipsește randamentul AAA); nu se înlocuiește cu 0.
            graham_y, graham_y_label = get_graham_yield(t_curr)
            graham_calc = fund.graham_revised(num(info, 'trailingEps'), growth_val, graham_y)
            graham_num = fund.graham_number(num(info, 'trailingEps'), num(info, 'bookValue'))
            if graham_calc is not None:
                graham_note = (f"Y = {ro(graham_y):.2f}%" + (f" · creștere plafonată la {ro(fund.GRAHAM_MAX_GROWTH):.0f}%"
                                                         if growth_val > fund.GRAHAM_MAX_GROWTH else ""))
            elif (num(info, 'trailingEps') or 0) <= 0:
                graham_note = "EPS lipsă sau negativ"
            else:
                graham_note = graham_y_label
            graham_num_txt = f"Nr. Graham: {ro(graham_num):.2f}" if graham_num is not None else "Nr. Graham: N/A"
            
            # 2. DCF pe free cash flow (analytics/fundamentals.py). None = modelul nu se aplică.
            dcf_res = fund.dcf_fcf(fcf_base, growth_val / 100, discount_rate, gterm_val / 100,
                                   dcf_in["net_debt"], dcf_in["shares"])
            if dcf_na:
                dcf_res = dict(dcf_res, per_share=None, reason=dcf_na, warnings=[])
            dcf_calc = dcf_res["per_share"]

            # --- AFISARE REZULTATE ---
            if price_f > 0:
                cv1, cv2, cv3 = st.columns(3)
                def _value_card(title, value, color, verdict="", note="", unit=""):
                    """Card de evaluare cu patru rânduri fixe (titlu, valoare, verdict, notă), ca
                    cele trei carduri să rămână aliniate indiferent câte rânduri au conținut.
                    Marginile sunt puse explicit: stilul implicit Streamlit pentru <p>/<h1> scotea textul din chenar."""
                    unit_html = f' <span style="font-size:14px; font-weight:400;">{html.escape(unit)}</span>' if unit else ""
                    return (
                        f'<div style="border:2px solid {color}; border-radius:12px; background-color:#161B22; '
                        'box-sizing:border-box; min-height:200px; padding:18px 16px; text-align:center; '
                        'display:flex; flex-direction:column;">'
                        f'<div style="color:#8B949E; font-size:13px; line-height:18px; text-transform:uppercase; letter-spacing:0.3px;">{html.escape(title)}</div>'
                        f'<div style="color:{"#C9D1D9" if color == "#30363D" else color}; font-size:40px; line-height:46px; font-weight:700; flex:1; display:flex; align-items:center; justify-content:center; gap:8px;">{html.escape(value)}{unit_html}</div>'
                        f'<div style="color:{color if color != "#30363D" else "#8B949E"}; font-size:13px; line-height:18px; font-weight:700; min-height:18px;">{html.escape(verdict)}</div>'
                        f'<div style="color:#C9D1D9; font-size:12px; line-height:16px; min-height:16px;">{html.escape(note)}</div>'
                        '</div>'
                    )

                # Verde / galben / roșu; galben = în banda neutră de ±10% (fund.VALUATION_NEUTRAL_BAND).
                VERDICT_COLORS = {"under": "#3FB950", "fair": "#D29922", "over": "#F85149"}
                with cv1:
                    st.markdown(_value_card("Preț curent", f"{ro(price_f):.2f}", "#30363D", unit=t_curr), unsafe_allow_html=True)

                with cv2:
                    g_verdict, g_zone, _ = fund.valuation_verdict(price_f, graham_calc)
                    if graham_calc is not None and g_verdict is not None:
                        g_col = VERDICT_COLORS[g_zone]
                        st.markdown(_value_card(
                            "Graham (formula revizuită)", f"{ro(graham_calc):.2f}", g_col,
                            verdict=g_verdict,
                            note=f"{graham_note} · {graham_num_txt}"), unsafe_allow_html=True)
                    else:
                        st.markdown(_value_card("Graham (formula revizuită)", "N/A", "#30363D",
                                                verdict=graham_note, note=graham_num_txt), unsafe_allow_html=True)

                with cv3:
                    d_verdict, d_zone, _ = fund.dcf_verdict(price_f, dcf_calc, bool(fcf_distortion), fcf_on_average)
                    if dcf_calc is not None and d_zone == "none":
                        # FCF curent deformat: valoarea rămâne afișată, dar cardul e gri și fără etichetă.
                        st.markdown(_value_card(
                            "Valoare justă (DCF pe FCF)", f"{ro(dcf_calc):.2f}", "#30363D",
                            verdict=d_verdict,
                            note=(f"Vezi varianta „{avg_label}”." if avg_label
                                  else "Media anilor fiscali lipsește: fără bază de comparație.")), unsafe_allow_html=True)
                    elif dcf_calc is not None and d_verdict is not None:
                        d_col = VERDICT_COLORS[d_zone]
                        st.markdown(_value_card(
                            "Valoare justă (DCF pe FCF)", f"{ro(dcf_calc):.2f}", d_col,
                            verdict=d_verdict,
                            note=f"scont {ro(discount_rate * 100):.1f}% · g terminal {ro(gterm_val):.1f}%"), unsafe_allow_html=True)
                    else:
                        st.markdown(_value_card("Valoare justă (DCF pe FCF)", "N/A", "#30363D",
                                                note=dcf_res["reason"] or "Date insuficiente."), unsafe_allow_html=True)

            st.write("")  # spațiu între carduri și tabelul de sensibilitate
            for dcf_warning in dcf_res["warnings"]:
                st.warning(f"⚠️ DCF: {dcf_warning}")

            if dcf_calc is not None:
                # Un singur număr induce în eroare: valoarea pe acțiune pe o grilă de ipoteze.
                st.markdown(f"**Sensibilitatea DCF** — valoare pe acțiune ({t_curr}) în funcție de rata de scont și de creșterea terminală")
                r_grid = [discount_rate + d for d in (-0.02, -0.01, 0.0, 0.01, 0.02)]
                g_grid = [0.01, 0.015, 0.02, 0.025, 0.03]
                sens = fund.dcf_sensitivity(fcf_base, growth_val / 100, dcf_in["net_debt"], dcf_in["shares"], r_grid, g_grid)
                sens.index = [f"scont {ro(r * 100):.1f}%" for r in r_grid]
                sens.columns = [f"g {ro(g * 100):.1f}%" for g in g_grid]

                def _sens_color(v):
                    if pd.isna(v) or not price_f:
                        return f"color: {DF_MUTED}"
                    return f"color: {DF_UP}" if v > price_f else f"color: {DF_DOWN}"

                st.dataframe(sens.style.format(ro_spec(",.2f")).map(_sens_color), width='stretch')
                st.caption(
                    f"Verde = peste prețul curent ({ro(price_f):.2f} {t_curr}), roșu = sub. Rândul din mijloc e rata de scont folosită. "
                    "Dacă verdictul se schimbă între celule vecine, modelul nu susține o concluzie fermă."
                )

            with st.expander("🔎 Baza de calcul a DCF-ului (de verificat față de situațiile financiare)"):
                if fin["errors"]:
                    st.caption("Situații financiare pe care Yahoo nu le-a trimis: " + ", ".join(fin["errors"]))
                bal_date = f"{dcf_in['balance_date']:%d.%m.%Y}" if dcf_in["balance_date"] is not None else "N/A"
                if dcf_in["shares"] is None:
                    shares_src = "N/A"
                else:
                    shares_src = "medie diluată, ultimul an fiscal" if dcf_in["shares_diluted"] else "din bilanț, NEDILUAT (media diluată lipsește)"
                base_rows = [
                    ("Flux de numerar din exploatare (CFO)", format_amount(dcf_in["cfo"]), dcf_in["fcf_basis"] or "N/A"),
                    ("Cheltuieli de capital (capex)", format_amount(dcf_in["capex"]), dcf_in["fcf_basis"] or "N/A"),
                    ("Free cash flow = CFO − |capex|", format_amount(dcf_in["fcf"]), dcf_in["fcf_basis"] or "N/A"),
                    ("FCF de pornire folosit în DCF", format_amount(fcf_base), fcf_base_label),
                    ("Datorie totală", format_amount(dcf_in["total_debt"]), f"bilanț {bal_date}"),
                    ("Numerar și plasamente pe termen scurt", format_amount(dcf_in["cash"]), f"bilanț {bal_date}"),
                    ("Datorie netă", format_amount(dcf_in["net_debt"]), "datorie totală − numerar"),
                    ("Număr de acțiuni", format_amount(dcf_in["shares"]), shares_src),
                ]
                st.dataframe(pd.DataFrame(base_rows, columns=["Element", f"Valoare ({fin_curr or t_curr})", "Sursă"]),
                             hide_index=True, width='stretch')

                st.markdown("**Rata de scont**")
                if wacc_res is not None:
                    kd_txt = f"{ro(wacc_res['kd'] * 100):.2f}%" if wacc_res['kd'] is not None else "N/A"
                    tax_txt = f"{ro(wacc_res['tax'] * 100):.1f}%" if wacc_res['tax'] is not None else "N/A"
                    st.write(
                        f"WACC = **{ro(wacc_res['wacc'] * 100):.2f}%** "
                        f"= {ro(wacc_res['w_e'] * 100):.0f}% × cost capital propriu {ro(wacc_res['ke'] * 100):.2f}% "
                        f"+ {ro(wacc_res['w_d'] * 100):.0f}% × cost datorie {kd_txt} × (1 − impozit {tax_txt})"
                    )
                    st.write(
                        f"Cost capital propriu (CAPM) = rată fără risc {ro(wacc_res['rf'] * 100):.2f}% [{rf_label}] "
                        f"+ beta {ro(wacc_res['beta']):.2f} ({beta_label}) × primă de risc {ro(wacc_res['erp'] * 100):.1f}% (ipoteză fixă)"
                    )
                    for wacc_note in wacc_res["notes"]:
                        st.caption(f"Aproximare: {wacc_note}")
                else:
                    st.write(f"WACC indisponibil. Rata fără risc: {rf_label if rf_val is None else f'{ro(rf_val * 100):.2f}% [{rf_label}]'}")
                st.write(f"Rata de scont folosită în calcul: **{ro(discount_rate * 100):.2f}%** ({'WACC calculat' if use_wacc else 'aleasă manual'}).")

                if dcf_res["flows"]:
                    st.markdown("**Proiecția pe 5 ani**")
                    df_flows = pd.DataFrame(dcf_res["flows"])
                    df_flows = pd.DataFrame({
                        "An": df_flows["year"],
                        "Creștere": df_flows["growth"].map(lambda v: f"{ro(v * 100):.1f}%"),
                        "FCF proiectat": df_flows["fcf"].map(format_amount),
                        "Valoare actualizată": df_flows["pv"].map(format_amount),
                    })
                    st.dataframe(df_flows, hide_index=True, width='stretch')
                    st.write(
                        f"Valoarea întreprinderii {format_amount(dcf_res['enterprise_value'])} "
                        f"= fluxuri actualizate {format_amount(dcf_res['pv_fcf'])} "
                        f"+ valoare terminală actualizată {format_amount(dcf_res['pv_terminal'])} "
                        f"({ro(dcf_res['terminal_share'] * 100):.0f}% din total). "
                        f"Minus datoria netă {format_amount(dcf_in['net_debt'])} "
                        f"= valoarea capitalului {format_amount(dcf_res['equity_value'])}."
                    )
                if len(dcf_in["fcf_history"]):
                    st.markdown("**Istoricul FCF (ani fiscali)**")
                    st.dataframe(pd.DataFrame({
                        "An fiscal încheiat": [f"{d:%d.%m.%Y}" for d in dcf_in["fcf_history"].index],
                        "FCF": [format_amount(v) for v in dcf_in["fcf_history"].values],
                    }), hide_index=True, width='stretch')
                st.caption(
                    "Limite: modelul proiectează un singur scenariu de creștere; beta depinde de perioada și de "
                    "benchmarkul folosit (sursa e scrisă mai sus); prima de risc de 5% este o ipoteză, nu o măsurătoare."
                )

            # --- ALTMAN Z: risc de dificultate financiară, din situațiile financiare ---
            st.markdown("---")
            st.subheader("🏦 Risc de dificultate financiară (Altman)")
            altman_res = calculate_altman_z(info, real_sym, fin)
            az_left, az_right = st.columns([1, 2])
            with az_left:
                az_value = f"{ro(altman_res['value']):.2f}" if altman_res["value"] is not None else "N/A"
                az_title = f"Altman {altman_res['variant']}" if altman_res["variant"] else "Altman Z"
                st.markdown(f"""
                <div style="background:#161B22; padding:25px; border-radius:15px; border:2px solid {altman_res['color']}; text-align:center;">
                    <p style="color:#8B949E; margin:0; font-size:12px; text-transform:uppercase;">{az_title}</p>
                    <h1 style="color:{altman_res['color']}; margin:10px 0; font-size:40px;">{az_value}</h1>
                    <p style="color:{altman_res['color']}; font-weight:bold; font-size:12px; margin:0;">{altman_res['label']}</p>
                </div>
                """, unsafe_allow_html=True)
            with az_right:
                st.write(altman_res["message"])
                if altman_res["variant"] == "Z":
                    st.caption("Z original (companii de producție): sub 1,81 dificultate · 1,81–2,99 zonă gri · peste 2,99 sigur.")
                elif altman_res["variant"] == "Z''":
                    st.caption("Z'' (servicii, piețe emergente sau sector necunoscut): sub 1,1 dificultate · 1,1–2,6 zonă gri · peste 2,6 sigur.")
                az_detail = altman_res["detail"]
                if az_detail is not None:
                    with st.expander("Componentele scorului"):
                        az_names = {
                            "X1": "X1 = capital de lucru / active", "X2": "X2 = rezultat reportat / active",
                            "X3": "X3 = EBIT / active", "X4": "X4 = capitalizare / datorii totale (în Z)",
                            "X4_book": "X4' = capital propriu contabil / datorii totale (în Z'')",
                            "X5": "X5 = venituri / active (în Z)",
                        }
                        st.dataframe(pd.DataFrame(
                            [(az_names[k], "N/A" if v is None else f"{ro(v):.3f}") for k, v in az_detail["components"].items()],
                            columns=["Componentă", "Valoare"]), hide_index=True, width='stretch')
                        az_date = f"{az_detail['balance_date']:%d.%m.%Y}" if az_detail["balance_date"] is not None else "N/A"
                        z_txt = f"{ro(az_detail['z']):.2f}" if az_detail["z"] is not None else "N/A"
                        z2_txt = f"{ro(az_detail['z2']):.2f}" if az_detail["z2"] is not None else "N/A"
                        st.caption(f"Bilanț: {az_date} · Z = {z_txt} · Z'' = {z2_txt}. Model statistic din 1968/1995: "
                                   "un semnal de avertizare, nu o predicție.")
                        if az_detail["retained_is_proxy"]:
                            st.caption("Aproximare: Yahoo nu are rândul „rezultat reportat” pentru această companie; "
                                       "X2 folosește capital propriu − capital social − prime de emisiune (rezerve + rezultat reportat).")
                        if az_detail["missing"] and az_detail["balance_rows"]:
                            st.caption("Rânduri disponibile în bilanțul trimis de Yahoo: " + ", ".join(az_detail["balance_rows"]))

            # --- PIOTROSKI F-SCORE și îndatorare, din situațiile financiare anuale ---
            st.markdown("---")
            st.subheader("📋 Calitate financiară (Piotroski F-Score)")
            pio_financial = fund.is_financial_issuer(info.get('sector'), real_sym)
            pio = fund.piotroski(fin.get("income"), fin.get("balance"), fin.get("cashflow"), financial=pio_financial)
            lev = fund.leverage_ratios(fin.get("income"), fin.get("balance"), fin.get("q_income"), fin.get("q_balance"))
            pio_left, pio_right = st.columns([1, 2])
            with pio_left:
                if pio["evaluable"] == 0:
                    pio_color, pio_value, pio_label = "#8B949E", "N/A", "Situații financiare indisponibile"
                else:
                    pio_share = pio["passed"] / pio["evaluable"]
                    pio_color = "#3FB950" if pio_share >= 7 / 9 else ("#D29922" if pio_share >= 4 / 9 else "#F85149")
                    pio_value = f"{pio['passed']}/{pio['evaluable']}"
                    pio_label = ("toate cele 9 criterii evaluate" if pio["evaluable"] == 9
                                 else f"{9 - pio['evaluable']} criterii fără date (N/A)")
                    if pio_financial:
                        # Scorul e calibrat pe companii nefinanciare: la bănci rămân puține criterii, deci fără verdict colorat.
                        pio_color = "#8B949E"
                        pio_label = f"emitent financiar: {9 - pio['evaluable']} criterii nu se aplică, fără verdict"
                st.markdown(f"""
                <div style="background:#161B22; padding:25px; border-radius:15px; border:2px solid {pio_color}; text-align:center;">
                    <p style="color:#8B949E; margin:0; font-size:12px; text-transform:uppercase;">Criterii trecute</p>
                    <h1 style="color:{pio_color}; margin:10px 0; font-size:40px;">{pio_value}</h1>
                    <p style="color:#C9D1D9; font-size:12px; margin:0;">{pio_label}</p>
                </div>
                """, unsafe_allow_html=True)
                nd_ebitda = lev["net_debt_to_ebitda"]
                st.metric("Datorie netă / EBITDA", "N/A" if nd_ebitda is None else f"{ro(nd_ebitda):.2f}x",
                          help=f"{RISK_HELP['nd_ebitda']}\n\n**Valori:** datorie netă {format_amount(lev['net_debt'])}, "
                               f"EBITDA {format_amount(lev['ebitda'])}.")
                int_cov = lev["interest_coverage"]
                st.metric("Acoperirea dobânzii", "N/A" if int_cov is None else f"{ro(int_cov):.1f}x",
                          help=f"{RISK_HELP['int_cov']}\n\n**Valori:** EBIT {format_amount(lev['ebit'])}, "
                               f"dobânzi {format_amount(lev['interest_expense'])}."
                               + (" N/A aici: dobânzile raportate depășesc 25% din datorie, deci rândul include și alte "
                                  "costuri financiare." if lev["interest_unreliable"] else ""))
            with pio_right:
                pio_icons = {True: "✅ trecut", False: "❌ picat", None: "➖ N/A"}
                st.dataframe(pd.DataFrame(
                    [(c["name"], pio_icons[c["passed"]], c["detail"]) for c in pio["criteria"]],
                    columns=["Criteriu", "Rezultat", "Valoare (an curent față de precedent)"]),
                    hide_index=True, width='stretch')
                if pio["year"] is not None and pio["prior_year"] is not None:
                    st.caption(f"An fiscal încheiat la {pio['year']:%d.%m.%Y} față de {pio['prior_year']:%d.%m.%Y}. "
                               "Scorul măsoară direcția (îmbunătățire sau deteriorare), nu nivelul: 7–9 solid, 0–3 slab. "
                               "La bănci, asigurători și fonduri, criteriile bazate pe fluxul din exploatare (2 și 4) nu se aplică, "
                               "iar cele de lichiditate și marjă brută nu au date: scorul rămâne orientativ, fără verdict.")
                else:
                    st.caption("Piotroski are nevoie de doi ani fiscali de situații financiare.")

            # --- RAPORT FINAL PE CATEGORII ---
            st.markdown("---")
            st.subheader("🕵️‍♂️ Audit Instituțional (6 Piloni)")
            
            # 1. EXECUTĂM CALCULELE (Aceste rânduri îți lipsesc acum!)
            h_score, pros, cons = calculate_health_score_ext(info)
            audit_report = generate_advanced_audit_v2(info, alpha_val, beta_val, h_score)
            
            # 2. PREGĂTIM DATELE PENTRU AFIȘARE (Plasa de siguranță anti-crash)
            display_beta = f"{ro(beta_val):.2f}" if beta_val is not None else "N/A"
            display_alpha = f"{ro(alpha_val*100):.1f}%" if alpha_val is not None else "N/A"

            # 3. STABILIM CULOAREA SCORULUI
            if h_score is None:
                h_color, h_display = "#8B949E", "N/A"
            else:
                h_color = "#3FB950" if h_score >= 8 else ("#D29922" if h_score >= 5 else "#F85149")
                h_display = str(h_score)
            
            c_left, c_right = st.columns([1, 2])
            
            with c_left:
                st.markdown(f"""
                <div style="background:#161B22; padding:30px; border-radius:15px; border:2px solid {h_color}; text-align:center;">
                    <p style="color:#8B949E; margin:0; font-size:12px; text-transform:uppercase;">Scor Sănătate Financiară</p>
                    <h1 style="color:{h_color}; margin:15px 0; font-size:54px;">{h_display}<span style="font-size:18px;">/10</span></h1>
                    <hr style="border-color:#30363D;">
                    <p style="font-size:13px; color:#8B949E;">Beta: {display_beta} | Alpha: {display_alpha}</p>
                </div>
                """, unsafe_allow_html=True)
                
            with c_right:
                st.markdown(f"**Analiză de Specialist în Sectorul:** `{info.get('sector') or 'N/A'}`")
                for line in audit_report:
                    st.info(line)

            # --- MODUL MARJA DE SIGURANȚĂ REPARAT (VARIANTA SENIOR DEV) ---
            st.markdown("---")
            st.subheader("🛡️ Analiza Marjei de Siguranță")
            
            # 1. INIȚIALIZARE OBLIGATORIE LA NIVEL 0 (Zero UnboundLocalError)
            current_p = num(info, 'currentPrice') or num(info, 'previousClose') or last_close
                
            target_val = 0.0
            mos_val = None  # None = DCF indisponibil; nu intră în scoruri
            mos_verdict = "N/A"
            mos_color = "#8B949E"

            # Validăm dacă DCF-ul a întors o valoare validă
            if dcf_calc is not None and dcf_calc > 0:
                target_val = dcf_calc

            # 2. LOGICĂ DE CALCUL UNICĂ ȘI SECURIZATĂ
            if target_val > 0 and current_p > 0:
                mos_val = ((target_val - current_p) / target_val) * 100
                
                if mos_val > 30:
                    mos_verdict = "🚀 **Oportunitate Majoră (Deep Value):** Acțiunea se vinde cu un discount masiv."
                    mos_color = "#3FB950" 
                elif mos_val > 10:
                    mos_verdict = "✅ **Preț Atractiv:** Există o marjă de siguranță rezonabilă."
                    mos_color = "#3FB950"
                elif mos_val > -10:
                    mos_verdict = "⚖️ **Evaluare neutră:** Prețul este corect. Nu ai marjă de siguranță clară."
                    mos_color = "#D29922" 
                else:
                    mos_verdict = "🚨 **SUPRAEVALUARE CRITICĂ:** Plătești un premium periculos."
                    mos_color = "#F85149"

                # Afișare interfață
                m_col1, m_col2 = st.columns([1, 2])
                with m_col1:
                    st.markdown(f"""
                    <div style="background:#161B22; padding:25px; border-radius:15px; border:2px solid {mos_color}; text-align:center;">
                        <p style="color:#8B949E; margin:0; font-size:12px; text-transform:uppercase;">Margin of Safety</p>
                        <h1 style="color:{mos_color}; margin:10px 0; font-size:40px;">{ro(mos_val):.1f}%</h1>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with m_col2:
                    if mos_val < -20:
                        st.error(f"⚠️ **ALERTĂ DE RISC:** {mos_verdict}")
                        st.write("👉 *Sfat Analist:* Istoric, cumpărarea la acest premium aduce randamente slabe.")
                    elif mos_val > 30:
                        st.success(f"🌟 **SURPRIZĂ DE VALOARE:** {mos_verdict}")
                    else:
                        st.info(mos_verdict)
                    
                    st.write(f"💵 **Preț Actual:** {ro(current_p):.2f} | 🎯 **Fair Value:** {ro(target_val):.2f}")
                    st.progress(max(0.0, min(mos_val / 100.0, 1.0)))
            else:
                st.warning(f"⚠️ Marja de siguranță indisponibilă: {dcf_res['reason'] or 'lipsește prețul curent.'}")

            # --- MODUL: SUSTENABILITATE ȘI CALITATE (VERSIUNE EXTINSĂ) ---
            st.markdown("---")
            st.subheader("🧬 Sustenabilitate și Calitatea Profitului")
            
            div_verdicts, q_ratio, p_ratio = analyze_dividend_quality(info)
            q_display = f"{ro(q_ratio):.2f}x" if q_ratio is not None else "N/A"
            p_display = f"{ro(p_ratio):.1f}%" if p_ratio is not None else "N/A"
            if q_ratio is None:
                q_color = "#8B949E"
            else:
                q_color = "#3FB950" if q_ratio > 1 else ("#D29922" if q_ratio > 0.7 else "#F85149")
            
            c_q1, c_q2 = st.columns([1, 2])
            
            with c_q1:
                # Vizualizare principală scor cash
                st.markdown(f"""
                <div style="background:#161B22; padding:20px; border-radius:15px; border:2px solid {q_color}; text-align:center;">
                    <p style="color:#8B949E; margin:0; font-size:12px; text-transform:uppercase;">Cash-to-Income Ratio</p>
                    <h1 style="color:{q_color}; margin:10px 0; font-size:35px;">{q_display}</h1>
                    <p style="font-size:12px; color:#8B949E;">Plată Dividend (Payout): {p_display}</p>
                </div>
                """, unsafe_allow_html=True)
                
            with c_q2:
                # Verdictul textual
                if div_verdicts:
                    for v in div_verdicts:
                        st.write(v)
                else:
                    st.write("⚖️ Parametrii de sustenabilitate sunt în limitele de siguranță.")

            # --- EXPLICATII PERMANENTE (FĂRĂ CLICK) ---
            st.markdown("#### 💡 Ghid de Interpretare Rapidă")
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.markdown(f"""
                **Ce înseamnă {q_display} (Cash-to-Income)?**
                * **Peste 1.0x:** Afacere de tip 'Cash Machine'. Firma încasează mai mulți bani reali decât profitul declarat contabil.
                * **0.7x - 1.0x:** Nivel normal pentru companii în creștere.
                * **Sub 0.7x:** Profitul este doar pe hârtie. Există riscul ca facturile să nu fie încasate.
                """)
                
            with col_info2:
                st.markdown(f"""
                **Ce înseamnă {p_display} (Payout Ratio)?**
                * **30% - 60%:** Zona ideală (Sweet Spot). Dividend sigur și loc de creștere.
                * **Peste 80%:** Zona de pericol. Firma dă aproape tot profitul afară; orice scădere a vânzărilor va duce la tăierea dividendului.
                """)

            # --- 1. PROFILUL COMPANIEI (EXECUTIVE SUMMARY) ---
            st.markdown("---")
            with st.expander("🏢 Vezi Profilul Companiei & Modelul de Business", expanded=True):
                col_desc1, col_desc2 = st.columns([2, 1])
                
                with col_desc1:
                    st.markdown("#### 📖 Descriere Activitate")
                    summary = info.get('longBusinessSummary') or 'Descriere indisponibilă.'
                    # Afișăm doar primele 600 de caractere cu opțiune de expandare dacă e prea lungă
                    st.write(summary if len(summary) < 600 else summary[:600] + "...")
                
                with col_desc2:
                    st.markdown("#### 🛠️ Detalii Tehnice")
                    
                    # Extragem valoarea și verificăm dacă este număr
                    emp = info.get('fullTimeEmployees')
                    emp_display = f"{ro(emp):,}" if isinstance(emp, (int, float)) else "N/A"
                    
                    st.write(f"**Angajați:** {emp_display}")
                    st.write(f"**Sediu:** {info.get('city', 'N/A')}, {info.get('country', 'N/A')}")
                    st.write(f"**Website:** [Vizitează site]({info.get('website', '#')})")
                    st.write(f"**Industrie:** {info.get('industry', 'N/A')}")

            # --- 2. PUNCTE FORTE ȘI SLABE (DETECȚIE AUTOMATĂ) ---
            st.markdown("#### 🧠 Analiză Calitativă Rapidă")
            q_col1, q_col2 = st.columns(2)
            
            # Logica de generare a punctelor (bazată pe datele din INFO)
            with q_col1:
                st.success("**✅ Puncte Forte (Competitive Advantages)**")
                # Detectăm "Moat"-ul prin marje și cash
                if (num(info, 'profitMargins') or 0) > 0.20: st.write("• **Marje ridicate:** Putere mare de stabilire a prețurilor (Pricing Power).")
                if num(info, 'totalCash') is not None and num(info, 'totalDebt') is not None and num(info, 'totalCash') > num(info, 'totalDebt'): st.write("• **Poziție Net Cash:** Bilanț extrem de rezistent la crize.")
                if (num(info, 'returnOnEquity') or 0) > 0.15: st.write("• **Eficiență Capital:** Management performant în alocarea resurselor.")
                if "Technology" in (info.get('sector') or ''): st.write("• **Scalabilitate:** Model de business bazat pe IP și software.")

            with q_col2:
                st.error("**⚠️ Vulnerabilități (Potential Risks)**")
                if (num(info, 'debtToEquity') or 0) > 150: st.write("• **Levier ridicat:** Expunere mare la creșterea dobânzilor.")
                if (num(info, 'payoutRatio') or 0) > 0.80: st.write("• **Dividend la limită:** Spațiu restrâns pentru investiții viitoare.")
                if num(info, 'forwardPE') is not None and num(info, 'trailingPE') is not None and num(info, 'forwardPE') > num(info, 'trailingPE') > 0: st.write("• **Așteptări în scădere:** Piața anticipează o încetinire a profitului.")
                if beta_val is not None and beta_val > 1.5: st.write("• **Volatilitate Mare:** Sensibilitate ridicată la panica din piața generală.")
            
            # --- MODUL: ANALIZĂ STRATEGICĂ IA (SWOT) ---
            st.markdown("---")
            st.subheader("🎯 Analiză Strategică IA (SWOT)")
            try:
                # Importuri din modulul extern
                from ai_engine import analyze_sentiment_ai, generate_ai_swot_analysis
                
                # Colectare date necesare pentru SWOT
                c_news_ai = get_company_news_rss(real_sym, names=(info.get('longName'), info.get('shortName')))
                s_score_val = analyze_sentiment_ai(c_news_ai) if c_news_ai else None   # lipsă ≠ neutru
                mos_swot = ((dcf_calc - current_p) / dcf_calc * 100) if (dcf_calc is not None and dcf_calc > 0) else None
                z_val_swot = altman_res["zone"]   # zona, nu scorul: pragul depinde de variantă
                
                # Generare date SWOT
                swot_res = generate_ai_swot_analysis(info, h_score, z_val_swot, mos_swot, alpha_val, s_score_val, yield_spread=spread)
                
                # Randare vizuală pe coloane
                s_col1, s_col2 = st.columns(2)
                with s_col1:
                    st.success("**💪 PUNCTE TARI**")
                    for item in swot_res["Strengths"]:
                        st.write(f"• {item}")
                    st.warning("**🌟 OPORTUNITĂȚI**")
                    for item in swot_res["Opportunities"]:
                        st.write(f"• {item}")
                with s_col2:
                    st.error("**⚠️ PUNCTE SLABE**")
                    for item in swot_res["Weaknesses"]:
                        st.write(f"• {item}")
                    st.info("**🚩 AMENINȚĂRI**")
                    for item in swot_res["Threats"]:
                        st.write(f"• {item}")
            except Exception as e:
                st.warning(f"Modulul SWOT IA este momentan indisponibil.")
            st.markdown("---")

            # 6. Terminal Intelligence AI (SENTIMENT & PROGNOZĂ)
            st.subheader("🤖 Terminal Intelligence (AI & ML)")
            # --- RADAR REGIM DE PIAȚĂ CU VALIDARE STATISTICĂ ---
            with st.spinner("AI-ul validează regimul de piață..."):
                from ai_engine import detect_market_regime_ai
                # Primim acum 3 valori: mesaj, culoare și statistica de backtest
                regime_msg, regime_color, backtest_info = detect_market_regime_ai(hist)
                
                st.markdown(f"""
                <div style='background:#161B22; padding:20px; border-radius:15px; border-left: 5px solid {regime_color}; margin-bottom: 20px;'>
                    <p style='margin:0; color:#8B949E; font-size: 12px; text-transform: uppercase; font-weight: bold; letter-spacing:1px;'>Radar AI: Clasificare Regim (K-Means)</p>
                    <h3 style='margin:10px 0; color:{regime_color}; font-size: 22px;'>{regime_msg}</h3>
                    <div style='background:rgba(255,255,255,0.03); padding:10px; border-radius:8px; border: 1px dashed {regime_color}55;'>
                        <p style='margin:0; font-size:13px; color:#C9D1D9;'>{backtest_info}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                       
            c_news_ai = get_company_news_rss(real_sym, names=(info.get('longName'), info.get('shortName')))
            cai1, cai2 = st.columns([1, 2])
            
            # --- PARTEA ACTUALIZATĂ (Scorul FinBERT cu Legendă) ---
            with cai1:
                st.write("📊 **Analiză Sentiment AI (știrile recente)**")
                if c_news_ai:
                    with st.spinner("FinBERT procesează narațiunea pieței..."):
                        s_score = analyze_sentiment_ai(c_news_ai)
                        # Aceleași praguri ca în „Context” și în sinteză (decision.sentiment_band, 14.11)
                        if s_score is None:      # FinBERT indisponibil: fără scor, nu 0 „neutru”
                            c_ai, status_text = "#8B949E", "Indisponibil: FinBERT nu a putut analiza știrile"
                        else:
                            s_lbl, s_lvl = sentiment_band(s_score)
                            c_ai = {"up": "#3FB950", "down": "#F85149"}.get(s_lvl, "#8B949E")
                            status_text = s_lbl
                        news_txt = html.escape(news_window(c_news_ai[:10]))   # exact titlurile analizate

                        st.markdown(f"""
                        <div style='background:#161B22; padding:20px; border-radius:15px; border-left: 5px solid {c_ai}; text-align:center;'>
                            <h1 style='color:{c_ai}; margin:0; font-size:48px;'>{"N/A" if s_score is None else f"{ro(s_score):.2f}"}</h1>
                            <div style='color:{c_ai}; font-weight:bold; font-size:18px; margin-top:5px;'>{status_text}</div>
                            <p style='color:#8B949E; font-size:12px; margin-top:10px;'>{news_txt} · netestat</p>
                        </div>

                        <div style='margin-top: 15px; padding: 15px; background: #21262D; border-radius: 10px; border-left: 3px solid #58A6FF;'>
                            <p style='color:#8B949E; font-size: 12px; margin:0 0 8px 0; text-transform: uppercase; font-weight: bold;'>Cum se calculează (scală −1 la +1)</p>
                            <div style='color:#C9D1D9; font-size: 13px; line-height: 1.8;'>
                                <div>Media pe titluri: +încrederea FinBERT la „pozitiv”, −încrederea la „negativ”, 0 la „neutru”. Nu e o probabilitate.</div>
                                <div><span style='color:#3FB950;'>■</span> <b>≥ +0,25:</b> Euforie media</div>
                                <div><span style='color:#3FB950; opacity:0.7;'>■</span> <b>+0,15 la +0,25:</b> Optimism</div>
                                <div><span style='color:#8B949E;'>■</span> <b>−0,15 la +0,15:</b> Neutru</div>
                                <div><span style='color:#F85149; opacity:0.7;'>■</span> <b>−0,25 la −0,15:</b> Pesimism</div>
                                <div><span style='color:#F85149;'>■</span> <b>≤ −0,25:</b> Panică</div>
                                <div style='color:#8B949E; font-size:12px;'>Praguri netestate; cu puține titluri, unul singur poate muta scorul.</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("Sentiment N/A: în fluxul Yahoo nu există știri recente care să menționeze compania "
                               "(știrile despre alte companii sunt excluse).")
            
            # --- PREDICȚIE AI CU CACHING ---
            with cai2:
                st.write("📈 **Prognoză Algoritmică (Next 90 Days)**")
                if len(hist) > 100:
                    from ai_engine import get_cached_prophet_prediction, get_daily_hash, render_ai_chart
                    
                    # Trimitem 'get_daily_hash()' ca parametru pentru a reseta cache-ul automat a doua zi
                    forecast = get_cached_prophet_prediction(real_sym, hist, get_daily_hash())
                    
                    if forecast is not None:
                        render_ai_chart(forecast, hist)
                    else:
                        st.warning("Modelul AI nu a putut genera predicția.")
            
            # Verdict Final și Master AI vechi: șterse 13.11 (înlocuite de „Verdict final” în 4 pași).
            st.markdown("---")
            
            # ==================================================
            # MODUL NOU: HARTA SEZONALITĂȚII
            # ==================================================
           
            st.subheader("📅 Harta Sezonalității (Avantajul Statistic)")
            from ai_engine import calculate_and_plot_seasonality
            with st.spinner("Se analizează tiparele istorice lunare..."):
                fig_season, df_stats = calculate_and_plot_seasonality(hist)
                
                if fig_season is not None:
                    s_col1, s_col2 = st.columns([2, 1])
                    with s_col1:
                        plot_ro(fig_season, width='stretch')
                    
                    with s_col2:
                        # Extragem extremele pentru a oferi un verdict clar utilizatorului
                        best_m = df_stats.loc[df_stats['Win Rate (%)'].idxmax()]
                        worst_m = df_stats.loc[df_stats['Win Rate (%)'].idxmin()]
                        
                        st.markdown("#### 💡 Analiză Quant")
                        # Randamentul mediu poate avea orice semn: se afișează cu semnul real, nu cu „+" sau „scădere" fix.
                        # Numărul de ani contează: cu 5 observații, 80% înseamnă 4 din 5.
                        best_n, worst_n = int(best_m['Ani']), int(worst_m['Ani'])
                        best_wins = round(best_m['Win Rate (%)'] * best_n / 100)
                        worst_wins = round(worst_m['Win Rate (%)'] * worst_n / 100)
                        st.success(f"**🌟 Cea mai bună lună: {best_m['Luna']}**\n\nIstoric, prețul a crescut în **{best_wins} din {best_n} ani** ({ro(best_m['Win Rate (%)']):.0f}%), cu un randament mediu de **{ro(best_m['Randament Mediu (%)']):+.2f}%**.")
                        
                        st.error(f"**🚨 Cea mai slabă lună: {worst_m['Luna']}**\n\nIstoric, prețul a crescut în doar **{worst_wins} din {worst_n} ani** ({ro(worst_m['Win Rate (%)']):.0f}%), cu un randament mediu de **{ro(worst_m['Randament Mediu (%)']):+.2f}%**.")
                        
                        st.info("📉 **Cum folosești acest modul:** Elimină ghicitul și emoția. Dacă dorești să cumperi această acțiune, dar te afli într-o lună cu Win Rate sub 40%, șansele matematice sunt împotriva ta. Așteaptă luna verde pentru a deschide o poziție la un preț statistic favorabil.")
                else:
                    # Dacă df_stats este string, înseamnă că a returnat mesajul de eroare
                    st.info(df_stats)
            
            # =================================================================
            # 🕵️‍♂️ MODUL PRO: RAZE X - ANALIZĂ GEX & OPTIUNI (VERSIUNE COMPLETĂ)
            # =================================================================
            st.markdown("---")
            st.subheader("🕵️‍♂️ Raze X: Fluxul de Bani din Opțiuni (Pro)")

            from ai_engine import get_options_analysis_ai
            with st.spinner("Se citește lanțul de opțiuni..."):
                # Folosim simbolul real identificat anterior; dividendul anual pentru q (Black-Scholes)
                opt_data, opt_msg = get_options_analysis_ai(real_sym, num(info, 'dividendRate'))
            if opt_data:
                def _f(v, fmt, na="N/A"):
                    return na if v is None else format(ro(v), fmt)

                spot_opt = opt_data['spot']
                q_txt = ("q = 0 presupus (Yahoo nu dă dividendul anual)" if opt_data['q_assumed']
                         else f"dividend continuu q = {ro(opt_data['q']*100):.2f}%")
                r_txt = (f"r = {ro(opt_data['r']*100):.2f}% din {opt_data['r_src']} (proxy)" if opt_data['r'] is not None
                         else "r indisponibil → greeks și GEX N/A")
                st.caption(f"⏱️ Date opțiuni la ora {opt_data.get('timestamp', 'N/A')} · expirarea **{opt_data['expiration']}** "
                           f"({opt_data['days']} zile; aleasă din 20–45, cea mai apropiată de 30) · preț {ro(spot_opt):.2f} · {r_txt} · {q_txt}")
                if opt_data['missing']:
                    st.caption("Lipsesc: " + "; ".join(opt_data['missing']) + ".")

                # --- 1. CARD GEX (estimare cu gamma Black-Scholes) ---
                gex_val = opt_data['net_gex']
                gex_txt = f"{ro(gex_val):,.0f} USD / 1%" if gex_val is not None else "N/A"
                gex_head, _, gex_body = opt_data['gex_verdict'].partition(': ')
                st.markdown(compact_html(f"""
                    <div style="background:{opt_data['gex_color']}22; padding:25px; border-radius:15px; border-left: 10px solid {opt_data['gex_color']}; margin-bottom:25px; border: 1px solid {opt_data['gex_color']}44;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <p style="margin:0; color:#8B949E; text-transform:uppercase; font-size:12px; letter-spacing:1px; font-weight:bold;">Expunere netă la gamma (GEX, estimare) · {opt_data['gex_n']} contracte</p>
                                <h1 style="margin:10px 0; color:{opt_data['gex_color']}; font-size:36px;">{gex_txt}</h1>
                            </div>
                            <div style="text-align:right;">
                                <span style="background:{opt_data['gex_color']}; color:var(--tk-on-fill); padding:8px 16px; border-radius:25px; font-weight:bold; font-size:13px;">
                                    {gex_head}
                                </span>
                            </div>
                        </div>
                        <p style="margin:10px 0 0 0; font-size:15px; color:#C9D1D9;">{gex_body}</p>
                    </div>
                """), unsafe_allow_html=True)
                st.caption("GEX = Σ gamma Black-Scholes × open interest × 100 × preț² × 1%, call cu +, put cu −, pe expirarea aleasă. "
                           "Arată cu cât s-ar schimba delta totală a dealerilor (în USD) la o mișcare de 1% a prețului. "
                           "Semnul presupune că dealerii sunt long call și short put: e o convenție, nu poziția lor reală.")

                # --- RÂNDUL 1: VITEZOMETRU ȘI METRICE ---
                col_gau, col_met = st.columns([1.5, 2])

                with col_gau:
                    iv_display_color = opt_data.get('iv_color', '#8B949E')
                    if opt_data['iv'] is not None:
                        fig_iv = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = opt_data['iv'],
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': f"IV la bani · {opt_data['days']} zile", 'font': {'size': 18, 'color': '#8B949E'}},
                            number = {
                                'suffix': "%",
                                'font': {'color': iv_display_color, 'size': 50}
                            },
                            gauge = {
                                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                                'bar': {'color': iv_display_color},
                                'bgcolor': "rgba(0,0,0,0)",
                                'borderwidth': 2,
                                'bordercolor': "#30363D",
                                'steps': [
                                    {'range': [0, 25], 'color': 'rgba(63, 185, 80, 0.1)'},
                                    {'range': [25, 50], 'color': 'rgba(210, 153, 34, 0.1)'},
                                    {'range': [50, 100], 'color': 'rgba(248, 81, 73, 0.1)'}
                                ]
                            }
                        ))
                        fig_iv.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)')
                        plot_ro(fig_iv, width='stretch')
                        st.caption(f"IV la bani = media IV call și put interpolate la preț ({opt_data['iv_sides']} din 2 părți cu cotații). "
                                   "Culorile folosesc praguri absolute 25% / 50% · netestat.")
                    else:
                        st.metric("IV la bani", "N/A", help="Nu există cotații cu bid > 0 care să încadreze prețul curent pe expirarea aleasă.")

                with col_met:
                    oi_pc, vol_pc = opt_data['oi_pc_ratio'], opt_data['vol_pc_ratio']
                    m1, m2 = st.columns(2)
                    m1.metric("Put/Call (open interest)", _f(oi_pc, ".2f"),
                              help=OPT_HELP["pc_oi"])
                    m2.metric("Put/Call (volum)", _f(vol_pc, ".2f"),
                              help=OPT_HELP["pc_vol"])

                    m3, m4 = st.columns(2)
                    mp = opt_data['max_pain']
                    mp_display = f"{ro(mp):.2f} USD" if mp is not None else "N/A"
                    mp_help = OPT_HELP["max_pain"]
                    if mp is not None:
                        diff_mp = ((mp / spot_opt) - 1) * 100
                        m3.metric("Preț Max Pain", mp_display, f"{ro(diff_mp):.1f}% față de preț", help=mp_help)
                    else:
                        m3.metric("Preț Max Pain", mp_display, help=mp_help)
                    m4.metric("Expirare", opt_data['expiration'], f"{opt_data['days']} zile", delta_color="off",
                              help=OPT_HELP["expirare"])

                # --- IV RANK, PERCENTILE ȘI IV / VOLATILITATE REALIZATĂ ---
                with col_met:
                    st.write("")
                    hv30 = opt_calc.realized_vol(hist['Close']) if hist is not None and not hist.empty else None
                    iv_hv = (opt_data['iv'] / (hv30 * 100)) if (opt_data['iv'] is not None and hv30) else None
                    # IV Rank din istoricul IV la bani (foaia IV_istoric), salvat zilnic în timpul ședinței SUA.
                    iv_store = update_iv_history(real_sym, opt_data['iv'])
                    rk = iv_store["rank"]
                    iv_rank = rk["rank"]
                    iv_days = rk["n"]
                    obs_txt = f"{rk['n']} zile de istoric" + ("" if rk["n"] >= opt_calc.MIN_IV_OBS else f" (minimum {opt_calc.MIN_IV_OBS})")
                    c_rank1, c_rank2, c_rank3 = st.columns(3)
                    c_rank1.metric("IV Rank", _f(iv_rank, ".1f") + ("%" if iv_rank is not None else ""), obs_txt, delta_color="off",
                                   help=OPT_HELP["iv_rank"] + f"\n\n**Minimum:** {opt_calc.MIN_IV_OBS} de zile de istoric.")
                    c_rank2.metric("IV Percentile", _f(rk["percentile"], ".1f") + ("%" if rk["percentile"] is not None else ""),
                                   obs_txt, delta_color="off",
                                   help=OPT_HELP["iv_pct"])
                    c_rank3.metric("IV / volatilitate realizată 30z", _f(iv_hv, ".2f"),
                                   f"realizată {ro(hv30*100):.1f}%" if hv30 else None, delta_color="off",
                                   help=OPT_HELP["iv_hv"])

                st.caption(iv_store["note"])

                # --- VERDICT IV RANK (praguri netestate; activ doar când există istoric IV) ---
                if iv_rank is not None and iv_rank > 80:
                    st.warning("💎 **IV Rank peste 80 (netestat):** opțiunile sunt scumpe față de propriul istoric pe un an; vânzarea de volatilitate (ex. covered calls) e în avantaj față de cumpărare.")
                elif iv_rank is not None and iv_rank < 20:
                    st.success("🛒 **IV Rank sub 20 (netestat):** opțiunile sunt ieftine față de propriul istoric pe un an; cumpărarea de protecție (put) sau de call-uri costă mai puțin.")

                # --- GREEKS LA BANI ȘI MIȘCAREA AȘTEPTATĂ ---
                st.markdown(f"#### 📐 Greeks la bani (strike {_f(opt_data['atm_strike'], '.2f')}, Black-Scholes)")
                gc, gp = opt_data['greeks_call'], opt_data['greeks_put']
                if gc and gp:
                    g1, g2, g3, g4 = st.columns(4)
                    g1.metric("Delta call / put", f"{ro(gc['delta']):.2f} / {ro(gp['delta']):.2f}",
                              help=OPT_HELP["delta"])
                    g2.metric("Gamma", f"{ro(gc['gamma']):.4f}",
                              help=OPT_HELP["gamma"])
                    g3.metric("Theta call / put (pe zi)", f"{ro(gc['theta_day']):.3f} / {ro(gp['theta_day']):.3f}",
                              help=OPT_HELP["theta"])
                    g4.metric("Vega (la 1 pp de IV)", f"{ro(gc['vega_pt']):.3f}",
                              help=OPT_HELP["vega"])
                else:
                    st.info("Greeks N/A: lipsesc IV la bani sau rata fără risc (vezi antetul secțiunii).")
                em = opt_data['expected_move']
                if em is not None:
                    st.metric("Mișcare așteptată până la expirare (±1σ)", f"±{ro(em):.2f} ({ro(em / spot_opt * 100):.1f}%)",
                              help=OPT_HELP["expected_move"])

                # --- RÂNDUL 2: VERDICTUL PUT/CALL (praguri netestate) ---
                iv_val = opt_data['iv']
                if oi_pc is None or vol_pc is None:
                    v_text, v_col = "⚖️ **PUT/CALL INCOMPLET:** lipsește open interest-ul sau volumul la call-uri; fără verdict.", "#8B949E"
                elif oi_pc < 0.7 and vol_pc < 0.7:
                    v_text, v_col = "🚀 **OPTIMISM ÎN OPȚIUNI (netestat):** call-urile domină și în pozițiile deschise, și în volumul de azi.", "#3FB950"
                elif oi_pc < 0.7 and vol_pc > 1.1:
                    v_text, v_col = "🔄 **DIVERGENȚĂ (netestat):** pozițiile deschise sunt optimiste, dar azi volumul e dominat de put-uri.", "#D29922"
                elif oi_pc > 1.1:
                    v_text, v_col = "🐻 **PESIMISM ÎN OPȚIUNI (netestat):** put-urile domină pozițiile deschise (pot fi și acoperiri, nu doar pariuri pe scădere).", "#F85149"
                else:
                    v_text, v_col = "⚖️ **NEUTRU (netestat):** echilibru între put-uri și call-uri.", "#8B949E"

                # caseta e HTML: **text** din Markdown nu se interpretează acolo, deci devine <b>text</b>
                v_html = md_bold_html(v_text)
                st.markdown(f"<div style='background:{v_col}22; padding:20px; border-radius:12px; border-left: 5px solid {v_col}; margin-bottom:20px;'>{v_html}</div>", unsafe_allow_html=True)

                # --- RÂNDUL 3: NIVELUL IV (praguri absolute netestate) ---
                if iv_val is not None and iv_val > 45:
                    st.warning(f"⚠️ **IV ridicat (prag absolut 45%, netestat):** opțiunile la bani sunt scumpe ({ro(iv_val):.1f}%). Pragul absolut nu ține cont de volatilitatea obișnuită a acestei acțiuni.")
                elif iv_val is not None and iv_val < 20:
                    st.success(f"💎 **IV scăzut (prag absolut 20%, netestat):** opțiunile la bani sunt ieftine ({ro(iv_val):.1f}%).")

                # --- RÂNDUL 4: DETALIEREA INDICATORILOR ---
                st.markdown("#### 🔍 Detalierea Indicatorilor")
                def _interp(v, low, high, t_low, t_high, t_mid):
                    if v is None:
                        return "N/A"
                    return (t_low if v < low else t_high if v > high else t_mid) + " (netestat)"
                interpretare_data = [
                    {"Indicator": "📉 Put/Call (OI)", "Valoare": _f(oi_pc, ".2f"), "Interpretare": _interp(oi_pc, 0.7, 1.1, "Optimism", "Pesimism", "Neutru")},
                    {"Indicator": "⚡ Put/Call (Volum)", "Valoare": _f(vol_pc, ".2f"), "Interpretare": _interp(vol_pc, 0.7, 1.1, "Optimism azi", "Presiune put azi", "Normal")},
                    {"Indicator": "🧲 Max Pain Price", "Valoare": mp_display, "Interpretare": f"Reper la expirarea din {opt_data['expiration']}, nu țintă de preț"},
                    {"Indicator": "🌡️ IV la bani", "Valoare": _f(iv_val, ".1f") + ("%" if iv_val is not None else ""), "Interpretare": _interp(iv_val, 20, 40, "Ieftin", "Scump", "Mediu")},
                    {"Indicator": "📏 IV / realizată 30z", "Valoare": _f(iv_hv, ".2f"), "Interpretare": "N/A" if iv_hv is None else ("IV peste mișcarea recentă" if iv_hv > 1 else "IV sub mișcarea recentă")},
                ]
                st.table(interpretare_data)
            else:
                st.info(opt_msg)
                        
            # =================================================================
            # 🏁 VERDICT FINAL: 4 PAȘI (fundamental, moment, criteriile tale, evaluare) + context
            # Pașii nu se adună. Fundamentalul și momentul vin neschimbate din secțiunea de scor de sus;
            # contextul e o listă de alerte fără puncte. Metoda veche (Verdict Final, Master AI pe 10 piloni) ștearsă 13.11.
            # =================================================================
            st.markdown("---")
            st.subheader("🏁 Verdict final")
            from analytics.decision import (characterize, context_flags, count_alerts, final_verdict, my_criteria,
                                            growth as dec_growth, upside as dec_upside, verdict_note,
                                            VERDICT_NOTE_NONE)
            rsi_now = hist["RSI"].iloc[-1] if "RSI" in hist.columns else None
            opt_ok = bool(opt_data) and opt_data.get('oi_pc_ratio') is not None and opt_data.get('iv') is not None
            is_bvb_sym = str(real_sym).upper().endswith(".RO")
            is_fin = (score_decision["financial"] if score_decision
                      else (info.get('sector') or "") == "Financial Services")
            ctx_flags = context_flags(
                is_bvb=is_bvb_sym, is_financial=is_fin, altman_zone=z_val_swot,
                spread=spread, spread_label=spread_label, regime_msg=regime_msg,
                opt_pc=opt_data.get('oi_pc_ratio') if opt_ok else None,
                opt_iv=opt_data.get('iv') if opt_ok else None,
                iv_rank=iv_rank, iv_days=iv_days, min_iv_days=opt_calc.MIN_IV_OBS,
                sentiment=s_score_val if c_news_ai else None, n_news=len(c_news_ai or []),
                inst_pct=inst_percent, q_ratio=q_ratio, rsi=rsi_now, rvol=rvol)
            n_alerts = count_alerts(ctx_flags)

            # --- Pasul 3: criteriile lui Daniel („Hartă în investiții”) — aceleași valori ca în cardurile de sus ---
            pe_v, pbv_v = num(info, 'trailingPE'), num(info, 'priceToBook')
            crit_src = []
            if score_decision is not None and score_decision.get("is_bvb"):
                rp, np_, dr = score_decision.get("rev_pair"), score_decision.get("ni_pair"), score_decision.get("debt_ratio")
                crit_src.append(f"creșteri și îndatorare din fișierul „{COMPANY_FOLDER_NAME}”"
                                + (f" ({rp[2]} → {rp[0]})" if rp else ""))
            else:
                fs_v = get_financial_statements(real_sym)
                rp = fund.annual_pair_statements(fs_v.get("income"), ("Total Revenue", "Operating Revenue"))
                np_ = fund.annual_pair_statements(fs_v.get("income"), ("Net Income", "Net Income Common Stockholders"))
                liab = fund.annual_pair_statements(fs_v.get("balance"), ("Total Liabilities Net Minority Interest", "Total Liabilities"))
                assets = fund.annual_pair_statements(fs_v.get("balance"), ("Total Assets",))
                dr = (liab[1] / assets[1], None, None) if (liab and assets and assets[1] > 0) else None
                crit_src.append("creșteri și îndatorare din situațiile financiare anuale Yahoo"
                                + (f" ({rp[2]:%Y} → {rp[0]:%Y})" if rp else ""))
            cap_ron = None
            cap_v, cap_cur = num(info, 'marketCap'), (info.get('currency') or "")
            if cap_v is not None:
                if cap_cur.upper() == "RON":
                    cap_ron = cap_v
                else:
                    fx_v, _fx_d = rezumat_fx()
                    rate = fx_v.get("GBP") / 100 if cap_cur == "GBp" and fx_v.get("GBP") else fx_v.get(cap_cur.upper())
                    cap_ron = cap_v * rate if rate else None
            crit_rows, crit_ok, crit_n = my_criteria({
                "margin": num(info, 'profitMargins'),
                "rev_growth": dec_growth(rp[1], rp[3]) if rp else None,
                "ni_growth": dec_growth(np_[1], np_[3]) if np_ else None,
                "debt": dr[0] if dr else None,
                "pe": pe_v, "pbv": pbv_v,
                "gn": pe_v * pbv_v if (pe_v is not None and pbv_v is not None and pe_v > 0 and pbv_v > 0) else None,
                "roa": num(info, 'returnOnAssets'), "roe": num(info, 'returnOnEquity'), "cap_ron": cap_ron})

            # --- Pasul 4: evaluarea (DCF de mai sus și prețul țintă al analiștilor, citit neschimbat din Yahoo) ---
            dcf_up = None if (mos_val is None or mos_val >= 100) else 1 / (1 - mos_val / 100) - 1
            an_up = dec_upside(num(info, 'targetMeanPrice'), curr_price)

            v_note = VERDICT_NOTE_NONE   # fără scor (ETF, .RO fără fișier): nota despre testul SUA nu are sens aici
            if score_decision is None:
                v_label, v_level = "FĂRĂ VERDICT", "muted"
                st.info("Fără scor fundamental pentru acest simbol (ETF, fond, indice sau emitent .RO fără fișier în "
                        f"„{COMPANY_FOLDER_NAME}”): verdictul lipsește; criteriile și contextul de mai jos rămân.")
            else:
                d = score_decision
                v_label, v_level = final_verdict(d["fund_label"], d["moment"])
                v_note = verdict_note(d["is_bvb"], d["history_text"])
            v_color = ui_theme.var(v_level) if v_level in ("up", "down", "warn") else ui_theme.var("muted")
            st.markdown(compact_html(f"""
                <div style="background:var(--tk-surface); border:1px solid var(--tk-border); border-left:8px solid {v_color};
                            border-radius:14px; padding:16px 18px; margin:8px 0 12px 0;">
                    <div style="color:var(--tk-muted); font-size:12px; text-transform:uppercase; letter-spacing:1px;">Verdict (descriptiv)</div>
                    <div style="color:{v_color}; font-weight:700; font-size:clamp(22px, 6vw, 30px); margin:6px 0;
                                overflow-wrap:normal; word-break:keep-all;">{html.escape(v_label)}</div>
                    <div style="color:var(--tk-muted); font-size:13px;">{html.escape(v_note)}</div>
                </div>"""), unsafe_allow_html=True)

            pct_sig = lambda v: "N/A" if v is None else f"{ro(v * 100):+.1f}%"
            vs1, vs2, vs3, vs4 = st.columns(4)
            if score_decision is not None:
                vs1.metric("Pasul 1 · Fundamental", f"{d['fund_label']}", d['fund_detail'], delta_color="off", delta_arrow="off",
                           help=f"Scorul companiei din secțiunea de scor de sus, neschimbat. {d['fund_basis']}.")
                vs2.metric("Pasul 2 · Moment", d['moment_label'], d['moment_detail'], delta_color="off", delta_arrow="off",
                           help=BVB_HELP["moment"] + f"\n\n**Aici:** {d['moment_basis']}.")
            else:
                vs1.metric("Pasul 1 · Fundamental", "N/A", help="Fără scor fundamental (ETF, fond, indice sau .RO fără fișier).")
                vs2.metric("Pasul 2 · Moment", "N/A", help="Fără scor fundamental, verdictul nu se calculează; momentul e în secțiunile de mai sus.")
            vs3.metric("Pasul 3 · Criteriile tale", f"{crit_ok} din {crit_n}" if crit_n else "N/A",
                       f"{10 - crit_n} fără date" if crit_n < 10 else None, delta_color="off", delta_arrow="off",
                       help="**Ce e:** câte din criteriile tale minime de cumpărare („Hartă în investiții”) îndeplinește compania.\n\n"
                            "**Calcul:** marjă ≥ 10%, cifra de afaceri și profitul în creștere față de anul precedent, datorii / active "
                            "≤ 70%, P/E ≤ 15, P/BV ≤ 1,5, GN ≤ 22,5, ROA ≥ 5%, ROE ≥ 15%, capitalizare ≥ 1 mld. lei. Criteriile fără date "
                            "nu se numără.\n\n**Îndatorarea:** datorii totale (inclusiv furnizori, provizioane) / active totale = cât din "
                            "activele companiei e finanțat de creditori; rata de îndatorare globală din foaia ta BVB. 70% din active "
                            "≈ levier datorii / capital propriu de 2,33 (rândul „Levier financiar”): levier = rată / (1 − rată). "
                            "Repere uzuale: sub 50% confortabil, 50–70% acceptabil, peste 70% îndatorare mare; băncile și "
                            "utilitățile funcționează normal cu valori mari.\n\n**Atenție:** e regula ta, nu un test istoric; nu schimbă verdictul. Pe BVB, testul a găsit "
                            "putere de predicție la P/E și P/BV, dar nu la ROE, marjă și îndatorare.\n\n**Sursă:** " + "; ".join(crit_src) + ".")
            vs4.metric("Pasul 4 · Evaluare (DCF)", pct_sig(dcf_up),
                       None if an_up is None else f"analiști {pct_sig(an_up)}", delta_color="off", delta_arrow="off",
                       help="**Ce e:** cât e de la prețul de azi până la valoarea DCF calculată mai sus (cu ipotezele tale) și, "
                            "dedesubt, până la prețul țintă mediu al analiștilor (Yahoo, neschimbat).\n\n**Calcul:** valoare / preț − 1.\n\n"
                            "**Praguri orientative (reguli practice din literatură, nu standard oficial și netestate pe datele tale):** "
                            "peste +10% sub valoare · ±10% la valoare · sub −10% peste valoare (banda neutră a DCF-ului).\n\n"
                            "**Atenție:** DCF-ul depinde de ipoteze; ținta analiștilor e opinia lor. Nu schimbă verdictul.")

            if score_decision is not None:
                phrase = characterize(is_bvb=d["is_bvb"], fund_label=d["fund_label"], fund_score=d["fund_score"],
                                      moment_score=d["moment"], pillars=d["pillars"], n_alerts=n_alerts,
                                      history_text=d["history_text"])
                st.markdown(f"<div style='color:var(--tk-text); font-size:14px; line-height:1.5; margin:8px 0;'>"
                            f"{html.escape(phrase)}</div>", unsafe_allow_html=True)
                if d["pillars"]:
                    p_cols = st.columns(len(d["pillars"]))
                    for col, (p_name, p_score) in zip(p_cols, d["pillars"]):
                        col.metric(p_name, "N/A" if p_score is None else f"{ro(p_score):.0f}/100",
                                   help="Pilonul scorului fundamental din secțiunea „Scor fundamental” de sus; aceeași valoare.")

            with st.expander(f"Criteriile tale: {crit_ok} din {crit_n} îndeplinite" if crit_n else "Criteriile tale"):
                st.dataframe(pd.DataFrame([{"Criteriu": r["name"], "Prag": r["threshold"], "Valoare": r["value"],
                                            "Îndeplinit": {True: "✅", False: "❌", None: "➖ fără date"}[r["ok"]]}
                                           for r in crit_rows]), hide_index=True, width='stretch')

            st.markdown("#### 🧭 Context (alerte fără puncte)")
            flag_style = {"alerta": ("rgba(248, 81, 73, 0.15)", "#F85149", "🚨"),
                          "atentie": ("rgba(210, 153, 34, 0.1)", "#D29922", "⚠️"),
                          "pozitiv": ("rgba(63, 185, 80, 0.05)", "#3FB950", "✅"),
                          "info": ("#21262D", "#8B949E", "ℹ️")}
            for f in ctx_flags:
                bg, border, icon = flag_style[f["level"]]
                st.markdown(f"<div style='background:{bg}; padding:10px 12px; border-radius:8px; margin-bottom:6px; "
                            f"border-left:4px solid {border};'>{icon} <b>{html.escape(f['title'])}:</b> {html.escape(f['text'])}</div>",
                            unsafe_allow_html=True)
            st.caption("Pașii nu se adună într-un scor total. Contextul nu modifică scorurile: opțiunile și știrile nu au istoric "
                       "pentru un test, iar curba randamentelor e un semnal al pieței, nu al companiei.")

            # ==================================================
            # MODUL NOU: PREZENTARE GENERALĂ AI (CLONA XTB PERPLEXITY)
            # ==================================================
            st.markdown("---")
            st.subheader("✨ Prezentare generală cu inteligența artificială")
            
            with st.spinner("Motorul Quant-AI sintetizează fundamentele și media..."):
                bullets = []
                
                # Extragem datele vitale la cald
                # Lipsă = None: bulletul respectiv nu se afișează (nu se presupune 0 sau beta 1).
                pe_ratio = num(info, 'trailingPE')
                pb_ratio = num(info, 'priceToBook')
                margins = num(info, 'profitMargins')
                margins = margins * 100 if margins is not None else None
                beta = beta_val
                pe_src = ("foaia BVB, la prețul curent" if 'trailingPE' in (info.get('_bvb_repriced') or [])
                          else "Yahoo")
                
                # --- BULLET 1: Profitabilitate & Venituri ---
                if margins is not None and margins > 0:
                    if margins > 15:
                        text_fin = f"Compania raportează rezultate fundamentale puternice, cu o marjă netă de profit de {ro(margins):.2f}%. Această eficiență operațională ridicată consolidează încrederea investitorilor pe termen lung."
                        bullets.append({"icon": "↗️", "color": "#3FB950", "text": text_fin})
                    else:
                        text_fin = f"Compania menține o profitabilitate moderată, operând cu o marjă netă de {ro(margins):.2f}%, o valoare standard pentru dinamica actuală a sectorului său."
                        bullets.append({"icon": "➡️", "color": "#8B949E", "text": text_fin})
                elif margins is not None and margins < 0:
                    text_fin = f"Rezultatele recente indică presiuni operaționale, compania raportând pierderi cu o marjă netă negativă de {ro(margins):.2f}%, un factor de risc major pentru acționari."
                    bullets.append({"icon": "↘️", "color": "#F85149", "text": text_fin})

                # --- BULLET 2: Sentimentul Știrilor (Context) ---
                c_news_ai = get_company_news_rss(real_sym, names=(info.get('longName'), info.get('shortName')))
                if c_news_ai:
                    top_news = html.escape(str(c_news_ai[0]['title']))   # text extern în HTML
                    from ai_engine import analyze_sentiment_ai
                    s_score = analyze_sentiment_ai(c_news_ai)
                    
                    if s_score is None:
                        bullets.append({"icon": "➡️", "color": "#8B949E",
                                        "text": "Sentimentul știrilor nu a putut fi calculat acum (FinBERT indisponibil)."})
                    elif s_score > 0.15:
                        text_news = f"Sentimentul media general este optimist. Atenția presei este captată de evoluții pozitive, un catalizator recent fiind: «{top_news}»."
                        bullets.append({"icon": "↗️", "color": "#3FB950", "text": text_news})
                    elif s_score < -0.15:
                        text_news = f"Narațiunea mediatică este dominată de îngrijorări sau reglementări stricte. Titlurile recente reflectă pesimism, precum: «{top_news}»."
                        bullets.append({"icon": "↘️", "color": "#F85149", "text": text_news})
                    else:
                        text_news = f"Acoperirea în presă este neutră în știrile recente, fără șocuri reputaționale majore. Subiectul momentului: «{top_news}»."
                        bullets.append({"icon": "➡️", "color": "#8B949E", "text": text_news})

                # --- BULLET 3: Volatilitate / Risc (Analiză Beta) ---
                if beta is not None and beta > 1.3:
                    text_vol = f"Prețul acțiunilor a fost volatil comparativ cu piața de referință (Beta de {ro(beta):.2f}). Această fluctuație amplă atrage speculatorii, dar poate îngrijora investitorii conservatori."
                    bullets.append({"icon": "↘️", "color": "#F85149", "text": text_vol})
                elif beta is not None and beta < 0.8:
                    text_vol = f"Acțiunea prezintă o volatilitate redusă față de piața generală (Beta de {ro(beta):.2f}), comportându-se ca un activ defensiv în perioadele de incertitudine."
                    bullets.append({"icon": "↗️", "color": "#3FB950", "text": text_vol})

                # --- BULLET 4: Evaluare Multipli ---
                pb_txt = f" și P/B de {ro(pb_ratio):.2f}" if pb_ratio is not None else ""
                if pe_ratio is not None and pe_ratio > 30:
                    text_val = (f"Multipli de evaluare ridicați, inclusiv un raport P/E de {ro(pe_ratio):.2f}{pb_txt} ({pe_src}), sugerează că acțiunea "
                                f"ar putea fi supraevaluată în raport cu fundamentele contabile curente (prag fix P/E 30, netestat).")
                    bullets.append({"icon": "↘️", "color": "#F85149", "text": text_val})
                elif pe_ratio is not None and 0 < pe_ratio < 15:
                    text_val = (f"Acțiunea se tranzacționează la multipli de evaluare atractivi (P/E de {ro(pe_ratio):.2f}, {pe_src}), indicând o potențială "
                                f"subevaluare din partea pieței raportat la capacitatea sa de generare a profitului (prag fix P/E 15, netestat).")
                    bullets.append({"icon": "↗️", "color": "#3FB950", "text": text_val})

                # --- RANDARE UI PREMIUM ---
                if bullets:
                    render_bullet_box(bullets, "Informațiile de mai sus au fost generate algoritmic. "
                                      "Nu le trata ca recomandare sau sfat de investiții.")
                else:
                    st.info("Date insuficiente pentru generarea sintezei AI.")
            st.markdown("---")    
            
            # 7. Ultimele Știri (LOGICĂ REPARATĂ)
            st.subheader(f"📰 Ultimele Știri despre {real_sym}")
            if c_news_ai:
                from ai_engine import get_gemini_sentiment_label # Importăm o singură dată
                
                for n in c_news_ai:
                    # Obținem eticheta de la Gemini
                    # Folosim 'with st.spinner' doar dacă vrei să vezi că lucrează
                    label_txt, css_cls, icon = get_gemini_sentiment_label(n['title'])
                    
                    c_t, c_i = st.columns([5, 1.2])
                    with c_t:
                        st.markdown(f"**[{n['title']}]({n['link']})**")
                        st.caption(f"{n['publisher']} • {n['date_str']}")
                    with c_i:
                        # Folosim label_txt, variabila definită mai sus
                        st.markdown(f'<span class="{css_cls}">{icon} {label_txt}</span>', unsafe_allow_html=True)
                    st.divider()
        _an_timer.caption(f"⏱️ Pagina a rulat complet în {format_ro(time.perf_counter() - _an_t0, 1)} s, la {now_ro():%H:%M:%S}.")
                                
    # ==================================================
    # 3. PORTOFOLIU (MODIFICAT PENTRU MOBIL)
    # ==================================================
    elif sectiune == "3. Portofoliu":
        st.title("💼 Portofoliu Personal")
        # Timpul rulării complete (Etapa 5): se scrie la final, ca viteza paginii să se poată măsura.
        _pf_t0 = time.perf_counter()
        _pf_timer = st.empty()
        
        # Formularul „Adaugă Tranzacție Nouă” (scria în foaia Portofoliu) a fost scos la 04.11.2026, cu acordul
        # lui Daniel: tranzacțiile intră doar prin importul exporturilor de la broker (deduplicare pe ID).

        # Toată pagina folosește pozițiile din registrul `Tranzactii` (exporturile brokerilor);
        # foaia Portofoliu nu mai alimentează pagina.
        ledger_positions = render_ledger_results()
        render_sec_portfolio_signals()
        df_pf = ledger_calc.tab_frame(ledger_positions)
        st.markdown("---")

        if df_pf.empty:
            st.info("Nu există acțiuni sau ETF-uri deschise în registrul `Tranzactii` (sau registrul nu a putut fi citit).")
        else:
            st.markdown("### Perioadă Analiză")
            hist_range = st.select_slider("Perioadă analiză", label_visibility="collapsed", options=["1Z", "1S", "1L", "3L", "6L", "1A", "3A", "5A", "Max"], value="1A", key="range_slider",
                                          help="Fereastra pentru grafice și indicatori. Max = tot istoricul disponibil: pentru performanța reală (TWR, alpha, risc), de la prima deținere din registrul `Tranzactii` în moneda tabului; pentru graficul valorii, Monte Carlo și optimizator, istoricul de preț descărcat (cel mult 5 ani), limitat de poziția cu istoricul cel mai scurt.")
            
            tab_usd, tab_eur, tab_ron = st.tabs(["🇺🇸 Portofoliu USD", "🇪🇺 Portofoliu EUR", "🇷🇴 Portofoliu BVB (RON)"])
            # Registrul, pentru performanța reală pe monedă (citirea foii e în cache)
            _ledger_vals = load_ledger_values()
            ledger_df_tabs = ledger_calc.parse_ledger(_ledger_vals)[0] if _ledger_vals is not None else None
            
            def render_portfolio_tab(df_subset, currency_symbol):
                if df_subset.empty:
                    st.info(f"Nu ai poziții deschise în {currency_symbol}.")
                    return

                with st.spinner(f"Calculăm performanța pentru {currency_symbol}..."):
                    df_calc, hist_curve, daily_abs, daily_pct, pf_notes = calculate_portfolio_performance(df_subset, hist_range)

                # Pozițiile fără preț nu intră în valoarea curentă și în profit (altfel ar apărea ca pierdere de 100%).
                priced = df_calc.dropna(subset=['MarketValue']) if not df_calc.empty else df_calc
                total_invested = (df_calc['Quantity'] * df_calc['AvgPrice']).sum() if not df_calc.empty else 0
                invested_priced = (priced['Quantity'] * priced['AvgPrice']).sum() if not priced.empty else 0
                total_current = priced['MarketValue'].sum() if not priced.empty else 0
                
                total_profit_val = total_current - invested_priced
                total_profit_pct = (total_profit_val / invested_priced * 100) if invested_priced != 0 else 0

                if pf_notes.get('missing_price'):
                    st.warning(f"⚠️ Fără preț disponibil pentru: **{', '.join(pf_notes['missing_price'])}**. "
                               "Aceste poziții nu sunt incluse în valoarea curentă și în profit.")
                if pf_notes.get('price_from_close'):
                    st.caption(f"ℹ️ Preț live indisponibil pentru {', '.join(pf_notes['price_from_close'])}: s-a folosit ultima închidere.")

                c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
                c_kpi1.metric(f"Total Investit ({currency_symbol})", f"{format_ro(total_invested)} {currency_symbol}",
                              help="Σ cantitate × preț mediu de cost, pe toate pozițiile deschise din tab (din registrul "
                                   "`Tranzactii`), inclusiv cele fără preț curent.")
                c_kpi2.metric(f"Valoare Curentă ({currency_symbol})", f"{format_ro(total_current)} {currency_symbol}",
                              help="Σ cantitate × prețul de acum (live, sau ultima închidere dacă live lipsește). "
                                   "Pozițiile fără niciun preț nu intră (sunt listate în avertismentul de deasupra).")
                c_kpi3.metric(f"Profit/Pierdere ({currency_symbol})", f"{format_ro(total_profit_val)} {currency_symbol}", f"{format_ro(total_profit_pct)}%",
                              help="Nerealizat: valoarea curentă − costul pozițiilor cu preț; procentul = profit ÷ acel cost. "
                                   "Fără dividende încasate și fără vânzările închise (acelea sunt în performanța din registru, mai sus).")

                if pf_notes.get('curve_start') is not None:
                    st.caption(
                        f"ℹ️ Graficul valorii, Monte Carlo și optimizatorul aplică deținerile de azi pe trecut, începând cu "
                        f"{pf_notes['curve_start'].strftime('%d.%m.%Y')} (prima zi în care toate pozițiile au preț; "
                        f"istoricul cel mai scurt: {pf_notes.get('limiting_symbol')}). Nu reprezintă performanța realizată efectiv."
                    )
                if pf_notes.get('no_history'):
                    st.caption(f"ℹ️ Fără istoric de preț, deci neincluse în grafice: {', '.join(pf_notes['no_history'])}.")

                # --- INTEGRARE MONTE CARLO ---
                st.markdown("---")
                st.subheader("🔮 Proiecție Probabilistă Monte Carlo (1 An)")

                # Bootstrap pe randamentele zilnice din fereastra aleasă (analytics/simulation.py); seed fix, deci stabil la rerun.
                paths = simulation.bootstrap_paths(hist_curve)
                if paths is None:
                    st.info("Simularea are nevoie de cel puțin 60 de zile de istoric: alege o perioadă de analiză mai lungă (minimum 6L).")
                else:
                    mc = simulation.summarize_paths(paths)
                    mc_flat = simulation.summarize_paths(simulation.bootstrap_paths(hist_curve, demean=True))
                    last_val, bands = mc["start"], mc["bands"]
                    x_range = list(range(len(bands)))

                    fig_mc = go.Figure()
                    # Primele 100 de traiectorii, ca fundal (toate ar supraîncărca browserul)
                    for i in range(min(100, paths.shape[1])):
                        fig_mc.add_trace(go.Scatter(x=x_range, y=paths[:, i], mode='lines', hoverinfo='skip',
                                                line=dict(width=0.5, color='rgba(139, 148, 158, 0.25)'), showlegend=False))
                    # Percentilele pe tot orizontul, nu doar la final
                    fig_mc.add_trace(go.Scatter(x=x_range, y=bands["p95"], line=dict(color='#199E70', width=2), name='Optimist (95%)',
                                                hovertemplate="%{y:,.0f}<extra>Optimist</extra>"))
                    fig_mc.add_trace(go.Scatter(x=x_range, y=bands["p50"], line=dict(color='#3987E5', width=2), name='Median (50%)',
                                                hovertemplate="%{y:,.0f}<extra>Median</extra>"))
                    fig_mc.add_trace(go.Scatter(x=x_range, y=bands["p5"], line=dict(color='#D95926', width=2), name='Pesimist (5%)',
                                                hovertemplate="%{y:,.0f}<extra>Pesimist</extra>"))
                    fig_mc.update_layout(height=400, template="plotly_dark",
                                        hovermode="x unified", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                        xaxis_title="Zile de tranzacționare de azi înainte",
                                        yaxis_title=f"Valoare Portofoliu ({currency_symbol})",
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    plot_ro(fig_mc, width='stretch')

                    c_mc1, c_mc2, c_mc3 = st.columns(3)
                    c_mc1.metric("Valoare în cel mai nefavorabil caz", f"{format_ro(mc['p5'])} {currency_symbol}",
                                help=PORTF_HELP["mc_p5"])
                    c_mc2.metric("Variația în scenariul pesimist", f"{ro(mc['change_p5_pct']):+.2f}%",
                                help=PORTF_HELP["mc_p5_pct"])
                    c_mc3.metric("Probabilitatea de profit", f"{ro(mc['prob_profit']):.1f}%",
                                help=PORTF_HELP["mc_prob"])
                    st.caption(
                        f"Din {mc['simulations']} scenarii pe {mc['days']} de zile, construite prin reeșantionarea randamentelor zilnice din perioada aleasă mai sus "
                        f"({len(hist_curve) - 1} zile). Simularea presupune că randamentul mediu din această perioadă continuă. "
                        f"Fără această presupunere (doar volatilitatea, tendință zero): scenariu pesimist {ro(mc_flat['change_p5_pct']):+.1f}%, "
                        f"median {ro(mc_flat['change_p50_pct']):+.1f}%, probabilitate de profit {ro(mc_flat['prob_profit']):.0f}%. "
                        "Curba folosită aplică deținerile de azi pe trecut; nu e o prognoză."
                    )
                
                if not hist_curve.empty:
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Scatter(
                        x=hist_curve.index, y=hist_curve.values, 
                        fill='tozeroy', line=dict(color='#238636'), name=f'Valoare {currency_symbol}'
                    ))
                    fig_hist.update_layout(height=350, template="plotly_dark", margin=dict(t=10, b=10), paper_bgcolor='rgba(0,0,0,0)')
                    plot_ro(fig_hist, width='stretch')
                    # --- NOU: COMPARAȚIE BENCHMARK ---
                st.markdown("---")
                st.subheader("🏁 Performanță Relativă (Benchmark)")
                cur_code = {"$": "USD", "€": "EUR"}.get(currency_symbol, currency_symbol)
                bench_ticker, bench_name = TAB_BENCHMARKS[cur_code]
                # Performanța și riscul se măsoară pe ce s-a întâmplat efectiv: acțiunile și ETF-urile din
                # această monedă, din registrul `Tranzactii`, cu TWR (fără efectul cumpărărilor și vânzărilor).
                sleeve = None
                if ledger_df_tabs is not None and not ledger_df_tabs.empty:
                    sleeve_syms = sorted(set(ledger_df_tabs.loc[
                        (ledger_df_tabs["Currency"] == cur_code) & ledger_df_tabs["Symbol"].ne("")
                        & ~ledger_df_tabs["Class"].isin(ledger_calc.NOMINAL_CLASSES), "Symbol"]))
                    sleeve_start = (ledger_df_tabs["Date"].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
                    sleeve_index = pd.bdate_range(ledger_df_tabs["Date"].min().normalize(), pd.Timestamp(now_ro().replace(tzinfo=None)).normalize())
                    with st.spinner(f"Reconstruim istoricul real al pozițiilor în {cur_code}..."):
                        sleeve_market = get_ledger_market_history(tuple(sleeve_syms), sleeve_start)
                    s_values, s_in, s_out, s_sources, s_blockers = ledger_calc.currency_sleeve(
                        ledger_df_tabs, cur_code, sleeve_index, sleeve_market)
                    s_daily, s_cum, s_total = ledger_calc.twr_in_out(s_values, s_in, s_out)
                    if s_total is not None:
                        sleeve = {"wealth": slice_window(1.0 + s_cum, hist_range), "sources": s_sources, "blockers": s_blockers}
                if sleeve is None or len(sleeve["wealth"]) < 2:
                    st.info(f"Istoricul real al pozițiilor în {cur_code} nu poate fi reconstruit din registru "
                            "(lipsesc prețurile istorice sau tranzacțiile); indicatorii de mai jos sunt N/A.")
                    wealth_w = pd.Series(dtype=float)
                else:
                    wealth_w = sleeve["wealth"]
                    with st.spinner(f"Se compară cu {bench_name}..."):
                        render_benchmark_comparison(wealth_w, bench_ticker, bench_name)
                    proxy = sorted(s for s, src in sleeve["sources"].items() if src in ("proxy", "mixed"))
                    st.caption(
                        f"ℹ️ Curba de sus e randamentul ponderat cu timpul (TWR) al acțiunilor și ETF-urilor în {cur_code} din registrul "
                        f"`Tranzactii`, {wealth_w.index[0]:%d.%m.%Y}–{wealth_w.index[-1]:%d.%m.%Y}: cumpărările și vânzările nu "
                        "contează ca randament, dividendele încasate (nete de impozit) da. Fără numerar și fără titluri de stat."
                        + (f" Fără istoric complet de piață, evaluate la prețul ultimei tranzacții (proxy): {', '.join(proxy)}." if proxy else "")
                        + (f" Zile fără preț din cauza: {', '.join(sleeve['blockers'])}; TWR începe după ultima." if sleeve["blockers"] else ""))

                rets_w = wealth_w.pct_change(fill_method=None).dropna() if len(wealth_w) > 1 else pd.Series(dtype=float)
                rf_tab, rf_tab_label = get_risk_free_for_currency(cur_code)
                risk_tab = performance_risk(rets_w, rf_tab) if len(rets_w) else None
                bench_close_tab = get_benchmark_close(bench_ticker)
                bc = beta_corr_weekly(wealth_w, bench_close_tab) if len(wealth_w) > 1 else None

                st.markdown(f"#### 🛡️ Diagnostic Portofoliu ({currency_symbol})")
                if risk_tab is None:
                    st.info(f"Prea puține zile cu randament în fereastra aleasă ({len(rets_w)}); indicatorii de risc cer cel puțin 60. "
                            "Alege o perioadă mai lungă (de ex. 1A).")
                pct_t = lambda v, d=2: "N/A" if v is None else f"{ro(v * 100):.{d}f}%"
                rat_t = lambda v: "N/A" if v is None else f"{ro(v):.2f}"
                amt_t = lambda v: "N/A" if (v is None or not total_current) else f"{currency_symbol} {format_ro(abs(v * total_current))}"
                rk = risk_tab or {}
                rf_txt = (f"Rata fără risc: {ro(rf_tab * 100):.2f}% [{rf_tab_label}]." if rf_tab is not None
                          else f"Rata fără risc indisponibilă ({rf_tab_label}): Sharpe și Sortino sunt N/A.")
                n_txt = f" Eșantion: {rk['n']} zile ({rk['start']:%d.%m.%Y}–{rk['end']:%d.%m.%Y})." if rk else ""

                c_r1, c_r2, c_r3, c_r4 = st.columns(4)
                c_r1.metric("Max Drawdown", pct_t(rk.get("max_drawdown")),
                            help=PORTF_HELP["max_dd"] + "\n\n" + n_txt.strip())
                c_r2.metric("Sharpe Ratio", rat_t(rk.get("sharpe")),
                            help=PORTF_HELP["sharpe"] + "\n\n**Date:** " + rf_txt + n_txt)
                c_r3.metric("VaR istoric (95%, 1 zi)", amt_t(rk.get("var95_hist")),
                            help=PORTF_HELP["var_hist"] + f"\n\n**Procent:** {pct_t(rk.get('var95_hist'))}." + n_txt)
                c_r4.metric("Volatilitate Anualizată", pct_t(rk.get("vol_ann")),
                            help=PORTF_HELP["vol"] + "\n\n" + n_txt.strip())

                st.write("")
                c_b1, c_b2, c_b3, c_b4 = st.columns(4)
                c_b1.metric("Benchmark", bench_name, help="Indicele pieței în aceeași monedă cu pozițiile.")
                bc_txt = (f" {bc['n']} randamente săptămânale comune, {bc['start']:%d.%m.%Y}–{bc['end']:%d.%m.%Y}." if bc
                          else " Sub 52 de săptămâni comune: N/A.")
                corr_v = bc["corr"] if bc else None
                c_b2.metric("Corelație cu Piața", rat_t(corr_v),
                            help=PORTF_HELP["corr"] + f"\n\n**Indice:** {bench_name}." + bc_txt)
                c_b3.metric("Beta Portofoliu", rat_t(bc["beta"] if bc else None),
                            help=PORTF_HELP["beta"] + f"\n\n**Indice:** {bench_name}." + bc_txt)
                c_b4.metric("Scor Diversificare", "N/A" if corr_v is None else f"{ro((1 - abs(corr_v)) * 100):.1f}%",
                            help=PORTF_HELP["divers"])

                st.write("")
                c_s1, c_s2, c_s3 = st.columns(3)
                sortino_val = rk.get("sortino")
                c_s1.metric("Sortino Ratio", rat_t(sortino_val),
                            help=PORTF_HELP["sortino"] + "\n\n**Date:** " + rf_txt + n_txt)
                c_s2.metric("CVaR (95%, 1 zi)", amt_t(rk.get("cvar95")),
                            help=PORTF_HELP["cvar"] + f"\n\n**Procent:** {pct_t(rk.get('cvar95'))}." + n_txt)
                c_s3.metric("VaR parametric (95%)", amt_t(rk.get("var95_param")),
                            help=PORTF_HELP["var_param"] + f"\n\n**Procent:** {pct_t(rk.get('var95_param'))}." + n_txt)
                if sortino_val is not None:
                    if sortino_val > 2:
                        st.success("🌟 **Sortino peste 2:** randament mare față de scăderile din perioadă (prag orientativ, netestat).")
                    elif sortino_val > 1:
                        st.info("⚖️ **Sortino între 1 și 2:** randamentul acoperă scăderile (prag orientativ, netestat).")
                    else:
                        st.warning("⚠️ **Sortino sub 1:** scăderile au fost mari față de randament (prag orientativ, netestat).")

                st.markdown("---")

                # 2. Grafice Plăcintă: Simboluri vs Sectoare
                st.subheader("🍰 Distribuția Activelor")
                col_pie1, col_pie2 = st.columns(2)
                
                with col_pie1:
                    st.caption("**După Companie (Simbol)**")
                    if not priced.empty:
                        fig_sym = go.Figure(data=[go.Pie(
                            labels=priced['Symbol'], 
                            values=priced['MarketValue'], 
                            hole=.4,
                            textinfo='percent',
                            hovertemplate="<b>%{label}</b><br>Valoare: %{value:,.2f} " + currency_symbol + "<br>Pondere: %{percent}<extra></extra>"
                        )])
                        fig_sym.update_layout(height=350, margin=dict(t=0, b=0, l=0, r=0), 
                                              template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                        plot_ro(fig_sym, width='stretch', key=f"pie_sym_{currency_symbol}")

                # 4. Detaliu Poziții
                st.subheader("📋 Detaliu Poziții")
                if not df_calc.empty:
                    df_detail = positions_detail(df_calc)

                    def color_profit(val):
                        if pd.isna(val): return ''
                        color = DF_UP if val >= 0 else DF_DOWN
                        return f'color: {color}'

                    st.dataframe(
                        df_detail.style.map(color_profit, subset=['Profit', 'Profit %'])
                        .format({
                            # format românesc; cantitatea până la 4 zecimale, fără zerouri inutile (1.250, 0,3048)
                            'Cantitate': lambda q: format_ro(q, 4, trim=True),
                            'Cost mediu': lambda v: format_ro(v, 4), 'Preț curent': lambda v: format_ro(v, 4),
                            'Valoare': lambda v: format_ro(v), 'Pondere %': lambda v: format_ro(v, 2, "%"),
                            'Profit': lambda v: format_ro(v), 'Profit %': lambda v: format_ro(v, 2, "%")
                        }, na_rep="N/A"),
                        width='stretch', hide_index=True,
                        column_config={
                            "Simbol": st.column_config.Column(help="Simbolul Yahoo al poziției deschise din registrul `Tranzactii`."),
                            "Cantitate": st.column_config.Column(help="Cantitatea deținută azi: cumpărări − vânzări din registru, plus acțiunile gratuite."),
                            "Cost mediu": st.column_config.Column(help=f"Cost mediu ponderat pe unitate, în {currency_symbol}: costul deschis (cu comisioanele de cumpărare) ÷ cantitate. "
                                                                         "Acțiunile gratuite intră la cost 0, deci îl scad; la o vânzare parțială costul mediu nu se schimbă."),
                            "Preț curent": st.column_config.Column(help="Prețul live de la Yahoo; dacă lipsește, ultima închidere (vezi nota de deasupra). Fără niciunul: N/A."),
                            "Valoare": st.column_config.Column(help=f"Cantitate × preț curent, în {currency_symbol}. N/A dacă lipsește prețul."),
                            "Pondere %": st.column_config.Column(help="Valoarea poziției ÷ valoarea tuturor pozițiilor cu preț din acest tab × 100. "
                                                                     "Pozițiile fără preț nu intră la numitor."),
                            "Profit": st.column_config.Column(help=f"Valoare − cantitate × cost mediu, în {currency_symbol}. Profitul pe hârtie, "
                                                                  "fără dividende, impozite și profitul deja realizat la vânzări (acelea sunt în tabelul pe clase de sus)."),
                            "Profit %": st.column_config.Column(help="Profit ÷ (cantitate × cost mediu) × 100. N/A la cost 0 (doar acțiuni gratuite)."),
                        }
                    )

                with col_pie2:
                    st.caption("**După Sector Economic (%)**")
                    with st.spinner("Analizăm expunerea..."):
                        df_sectors, sec_unpriced, sec_sources = get_portfolio_sectors(df_calc)

                    if not df_sectors.empty:
                        # Culorile urmează sectorul în ordinea ponderii (paleta categorică, temă închisă); ETF/fond și
                        # „Nedefinit” în gri, ca să nu pară sectoare. Peste 7 sectoare reale, restul intră la „Altele”.
                        cat_colors = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9']
                        neutral = {pf_calc.SECTOR_FUND: '#8b949e', pf_calc.SECTOR_UNKNOWN: '#484f58'}
                        real = df_sectors[~df_sectors['Sector'].isin(list(neutral))]
                        pie_df = pd.concat([real.head(7), df_sectors[df_sectors['Sector'].isin(list(neutral))]])
                        rest = real.iloc[7:]
                        if not rest.empty:
                            pie_df = pd.concat([pie_df, pd.DataFrame([{"Sector": "Altele", "Valoare": rest['Valoare'].sum(),
                                                                        "Pondere %": rest['Pondere %'].sum(),
                                                                        "Simboluri": ", ".join(rest['Simboluri'])}])])
                        colors, k = [], 0
                        for sec_name in pie_df['Sector']:
                            if sec_name in neutral:
                                colors.append(neutral[sec_name])
                            elif sec_name == "Altele":
                                colors.append('#6e7681')
                            else:
                                colors.append(cat_colors[k]); k += 1
                        fig_sec = go.Figure(data=[go.Pie(
                            labels=pie_df['Sector'],
                            values=pie_df['Valoare'],
                            customdata=pie_df['Simboluri'],
                            hole=.4, sort=False,
                            textinfo='label+percent',
                            marker=dict(colors=colors, line=dict(color='#0e1117', width=2)),
                            hovertemplate="<b>%{label}</b><br>Valoare: %{value:,.2f} " + currency_symbol
                                          + "<br>Pondere: %{percent}<br>%{customdata}<extra></extra>"
                        )])
                        fig_sec.update_layout(height=350, margin=dict(t=0, b=0, l=0, r=0), showlegend=True,
                                              template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                        plot_ro(fig_sec, width='stretch', key=f"pie_sec_{currency_symbol}")

                        # VERDICT DIVERSIFICARE (pe sectoarele reale; ETF/fond și „Nedefinit” nu sunt sectoare)
                        sv = pf_calc.sector_verdict(df_sectors)
                        lim = f"{ro(pf_calc.SECTOR_CONCENTRATION_PCT):.0f}%"
                        if sv['status'] == 'na':
                            st.info("ℹ️ Nicio poziție din acest tab nu are un sector cunoscut (doar ETF-uri/fonduri sau sector lipsă), "
                                    "deci concentrarea pe sectoare nu se poate evalua.")
                        elif sv['status'] == 'concentrat':
                            st.warning(f"⚠️ **Concentrare mare:** sectorul „{sv['top']['Sector']}” ({sv['top']['Simboluri']}) are "
                                       f"{ro(sv['top']['Pondere %']):.1f}% din valoarea tabului, peste pragul de {lim} (prag orientativ, netestat).")
                        else:
                            st.success(f"✅ Niciun sector nu depășește {lim} din valoarea tabului; cel mai mare: „{sv['top']['Sector']}”, "
                                       f"{ro(sv['top']['Pondere %']):.1f}% (prag orientativ, netestat).")
                        notes_sec = []
                        if sv['fund_pct'] > 0:
                            notes_sec.append(f"{ro(sv['fund_pct']):.1f}% e în ETF-uri/fonduri: sunt diversificate în interior, "
                                             "dar compoziția lor pe sectoare nu e citită, deci nu intră în evaluare")
                        if sv['unknown_pct'] > 0:
                            notes_sec.append(f"{ro(sv['unknown_pct']):.1f}% are sector necunoscut (Yahoo nu l-a trimis sau citirea a eșuat) "
                                             "și nu intră în evaluare")
                        if sec_unpriced:
                            notes_sec.append(f"fără preț, deci în afara graficului: {', '.join(sec_unpriced)}")
                        bvb_src = sorted(s_ for s_, src in sec_sources.items() if src == "harta BVB")
                        if bvb_src:
                            notes_sec.append(f"sector din harta proprie BVB: {', '.join(bvb_src)}")
                        if notes_sec:
                            st.caption("ℹ️ " + "; ".join(notes_sec) + ".")
                    else:
                        st.info("Nu există date sectoriale (nicio poziție cu preț).")

                # 3. Matrice Corelare
                st.markdown("---")
                st.subheader("🧩 Analiză Diversificare (Corelare)")
                current_tickers = df_subset['Symbol'].unique().tolist()
                corr_for_cards = None   # cardurile pe perechi se afișează la finalul tabului (cerința lui Daniel)
                if len(current_tickers) > 1:
                    with st.spinner("Analizăm suprapunerea riscului..."):
                        corr_for_cards = plot_correlation_matrix(current_tickers, show_cards=False)
                else:
                    st.info("Adaugă cel puțin 2 active pentru analiză.")
                
                st.markdown("<br>", unsafe_allow_html=True)
                # =======================================================
                # MODUL NOU: OPTIMIZARE PORTOFOLIU AI (MARKOWITZ)
                # =======================================================
                st.markdown("---")
                st.subheader("🧠 Optimizator Portofoliu (Markowitz)")
                st.markdown("Două alocări calculate din istoricul pozițiilor tale: Sharpe maxim și varianță minimă, cu plafon pe poziție. "
                            "Sunt scenarii construite din trecut, nu recomandări: verifică mai jos cum s-ar fi comportat pe date pe care modelul nu le-a văzut.")

                if len(current_tickers) >= 2:
                    with st.spinner("Motorul Quant calculează Frontiera Eficientă..."):
                        # 1. Descărcăm prețurile de închidere curate pentru acțiunile tale
                        # close_frame elimină simbolurile fără nicio cotație (altfel optimizarea eșua cu un mesaj criptic)
                        # 5 ani: pe un singur an, Sharpe maxim alege doar câștigătorii recenți
                        hist_opt = close_frame(yf.download(current_tickers, period="5y", progress=False, auto_adjust=True), current_tickers)
                        skipped_opt = [t for t in current_tickers if t not in hist_opt.columns]
                        if skipped_opt:
                            st.caption(f"ℹ️ Fără istoric de preț, excluse din optimizare: {', '.join(skipped_opt)}.")
                        
                        # 2. Optimizare cu plafon pe poziție și covarianță Ledoit-Wolf (analytics/simulation.py)
                        opt_currency = {"$": "USD", "€": "EUR"}.get(currency_symbol, currency_symbol)
                        rf_opt, rf_opt_label = get_risk_free_for_currency(opt_currency)
                        if hist_opt.shape[1] >= 2:
                            opt_res, opt_msg = simulation.optimize_portfolio(hist_opt, rf_annual=rf_opt or 0.0, max_weight=0.25)
                        else:
                            opt_res, opt_msg = None, "Sunt necesare cel puțin 2 active cu istoric de preț pentru optimizare."

                        if opt_res:
                            # Calculăm ponderile actuale din portofoliul tău
                            total_val = df_calc['MarketValue'].sum()
                            current_w = (df_calc.groupby('Symbol')['MarketValue'].sum() / total_val * 100).to_dict()

                            # Creăm un tabel pentru a compara Ce ai TU vs Ce zice AI-ul
                            comp_data = []
                            for sym in current_tickers:
                                comp_data.append({
                                    "Simbol": sym,
                                    "Pondere Actuală (%)": current_w.get(sym, 0),
                                    "Pondere Optimă AI (%)": opt_res['allocation'].get(sym, 0),
                                    "Pondere Varianță Minimă (%)": opt_res['min_variance'].get(sym, 0)
                                })
                            df_comp = pd.DataFrame(comp_data)

                            # 3. Desenăm Graficul Comparativ Profesional
                            fig_opt = go.Figure()
                            
                            # Bara pentru Alocarea Ta
                            fig_opt.add_trace(go.Bar(
                                x=df_comp['Simbol'], 
                                y=df_comp['Pondere Actuală (%)'], 
                                name='Alocarea Ta', 
                                marker_color='#8B949E',
                                texttemplate='%{y:.1f}%',      # Scrie procentul pe bară
                                textposition='auto',           # Îl așează automat (sus sau în interior)
                                hovertemplate="<b>%{x}</b> (Acum): %{y:.2f}%<extra></extra>" # Formatare hover
                            ))
                            
                            # Bara pentru Sugestia AI
                            fig_opt.add_trace(go.Bar(
                                x=df_comp['Simbol'], 
                                y=df_comp['Pondere Optimă AI (%)'], 
                                name='Sharpe maxim', 
                                marker_color='#3987E5',
                                texttemplate='%{y:.1f}%',      # Scrie procentul pe bară
                                textposition='auto',
                                hovertemplate="<b>%{x}</b> (Sharpe maxim): %{y:.2f}%<extra></extra>" # Formatare hover
                            ))

                            # Bara pentru portofoliul de varianță minimă
                            fig_opt.add_trace(go.Bar(
                                x=df_comp['Simbol'],
                                y=df_comp['Pondere Varianță Minimă (%)'],
                                name='Varianță minimă',
                                marker_color='#C98500',
                                texttemplate='%{y:.1f}%',
                                textposition='auto',
                                hovertemplate="<b>%{x}</b> (Varianță minimă): %{y:.2f}%<extra></extra>"
                            ))
                            
                            fig_opt.update_layout(
                                barmode='group', 
                                template="plotly_dark", 
                                height=400, 
                                paper_bgcolor='rgba(0,0,0,0)', 
                                plot_bgcolor='rgba(0,0,0,0)',
                                yaxis=dict(ticksuffix="%"),    # Adaugă % pe axa verticală (Y)
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            plot_ro(fig_opt, width='stretch', key=f"opt_bar_{currency_symbol}")

                            # 4. Indicatorii alocării de Sharpe maxim, pe tot istoricul (în eșantion)
                            st.markdown("##### Alocarea de Sharpe maxim, pe tot istoricul")
                            c_op1, c_opt2, c_opt3 = st.columns(3)
                            
                            c_op1.metric("Randament Anual Istoric", f"{ro(opt_res['expected_return']):.2f}%",
                                        help=PORTF_HELP["opt_return"])
                            
                            c_opt2.metric("Volatilitate (Risc)", f"{ro(opt_res['expected_volatility']):.2f}%",
                                        help=PORTF_HELP["opt_vol"])
                            
                            c_opt3.metric("Sharpe Ratio", "N/A" if (opt_res['sharpe_ratio'] is None or rf_opt is None) else f"{ro(opt_res['sharpe_ratio']):.2f}",
                                        help=PORTF_HELP["opt_sharpe"])

                            oos = opt_res.get("out_of_sample") or []
                            if oos:
                                st.markdown(f"##### Verificare pe date nevăzute ({opt_res['test_days']} zile, din {opt_res['test_start']:%d.%m.%Y})")
                                st.dataframe(pd.DataFrame({
                                    "Alocare": [r["label"] for r in oos],
                                    "Randament anualizat": [f"{ro(r['return']):.1f}%" for r in oos],
                                    "Volatilitate": [f"{ro(r['volatility']):.1f}%" for r in oos],
                                    "Sharpe": ["N/A" if (r["sharpe"] is None or rf_opt is None) else f"{ro(r['sharpe']):.2f}" for r in oos],
                                }), hide_index=True, width='stretch')
                            st.caption(
                                f"Din {opt_res['n_days']} zile comune de tranzacționare, {opt_res['start']:%d.%m.%Y} – {opt_res['end']:%d.%m.%Y}. "
                                f"Plafon pe poziție: {ro(opt_res['cap']):.0f}%. Covarianță Ledoit-Wolf (shrinkage {ro(opt_res['shrinkage']):.2f}). "
                                + (f"Excluse pentru istoric prea scurt: {', '.join(opt_res['excluded'])}. " if opt_res.get('excluded') else "")
                                + (f"Rata fără risc: {ro(rf_opt * 100):.2f}% ({rf_opt_label}). " if rf_opt is not None else f"Rata fără risc indisponibilă ({rf_opt_label}): alocarea „Sharpe maxim” maximizează randament ÷ volatilitate (rată 0%), iar Sharpe se afișează N/A. ")
                                + ("Tabelul de verificare: ponderile au fost calculate pe primele 70% din zile și evaluate pe ultimele 30%. "
                                   "Dacă acolo nu bat ponderile egale, „optimul” e în mare parte potrivire pe trecut. " if oos else "")
                                + "Varianța minimă nu depinde de estimarea randamentelor, deci e mai stabilă decât Sharpe maxim."
                            )
                        else:
                            st.warning(opt_msg)
                else:
                    st.info("Adaugă cel puțin 2 acțiuni în portofoliu pentru ca AI-ul să poată calcula diversificarea optimă.") 

                # Ultima secțiune a tabului: cardurile pe perechi, după optimizator
                if corr_for_cards is not None:
                    st.markdown("---")
                    render_diversification_cards(corr_for_cards)

            with tab_usd:
                df_usd = df_pf[df_pf['Currency'] == 'USD']
                render_portfolio_tab(df_usd, "$")

            with tab_eur:
                df_eur = df_pf[df_pf['Currency'] == 'EUR']
                render_portfolio_tab(df_eur, "€")

            with tab_ron:
                df_ron = df_pf[df_pf['Currency'] == 'RON']
                # Folosim simbolul monedei locale
                render_portfolio_tab(df_ron, "RON")      
        _pf_timer.caption(f"⏱️ Pagina a rulat complet în {format_ro(time.perf_counter() - _pf_t0, 1)} s, la {now_ro():%H:%M:%S}.")

            # Butonul de reset nu poate șterge datele din Google Drive, doar le ignoră temporar
            # Așa că l-am comentat sau ar trebui scos, deoarece gestionarea datelor se face acum în Sheets.
            # st.markdown("---")
            # if st.button("⚠️ Șterge TOT Portofoliul (Reset)"):
            #     os.remove(FILE_PORTOFOLIU)
            #     st.rerun()

    # =================================================================
    # 4. PIAȚĂ GLOBALĂ (CU MASTER VERDICT AI INTEGRAT & SINTEZĂ DEEP DIVE)
    # =================================================================
    elif sectiune == "4. Piață Globală":
        st.title("🌐 Pulsul Pieței Globale")
        st.caption("Date în timp real și macroeconomie instituțională.")
        # Timpul rulării complete (Etapa 5): se scrie la final, ca viteza paginii să se poată măsura.
        # Fără st.fragment aici (09.11, decizia lui Daniel): după o sesiune nouă pe server (telefonul deconectat > 2 min,
        # containerul repornit) clicurile din fragmente se pierd fără mesaj, iar pagina completă rulează în ~3 s cu cache.
        _pg_t0 = time.perf_counter()
        _pg_timer = st.empty()

        with st.spinner("Motorul Macro AI descarcă și corelează indicatorii globali..."):
            # --- 1. DESCĂRCĂM ABSOLUT TOATE DATELE LA ÎNCEPUT ---
            macro_sectors = get_sector_performance()
            macro_risk_ratio = get_credit_risk_data("1y")
            macro_corr = get_cross_asset_correlation()
            macro_tickers, macro_data = get_macro_data_visuals() 
            
            # --- NOU: Descărcăm datele FRED aici sus pentru a le da motorului AI ---
            df_fred = get_fred_macro_data()
            
            # Lipsă = None (N/A), niciodată o valoare de rezervă (înainte: VIX 20, spread 0,5).
            try:
                vix_val, vix_date = get_vix_last()
            except Exception as exc:  # yfinance: rețea, JSON, date lipsă
                logger.warning("Piață Globală VIX: %s", exc)
                vix_val, vix_date = None, None
            curr_yield_spread, curve_desc = get_market_curve("US")
            try:
                sector_month = get_sector_month_returns()
            except Exception as exc:
                logger.warning("Piață Globală sectoare 1 lună: %s", exc)
                sector_month = {}

            news_samples = get_company_news_rss("^GSPC") + get_company_news_rss("^IXIC")
            from ai_engine import analyze_sentiment_ai
            macro_sentiment = analyze_sentiment_ai(news_samples) if news_samples else None

            # --- 2. CALCULĂM SCORUL GLOBAL (analytics/macro.py: piloni 0–100, media ponderată pe pilonii cu date) ---
            try:
                spy_uup = float(macro_corr.loc['Acțiuni (SPY)', 'Dolar (UUP)'])
            except (KeyError, TypeError, ValueError, AttributeError):
                spy_uup = None
            macro_pillars = {
                "credit": macro_calc.pillar_credit(macro_risk_ratio),
                "vix": macro_calc.pillar_vix(vix_val),
                "dolar": macro_calc.pillar_dollar(spy_uup),
                "sectoare": macro_calc.pillar_sectors(sector_month),
                "curba": macro_calc.pillar_curve(curr_yield_spread),
                "sentiment": macro_calc.pillar_sentiment(macro_sentiment),
            }
            mv = macro_calc.macro_verdict(macro_pillars)

        m_score = mv["score"]
        m_label = mv["label"] or "N/A"
        m_col = {"FAVORABIL": "#3FB950", "MIXT": "#D29922", "DEFAVORABIL": "#F85149"}.get(mv["label"], "#8B949E")
        m_desc = {
            "FAVORABIL": "Majoritatea pilonilor ponderați (credit, volatilitate, dolar, sectoare, curbă, sentiment) indică un mediu de piață favorabil.",
            "MIXT": "Pilonii dau semnale contradictorii; mediul de piață nu înclină clar într-o parte.",
            "DEFAVORABIL": "Majoritatea pilonilor ponderați indică stres: credit în retragere, volatilitate mare sau rotație defensivă.",
        }.get(mv["label"], f"Prea puține date: doar {mv['weight_used']}% din ponderi au valori (minimum {macro_calc.MACRO_MIN_WEIGHT}%).")
        m_score_txt = score_ro(m_score)  # o zecimală, tăiat: eticheta vine din pragurile 60/40 (#153)

        # --- 3. AFIȘARE BANNER ---
        st.markdown(f"""
            <div style="background:linear-gradient(90deg, #161B22 0%, #21262D 100%); padding:clamp(18px, 5vw, 30px); border-radius:15px; border-left: 10px solid {m_col}; margin-bottom:20px; box-shadow: 0 4px 15px rgba(0,0,0,0.5);">
                <div class="macro-hero" style="display: flex; flex-wrap: wrap; gap: 12px 24px; justify-content: space-between; align-items: center;">
                    <div style="flex: 1 1 260px; min-width: 0;">
                        <h4 style="color:#8B949E; margin:0; text-transform:uppercase; letter-spacing:1px; font-size:clamp(13px, 3.6vw, 18px);">Mediul de piață SUA (verdict macro, netestat)</h4>
                        <h1 style="color:{m_col}; margin:10px 0; font-size:clamp(26px, 8vw, 38px); overflow-wrap:normal; word-break:keep-all; hyphens:none;">{m_label}</h1>
                        <p style="color:#C9D1D9; font-size:16px;">{m_desc}</p>
                    </div>
                    <div style="text-align:center; min-width: 120px;">
                        <div style="font-size:12px; color:#8B949E;">SCOR MACRO</div>
                        <div style="font-size:clamp(40px, 11vw, 56px); font-weight:bold; color:{m_col}; font-variant-numeric:tabular-nums;">{m_score_txt}</div>
                        <div style="font-size:14px; color:#8B949E;">/ 100</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # --- 3a. STAREA PIEȚEI AZI: VIX și Fear & Greed (CNN), fără istoric (cerut de Daniel 13.11) ---
        try:
            fg = get_cnn_fear_greed()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("CNN Fear & Greed: %s", exc)
            fg = None
        sv1, sv2 = st.columns(2)
        vix_lbl = stress_calc.vix_state(vix_val)
        with sv1.container(key=ui_theme.signed_key("azi-vix", None if vix_val is None else 20 - vix_val)):
            st.metric("VIX azi", "N/A" if vix_val is None else format_ro(vix_val, 2), vix_lbl, delta_color="off",
                      delta_arrow="off",
                      help=MARKET_HELP["vix"] + (f"\n\n**Data:** {vix_date:%d.%m.%Y}." if vix_date is not None else ""))
        fg_score = None if fg is None else fg["score"]
        with sv2.container(key=ui_theme.signed_key("azi-fg", None if fg_score is None or 45 <= fg_score < 55 else fg_score - 50)):
            st.metric("Fear & Greed (CNN) azi", "N/A" if fg_score is None else f"{ro(fg_score):.0f} / 100",
                      None if fg is None else fg["label"], delta_color="off", delta_arrow="off",
                      help=MARKET_HELP["fear_greed"]
                           + ("\n\n**Acum:** CNN nu a răspuns (se reîncearcă la următoarea deschidere)." if fg is None else
                              "\n\n**Acum:** "
                              + (f"actualizat {fg['date'].tz_convert('Europe/Bucharest'):%d.%m.%Y %H:%M}; " if fg["date"] is not None else "")
                              + ("" if fg["prev_close"] is None else f"închiderea precedentă {ro(fg['prev_close']):.0f}; ")
                              + ("" if fg["prev_week"] is None else f"acum o săptămână {ro(fg['prev_week']):.0f}.")))

        # --- 3b. SINTEZA AI (mutată aici de jos, 06.11) ---
        st.subheader("✨ Sinteza Inteligenței Artificiale")
        with st.spinner("Sintetizăm intersecțiile de metale, valute, dobânzi și opțiuni..."):
            from ai_engine import generate_macro_ai_summary
            macro_bullets = generate_macro_ai_summary(
                vix_val, curr_yield_spread, macro_risk_ratio, macro_sectors, macro_corr, macro_data, df_fred
            )
            if macro_bullets:
                render_bullet_box(macro_bullets, "Informații derivate algoritmic din confluența dobânzilor, prețurilor de mărfuri "
                                  "(Aur, Petrol, Cupru), sentimentului și pieței de credit.")
            else:
                st.info("Așteptăm date suficiente pentru generarea sintezei.")

        # --- 3c. EXPLICAȚIA SCORULUI MACRO, ÎN PAGINĂ (analytics/macro.macro_explanation) ---
        st.markdown("#### 🧮 Cum s-a ajuns la scorul macro")
        st.markdown("\n\n".join(macro_calc.macro_explanation(macro_pillars, mv)["lines"]))

        # --- 4. PILONII, FIECARE CU SCORUL LUI ---
        st.markdown("#### 🔍 Pilonii verdictului")
        pillar_rows = []
        for key, weight in macro_calc.MACRO_WEIGHTS.items():
            pp = macro_pillars[key]
            pillar_rows.append({"Pilon": macro_calc.MACRO_NAMES[key], "Pondere %": weight,
                                "Scor": pp["score"], "Bază": pp["detail"]})
        st.dataframe(
            pd.DataFrame(pillar_rows).style.format({"Pondere %": lambda v: f"{v:.0f}%",
                                                    "Scor": lambda v: f"{ro(v):.0f}"}, na_rep="N/A"),
            hide_index=True, width='stretch',
            column_config={
                "Pondere %": st.column_config.Column(help="Ponderea pilonului în scor. Pilonii fără date ies din calcul, "
                                                          "iar ponderile celorlalți se reîmpart (nu contează ca neutri)."),
                "Scor": st.column_config.Column(help="0 = nefavorabil, 50 = neutru, 100 = favorabil. Regulile pe pilon:\n\n"
                    f"**Credit:** HYG/IEF față de media pe {macro_calc.CREDIT_SMA_DAYS} de ședințe: peste +1% = 100, sub −1% = 0, între = 50.\n\n"
                    f"**VIX:** ≤ {macro_calc.VIX_CALM:.0f} = 100, ≥ {macro_calc.VIX_PANIC:.0f} = 0, liniar între.\n\n"
                    "**Dolar:** corelația SPY/UUP pe 3 luni: ≤ −0,70 = 0, ≥ 0 = 100, liniar între.\n\n"
                    "**Sectoare:** primele 3 ETF-uri sectoriale SUA pe o lună: ≥ 2 ciclice (tehnologie, consum discreționar, financiar, comunicații) = 100; "
                    "altfel, utilități sau consum de bază printre ele = 0; altfel 50.\n\n"
                    "**Curba:** 10 ani − 3 luni: inversată = 0, 0–0,5 pp = 50, ≥ 0,5 pp = 100.\n\n"
                    "**Sentiment:** media FinBERT pe titlurile S&P 500 și Nasdaq: > 0,15 = 100, < −0,15 = 0, între = 50."),
                "Bază": st.column_config.Column(help="Valoarea din care s-a calculat scorul pilonului, sau motivul pentru care lipsește."),
            })
        st.caption(
            f"Scor = media ponderată a pilonilor cu date ({mv['weight_used']}% din ponderi). "
            f"FAVORABIL ≥ {macro_calc.MACRO_FAVORABLE:.0f}, DEFAVORABIL < {macro_calc.MACRO_ADVERSE:.0f}, altfel MIXT; "
            f"fără verdict sub {macro_calc.MACRO_MIN_WEIGHT}% din ponderi. Pragurile și ponderile sunt orientative, netestate istoric. "
            + (f"VIX: {vix_date:%d.%m.%Y}. " if vix_date is not None else "")
            + f"Curba: {curve_desc}.")

        # --- PIEȚELE ÎN CARE INVESTEȘTI: regim tehnic + randament în lei ---
        st.markdown("---")
        st.subheader("🌍 Piețele în care investești")
        with st.spinner("Citim indicii..."):
            try:
                mk_closes = get_market_closes()
            except Exception as exc:  # rețea / Yahoo
                logger.warning("piețe: %s", exc)
                mk_closes = pd.DataFrame()
            mk_fx, _mk_fx_failed = get_bnr_fx_history()
        if mk_closes.empty:
            st.info("Indicii nu au putut fi citiți acum de la Yahoo. Se reîncearcă la următoarea deschidere a paginii.")
        else:
            pct1 = lambda v: "N/A" if v is None or pd.isna(v) else f"{ro(v * 100):+.1f}%"
            reg_rows = []
            for key, name, sym, cur, note in markets_calc.MARKETS:
                r = markets_calc.market_regime(mk_closes[sym] if sym in mk_closes.columns else None)
                reg_rows.append({"Piață": name, "Ultima închidere": r["last"],
                                 "Data": r["date"], "Trend": None if r["trend_n"] is None else f"{r['trend_n']}/3",
                                 "Momentum 12-1": r["momentum"], "Scor moment": r["score"], "De la maximul pe 52 săpt.": r["from_high"]})
            st.markdown("**Regimul tehnic** (aceeași regulă ca eticheta „Moment” din Analiză Companie)")
            st.dataframe(
                text_frame(pd.DataFrame(reg_rows), {
                    "Ultima închidere": lambda v: format_ro(v, 2), "Data": lambda d: f"{d:%d.%m.%Y}",
                    "Trend": str, "Momentum 12-1": pct1, "Scor moment": lambda v: f"{ro(v):.0f}",
                    "De la maximul pe 52 săpt.": pct1}).style.map(sign_color, subset=["Momentum 12-1"]),
                hide_index=True, width='stretch',
                column_config={
                    "Trend": st.column_config.Column(help="Câte din 3 condiții sunt îndeplinite: închiderea peste media pe 50 de ședințe, "
                                                          "peste media pe 200, media pe 50 peste cea pe 200. 3/3 = trend ascendent complet."),
                    "Momentum 12-1": st.column_config.Column(help="Randamentul de acum 12 luni până acum o lună (fără ultima lună), pe date calendaristice. "
                                                                  "Peste +20% puternic · 0 – +20% pozitiv · −20% – 0 negativ · sub −20% slab (praguri orientative, netestate)."),
                    "Scor moment": st.column_config.Column(help="Sub-scorul tehnic 0–100 = media trendului (0–1) și a momentumului scalat (−30% → 0, +30% → 1). "
                                                                "Pragurile 80/40 sunt testate pe acțiuni BVB, nu pe indici: aici valoarea e descriptivă, fără verdict."),
                    "De la maximul pe 52 săpt.": st.column_config.Column(help="Ultima închidere ÷ cea mai mare închidere din ultimele 52 de săptămâni − 1. "
                                                                                "0% = pe maxim; −10% = corecție; −20% = piață în scădere (bear market, convenția uzuală)."),
                })

            hz = st.radio("Orizont", list(markets_calc.HORIZONS), index=2, horizontal=True, key="mk_horizon")
            ret_rows = []
            for key, name, sym, cur, note in markets_calc.MARKETS:
                fx = None if cur == "RON" else (mk_fx[cur] if (not mk_fx.empty and cur in mk_fx.columns) else pd.Series(dtype=float))
                rr = markets_calc.ron_return(mk_closes[sym] if sym in mk_closes.columns else None, fx, hz)
                ret_rows.append({"Piață": name, "Valută": cur, "Randament local": rr["local"],
                                 "Efect curs față de leu": rr["fx"], "Randament în lei": rr["ron"], "Notă": note})
            st.markdown(f"**Randamentul în lei** ({hz})")
            st.dataframe(
                text_frame(pd.DataFrame(ret_rows), {"Randament local": pct1, "Efect curs față de leu": pct1, "Randament în lei": pct1})
                .style.map(sign_color, subset=["Randament local", "Efect curs față de leu", "Randament în lei"]),
                hide_index=True, width='stretch',
                column_config={
                    "Randament local": st.column_config.Column(help="Variația indicelui în valuta lui, de la ultima închidere de la sau înainte de începutul orizontului."),
                    "Efect curs față de leu": st.column_config.Column(help="Variația cursului BNR al valutei față de leu pe aceeași perioadă "
                                                                            "(cursul din ultima zi de tranzacționare a indicelui, nu cel de azi). 0% la BET."),
                    "Randament în lei": st.column_config.Column(help="(1 + randament local) × (1 + efect curs) − 1: cât a câștigat un investitor în lei, "
                                                                     "fără dividende la indicii de preț și fără comisioane."),
                })
            st.caption("ℹ️ Indici de la Yahoo (ajustați), cursuri BNR. Indicii de preț nu includ dividendele (S&P 500, Nasdaq, STOXX 600); "
                       "DAX le include. BET e aproximat prin ETF-ul TVBETETF.")

        # --- LĂȚIMEA PIEȚEI SUA ---
        st.markdown("**Lățimea pieței SUA** (cât de larg participă companiile la creștere)")
        try:
            br_closes = get_breadth_closes()
        except Exception as exc:  # rețea / Yahoo
            logger.warning("lățime: %s", exc)
            br_closes = pd.DataFrame()
        if br_closes.empty:
            st.info("RSP / IWM / SPY nu au putut fi citite acum de la Yahoo.")
        else:
            br_rows = []
            for a_sym, b_sym, label in markets_calc.BREADTH_PAIRS:
                row_b = {"Comparație": label, "Pereche": f"{a_sym} / {b_sym}"}
                for mths in markets_calc.BREADTH_MONTHS:
                    rr_b = markets_calc.relative_return(br_closes.get(a_sym), br_closes.get(b_sym), mths)
                    row_b[f"Diferență {mths} luni"] = rr_b["diff"]
                    row_b[f"Detaliu {mths} luni"] = (None if rr_b["diff"] is None else
                                                    f"{a_sym} {ro(rr_b['a'] * 100):+.1f}% vs {b_sym} {ro(rr_b['b'] * 100):+.1f}%")
                br_rows.append(row_b)
            pct_b = lambda v: "N/A" if v is None or pd.isna(v) else f"{ro(v * 100):+.1f} pp"
            st.dataframe(
                pd.DataFrame(br_rows).style.format({f"Diferență {m_} luni": pct_b for m_ in markets_calc.BREADTH_MONTHS}, na_rep="N/A"),
                hide_index=True, width='stretch',
                column_config={f"Diferență {m_} luni": st.column_config.Column(
                    help=f"Randamentul primului ETF − al doilea pe {m_} luni (total return, până la ultima zi comună). "
                         "Negativ la RSP/SPY: creșterea S&P 500 vine din câteva companii mari (piață îngustă, mai fragilă). "
                         "Pozitiv: participare largă.") for m_ in markets_calc.BREADTH_MONTHS})

        # --- DOBÂNZI ȘI RISC DE ȚARĂ ---
        st.markdown("---")
        st.subheader("🏦 Dobânzi și risc de țară")
        with st.spinner("Citim dobânzile (FRED, BCE, Eurostat)..."):
            rate_rows = get_rates_dashboard()
        pp = lambda v: "N/A" if v is None or pd.isna(v) else f"{ro(v):+.2f}"
        df_rates = pd.DataFrame([{"Piață": r["Piață"], "Indicator": r["Indicator"], "Valoare (%)": r["value"],
                                  "Perioadă": rates_calc.period_label(r["period"]), "Δ 3 luni (pp)": r["d3m"],
                                  "Δ 1 an (pp)": r["d1y"], "Sursă": r["Sursă"]} for r in rate_rows])
        st.dataframe(
            df_rates.style.format({"Valoare (%)": lambda v: "N/A" if v is None or pd.isna(v) else f"{ro(v):.2f}",
                                   "Δ 3 luni (pp)": pp, "Δ 1 an (pp)": pp}, na_rep="N/A"),
            hide_index=True, width='stretch',
            column_config={
                "Valoare (%)": st.column_config.Column(help="Ultima valoare publicată, în procente pe an (inflația: variația prețurilor față de aceeași lună de anul trecut)."),
                "Perioadă": st.column_config.Column(help="Ziua (serii zilnice) sau luna (serii lunare) a ultimei valori. Seriile lunare BCE/Eurostat apar cu 1–2 luni întârziere."),
                "Δ 3 luni (pp)": st.column_config.Column(help="Variația în puncte procentuale față de acum 3 luni (aceeași lună, la seriile lunare)."),
                "Δ 1 an (pp)": st.column_config.Column(help="Variația în puncte procentuale față de acum un an."),
                "Indicator": st.column_config.Column(help=(
                    "**Dobânda reală** = randamentul pe 10 ani − inflația an/an din aceeași lună (ex-post, proxy: inflația așteptată nu e publicată). "
                    "Pozitivă = titlurile de stat bat inflația.\n\n"
                    "**Prima de risc de țară** = randamentul titlurilor românești pe 10 ani − Bund 10 ani, aceeași lună: cât cer investitorii în plus "
                    "pentru riscul României (fiscal, rating, curs). Când crește, titlurile de stat existente pierd din preț și evaluările BET scad; "
                    "când scade, invers.\n\n"
                    "**Panta** 10 ani − scurt: negativă = curbă inversată.")),
            })
        st.caption("ℹ️ Dobânda de politică monetară a BNR nu are o sursă automată: nu e în tabel. Randamentele RO și DE pe 10 ani sunt seriile lunare "
                   "de convergență (criteriile Maastricht) publicate de BCE.")

        # --- PANTA CURBEI SUA, ISTORIC (graficul pilonului „Curba”; înlocuiește „Early Warning System”) ---
        st.markdown("**Panta curbei SUA, 10 ani − 3 luni** (istoric; aceeași măsură ca pilonul „Curba” din verdict)")
        us_curve = get_us_curve_history()
        cv_sum = rates_calc.curve_inversion_summary(us_curve)
        if cv_sum is None:
            st.info("Seriile FRED DGS10 / DGS3MO nu au putut fi citite acum: graficul pantei e N/A.")
        else:
            fig_cv = go.Figure(go.Scatter(x=us_curve.index, y=us_curve.values, mode="lines",
                                          line=dict(color="#3987e5", width=2), name="10 ani − 3 luni",
                                          hovertemplate="%{x|%d.%m.%Y}: %{y:.2f} pp<extra></extra>"))
            fig_cv.add_hline(y=0, line=dict(color="#8B949E", dash="dot", width=1))
            fig_cv.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), template="plotly_dark",
                                 paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False,
                                 yaxis=dict(title="puncte procentuale", gridcolor='#30363D', zeroline=False))
            plot_ro(fig_cv, width='stretch', key="us_curve_chart")
            last_inv = ("niciuna în istoricul afișat" if cv_sum["last_inverted"] is None
                        else f"{cv_sum['last_inverted']:%d.%m.%Y}")
            st.caption(f"Ultima valoare {ro(cv_sum['value']):+.2f} pp ({cv_sum['date']:%d.%m.%Y}). În ultimele 12 luni curba a fost "
                       f"inversată în {cv_sum['inverted_window']} din {cv_sum['n_window']} ședințe; ultima zi inversată: {last_inv}. "
                       "Sub linia punctată (0) curba e inversată: în SUA, inversarea 10 ani − 3 luni a precedat recesiunile din 1969 "
                       "încoace, cu un decalaj de 6–24 de luni și cel puțin un semnal fals; nu dă momentul. "
                       f"Sursa: FRED DGS10 − DGS3MO, aceeași zi, din {cv_sum['start']:%m.%Y}. Pilonul din verdict folosește "
                       "^TNX − ^IRX (Yahoo, ^IRX convertit din randament de discount în randament echivalent de obligațiune), "
                       "din ultima ședință; FRED vine de obicei cu o zi în urmă, deci în zilele cu mișcări mari (ex. raportul "
                       "privind locurile de muncă) cele două pot diferi cu până la ~0,2 pp.")

        # --- PRIMA DE RISC A ACȚIUNILOR (proxy) ---
        st.markdown("**Acțiunile față de titlurile de stat** (prima de risc a acțiunilor, proxy)")
        with st.spinner("Calculăm randamentul profiturilor..."):
            erp_rows = get_equity_risk_rows()
        pc2 = lambda v: "N/A" if v is None or pd.isna(v) else f"{ro(v):.2f}%"
        # Text gata formatat: st.dataframe arată celulele nule ca „None” chiar cu Styler.format(na_rep=...).
        st.dataframe(
            text_frame(pd.DataFrame([{"Piață": r["Piață"], "Randamentul profiturilor (E/P)": r["ep"], "Titluri de stat 10 ani": r["bond"],
                                      "Prima de risc (pp)": r["gap"], "Bază": r["Bază"]} for r in erp_rows]), {
                "Randamentul profiturilor (E/P)": pc2, "Titluri de stat 10 ani": pc2,
                "Prima de risc (pp)": lambda v: f"{ro(v):+.2f}"}),
            hide_index=True, width='stretch',
            column_config={
                "Randamentul profiturilor (E/P)": st.column_config.Column(help="Profitul pe acțiune ÷ prețul = inversul P/E. "
                    "Cât profit „cumperi” la 100 de lei investiți, înainte de creștere."),
                "Prima de risc (pp)": st.column_config.Column(help="E/P − randamentul titlurilor de stat pe 10 ani, în puncte procentuale. "
                    "Mare și pozitivă: acțiunile sunt ieftine față de obligațiuni (plătesc bine riscul). Mică sau negativă: "
                    "titlurile de stat dau aproape cât profiturile, deci acțiunile sunt scumpe relativ, dacă profiturile nu cresc. "
                    "Nu ține cont de creșterea profiturilor și de inflație: proxy, fără verdict."),
            })

        # --- STRESUL PIEȚEI: structura VIX + spread-urile de credit ---
        st.markdown("---")
        st.subheader("🌡️ Stresul pieței")
        with st.spinner("Citim VIX, VIX3M și spread-urile de credit..."):
            try:
                vix_cl = get_vix_curve_closes()
                vterm = stress_calc.vix_term(vix_cl.get("^VIX"), vix_cl.get("^VIX3M"))
            except Exception as exc:  # rețea / Yahoo
                logger.warning("VIX/VIX3M: %s", exc)
                vterm = None
            credit, credit_ser = {}, {}
            for code in ("BAMLH0A0HYM2", "BAMLC0A0CM"):
                try:
                    credit_ser[code] = _fred_rate_series(code, 3700)
                    credit[code] = stress_calc.spread_stats(credit_ser[code])
                except Exception as exc:  # rețea / FRED
                    logger.warning("FRED %s: %s", code, exc)
                    credit_ser[code] = None
                    credit[code] = stress_calc.spread_stats(None)
        c_st1, c_st2, c_st3 = st.columns(3)
        state_txt = {"stres": "🔴 stres acut", "tensiune": "🟠 tensiune", "normal": "🟢 normal"}
        c_st1.metric("VIX / VIX3M", "N/A" if vterm is None else f"{ro(vterm['ratio']):.2f}",
                     None if vterm is None else state_txt[vterm["state"]], delta_color="off",
                     help=MARKET_HELP["vix_term"]
                          + ("" if vterm is None else f"\n\n**Acum:** VIX {ro(vterm['vix']):.2f}, VIX3M {ro(vterm['vix3m']):.2f}, {vterm['date']:%d.%m.%Y}."))
        for col, code, name in ((c_st2, "BAMLH0A0HYM2", "Spread high yield SUA"), (c_st3, "BAMLC0A0CM", "Spread investment grade SUA")):
            cs = credit[code]
            col.metric(name, "N/A" if cs["value"] is None else f"{ro(cs['value']):.2f} pp",
                       None if cs["d1m"] is None else f"{ro(cs['d1m']):+.2f} pp într-o lună", delta_color="inverse",
                       help=f"{MARKET_HELP['credit']}\n\n**Serie:** FRED {code}. "
                            + ("" if cs["value"] is None else
                               f"Ultima valoare {cs['date']:%d.%m.%Y}. Percentila în istoricul disponibil "
                               f"({cs['start']:%m.%Y}–{cs['date']:%m.%Y}, {cs['n']} zile): {ro(cs['pctile']):.0f}% "
                               "(100% = cel mai larg spread din perioadă). FRED publică doar ultimii ~3 ani din seriile ICE.")
                            + ("" if cs["d1y"] is None else f" Față de acum un an: {ro(cs['d1y']):+.2f} pp."))
        if vterm is not None and len(vterm["series"]) > 1:
            fig_vt = go.Figure(go.Scatter(x=vterm["series"].index, y=vterm["series"].values, mode="lines",
                                          line=dict(color="#3987e5", width=2), name="VIX / VIX3M",
                                          hovertemplate="%{x|%d.%m.%Y}: %{y:.2f}<extra></extra>"))
            fig_vt.add_hline(y=stress_calc.VIX_TERM_STRESS, line=dict(color="#8B949E", dash="dot", width=1))
            fig_vt.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0), template="plotly_dark",
                                 paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False,
                                 yaxis=dict(title="VIX / VIX3M", gridcolor='#30363D'))
            plot_ro(fig_vt, width='stretch', key="vix_term_chart")
            st.caption("Linia punctată = 1: peste ea, volatilitatea pe termen scurt depășește pe cea pe 3 luni.")

        # --- GRAFICELE CREDITULUI (înlocuiesc „Radar de Risc Sistemic”, scos 05.11) ---
        cr_c1, cr_c2 = st.columns(2)
        with cr_c1:
            st.markdown("**Spread-urile de credit SUA** (măsura directă)")
            sp_lines = [(code, name, colr) for code, name, colr in (("BAMLH0A0HYM2", "High yield", "#3987e5"),
                                                                    ("BAMLC0A0CM", "Investment grade", "#d95926"))
                        if credit_ser.get(code) is not None]
            if not sp_lines:
                st.info("Spread-urile ICE BofA nu au putut fi citite acum de la FRED: N/A.")
            else:
                fig_sp = go.Figure()
                for code, name, colr in sp_lines:
                    ser_sp = rates_calc.to_daily(credit_ser[code])
                    fig_sp.add_trace(go.Scatter(x=ser_sp.index, y=ser_sp.values, mode="lines", name=name,
                                                line=dict(color=colr, width=2),
                                                hovertemplate=name + " %{x|%d.%m.%Y}: %{y:.2f} pp<extra></extra>"))
                fig_sp.update_layout(height=260, margin=dict(l=0, r=0, t=30, b=0), template="plotly_dark",
                                     paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                                     yaxis=dict(title="puncte procentuale", gridcolor='#30363D'))
                plot_ro(fig_sp, width='stretch', key="credit_spreads_chart")
                st.caption("Randamentul obligațiunilor corporative SUA peste titlurile de stat (FRED, ICE BofA option-adjusted spread). "
                           "Urcă = investitorii cer mai mult pentru riscul de neplată. FRED publică doar ultimii ~3 ani din seriile ICE.")
        with cr_c2:
            st.markdown("**Pilonul „Piața de credit”: HYG / IEF** (un an)")
            cb = macro_calc.credit_band_frame(macro_risk_ratio)
            if cb.empty:
                st.info("HYG / IEF nu au putut fi citite acum de la Yahoo: N/A.")
            else:
                fig_cb = go.Figure()
                fig_cb.add_trace(go.Scatter(x=cb.index, y=cb["upper"], mode="lines", line=dict(width=0),
                                            showlegend=False, hoverinfo="skip"))
                fig_cb.add_trace(go.Scatter(x=cb.index, y=cb["lower"], mode="lines", line=dict(width=0),
                                            fill="tonexty", fillcolor="rgba(139,148,158,0.18)",
                                            name=f"Banda ±{ro(macro_calc.CREDIT_BAND * 100):.0f}% (neutru)", hoverinfo="skip"))
                fig_cb.add_trace(go.Scatter(x=cb.index, y=cb["sma"], mode="lines", name=f"Media pe {macro_calc.CREDIT_SMA_DAYS} de ședințe",
                                            line=dict(color="#8B949E", dash="dot", width=1.5),
                                            hovertemplate="Media %{x|%d.%m.%Y}: %{y:.3f}<extra></extra>"))
                fig_cb.add_trace(go.Scatter(x=cb.index, y=cb["ratio"], mode="lines", name="HYG / IEF",
                                            line=dict(color="#3987e5", width=2),
                                            hovertemplate="HYG/IEF %{x|%d.%m.%Y}: %{y:.3f}<extra></extra>"))
                fig_cb.update_layout(height=260, margin=dict(l=0, r=0, t=30, b=0), template="plotly_dark",
                                     paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                                     yaxis=dict(title="raport", gridcolor='#30363D'))
                plot_ro(fig_cb, width='stretch', key="credit_band_chart")
                cp = macro_pillars["credit"]
                cp_txt = "N/A" if cp["score"] is None else f"{cp['score']:.0f}"
                st.caption(f"Azi: {cp['detail']}; scor pilon {cp_txt}. "
                           "Peste bandă = apetit pentru risc (100), sub bandă = retragere din risc (0), în bandă = neutru (50); "
                           "prag orientativ, netestat. HYG/IEF (randament total, Yahoo) e un proxy: banii care trec din "
                           "obligațiunile firmelor în titlurile de stat trag raportul în jos.")

        # „✨ Sinteza Inteligenței Artificiale” a fost mutată sus, sub bannerul „Mediul de piață SUA” (06.11, cererea lui Daniel).
        st.markdown("---")
        
        if st.button("🔄 Reîmprospătează Piața"):
            _global_quote_closes.clear()
            _briefing_market.clear()
            get_macro_data_visuals.clear()
            st.rerun()

        # „Early Warning System” (10 ani − 3 luni) a fost scos (05.11, decizia lui Daniel): valoarea e în pilonul „Curba”,
        # iar istoricul în graficul pantei din „Dobânzi și risc de țară”.
        st.markdown("---")    

        # --- PASUL 1: DESCĂRCARE DATE (Trebuie să fie PRIMUL rând!) ---
        # Acum variabila macro_data este creată și poate fi folosită mai jos
        macro_tickers, macro_data = get_macro_data_visuals()

        # --- PASUL 2: MODUL INTERPRETARE DINAMICĂ DOBÂNZI ---
        st.markdown("### 🧭 Indicatori Macroeconomici")
        
        # Randamentul SUA pe 10 ani: variația în puncte procentuale pe o zi și pe o lună (analytics/macro.py).
        tnx_close = None
        if isinstance(macro_data.columns, pd.MultiIndex) and '^TNX' in macro_data.columns.levels[0]:
            tnx_close = macro_data['^TNX']['Close']
        tnx_d1 = macro_calc.last_change(tnx_close)
        tnx_m1 = macro_calc.change_over(tnx_close, 1)
        tnx_trend = macro_calc.yield_trend(tnx_m1)
        if tnx_trend is None:
            st.info("ℹ️ Randamentul SUA pe 10 ani (^TNX) nu are istoric suficient acum: interpretarea dobânzilor nu se poate face.")
        else:
            tnx_txt = (f"Randamentul pe 10 ani: {ro(tnx_d1['last']):.2f}% ({tnx_d1['date']:%d.%m.%Y}); "
                       f"{ro(tnx_d1['abs'] * 100):+.0f} pb față de ședința precedentă, {ro(tnx_m1 * 100):+.0f} pb într-o lună.")
            macro_msg = {
                "creste": "🚨 **RANDAMENTE ÎN CREȘTERE:** peste +25 pb într-o lună. Presiune pe acțiunile de creștere (Tehnologie) și cost mai mare al creditării.",
                "scade": "🟢 **REDUCERE COST CAPITAL:** peste −25 pb într-o lună. Mediu favorabil acțiunilor și refinanțării datoriilor companiilor.",
                "stabil": "⚖️ **STABILITATE DOBÂNZI:** sub ±25 pb într-o lună. Piața nu anticipează schimbări majore de politică monetară.",
            }[tnx_trend]
            st.info(f"{macro_msg}\n\n{tnx_txt} Prag orientativ (±25 pb pe o lună), netestat.")
                
        # --- 1. CONFIGURARE UI (Selectori) ---
        c_sel1, c_sel2 = st.columns([1, 3])
        
        with c_sel1:
            st.markdown("##### 1. Alege Indicator:")
            selected_macro_name = st.radio("Indicator", list(macro_tickers.keys()), label_visibility="collapsed")
            selected_macro_sym = macro_tickers[selected_macro_name]
            
            st.markdown("##### 2. Perioadă:")
            # Slider pentru timp
            time_frame = st.select_slider("Perioadă", label_visibility="collapsed", options=["1L", "3L", "6L", "1A", "3A", "5A"], value="1A")

        # --- 2. PROCESARE DATE ---
        with c_sel2:
            # Extragere Serie de Date
            series = pd.Series()
            if isinstance(macro_data.columns, pd.MultiIndex):
                try:
                    if selected_macro_sym in macro_data.columns.levels[0]:
                        series = macro_data[selected_macro_sym]['Close'].dropna()
                except Exception as exc:  # înainte: except: gol
                    logger.warning("render_portfolio_tab: %s", exc)
                    pass
            else:
                series = macro_data['Close'] # Fallback

            if not series.empty:
                # 1. Date pentru Interval (Subset)
                subset = slice_window(series, time_frame)  # fereastră calendaristică, nu număr de rânduri
                
                # --- CALCULE METRICI SEPARATE (MODIFICAREA CERUTĂ) ---
                
                curr_val = series.iloc[-1] # Valoarea curentă (Azi)

                # A. Calcul Interval (Start Slider vs Azi); lipsă = N/A, nu 0
                start_val = subset.iloc[0]
                diff_interval = curr_val - start_val
                pct_interval = (diff_interval / start_val) * 100 if start_val != 0 else None

                # B. Calcul Zi (ultima ședință vs precedenta cu valoare)
                day_chg = macro_calc.last_change(series)
                diff_day = None if day_chg is None else day_chg["abs"]
                pct_day = None if day_chg is None else day_chg["pct"]
                
                # --- FORMATARE TEXT ---
                suffix = "%" if "Yield" in selected_macro_name else ""
                val_fmt = f"{ro(curr_val):.4f}{suffix}"
                
                # --- AFIȘARE DUALĂ (2 Metric Carduri) ---
                m1, m2 = st.columns(2)
                
                fmt_delta = lambda d, pc: (None if d is None else
                                           f"{ro(d):.4f}" + ("" if pc is None else f" ({ro(pc):.2f}%)"))
                m1.metric(
                    f"Interval ({time_frame})", 
                    val_fmt, 
                    fmt_delta(diff_interval, pct_interval),
                    help=f"Față de prima valoare din fereastra aleasă ({subset.index[0]:%d.%m.%Y})."
                )
                
                m2.metric(
                    "Evoluție Azi", 
                    val_fmt, 
                    fmt_delta(diff_day, pct_day),
                    help="Ultima ședință față de precedenta cu valoare. La randamente (Yield) variația e în puncte procentuale."
                )
                
                # --- 3. GRAFIC PLOTLY ---
                fig_macro = go.Figure()
                
                fig_macro.add_trace(go.Scatter(
                    x=subset.index, 
                    y=subset.values,
                    mode='lines',
                    fill='tozeroy', 
                    line=dict(color='#58A6FF', width=2),
                    name=selected_macro_name
                ))
                
                # TRUC: Zoom-in pe axa Y pentru active stabile (Valute)
                y_min = subset.min()
                y_max = subset.max()
                is_stable = y_min > 0 and (y_max - y_min) / y_min < 0.1 
                
                range_y = [y_min * 0.999, y_max * 1.001] if is_stable else None
                
                fig_macro.update_layout(
                    height=350,
                    margin=dict(l=0, r=0, t=10, b=0),
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False),
                    yaxis=dict(
                        showgrid=True, 
                        gridcolor='#30363D',
                        autorange=True if not range_y else False,
                        range=range_y
                    )
                )
                
                plot_ro(fig_macro, width='stretch')

            else:
                st.warning("Date indisponibile sau eroare conexiune Yahoo.")

        # Ghidul, matricea, minighidul și rotația: pe toată lățimea, independent de indicatorul ales (10.11).
        # --- NOU: MODUL INTERPRETARE DINAMICĂ MACRO ---
        st.markdown("#### 🧠 Analiza Corelațiilor (Ghid Macro)")
                
        # Trimitem tot bulk-ul de date macro descărcat anterior
        try:
            _vix_for_guide = get_vix_curve_closes().get("^VIX")
        except Exception as exc:  # rețea / Yahoo
            logger.warning("ghid macro VIX: %s", exc)
            _vix_for_guide = None
        macro_verdicts = get_macro_interpretation(macro_data, _vix_for_guide)
                
        for v in macro_verdicts:
            st.info(v)
                
        # --- TABEL IMPACT SECTORIAL DINAMIC (AMBELE SCENARII) ---
        st.markdown("---")
        st.subheader("📊 Matricea de Sensibilitate Economică (Scenarii)")
                
        with st.expander("📖 Cum reacționează sectoarele la dobânzi (ghid static, nu date de azi)", expanded=False):
            impact_data = {
                "Sector": ["Tehnologie (Growth)", "Bancar & Finanțe", "Imobiliare (REITs)", "Energie & Mărfuri", "Consum de Bază"],
                "Dacă Dobânzile CRESC (↑)": [
                    "🔴 NEGATIV (Evaluări scăzute)", 
                    "🟢 POZITIV (Marje mai mari)", 
                    "🔴 NEGATIV (Costuri datorie)", 
                    "🟡 NEUTRU/POZITIV (Hedge)", 
                    "🟡 NEUTRU (Cerere stabilă)"
                ],
                "Dacă Dobânzile SCAD (↓)": [
                    "🟢 POZITIV (Expansiune multipli)", 
                    "🔴 NEGATIV (Venituri scăzute)", 
                    "🟢 POZITIV (Refinanțare ieftină)", 
                    "🔴 NEGATIV (Semnal încetinire)", 
                    "🟢 POZITIV (Randament dividend)"
                ]
            }
                
            # Afișare tabel profesional
            st.table(pd.DataFrame(impact_data))

            # --- ANALIZĂ STRATEGICĂ DUPĂ CAPITALIZARE ---
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.success("**🏢 Companii Large-Cap (Gigant)**")
                st.markdown("""
                * **Performanță optimă:** În medii cu dobânzi **MARI** (Higher for Longer).
                * **Avantaj:** Rezerve de cash care produc dobândă și rezistență la inflație.
                """)
            with col_c2:
                st.warning("**🚜 Companii Small-Cap (Mici)**")
                st.markdown("""
                * **Performanță optimă:** Când dobânzile încep să **SCADĂ** (Pivot).
                * **Avantaj:** Accesul la capital ieftin repornește motoarele de creștere și expansiune.
                """)
        with st.expander("📖 Vezi Minighid de Macroeconomie"):
            st.markdown("""
            **1. Aurul: Refugiu Financiar**
            * Tinde să aibă o corelație negativă cu dolarul.
            * Indicator de stres: Creșterea rapidă indică temeri financiare.
                    
            **2. Monedele de Refugiu**
            * USD, CHF, JPY: Investitorii migrează aici în crize.
                    
            **3. Petrolul & Mărfurile**
            * Corelat invers cu USD: Dolarul puternic = Petrol mai ieftin.
            * Gazele naturale: Influențate masiv de contextul geopolitic și sezonier.
            """)

        # =================================================================
        # MODUL ACTUALIZAT: INDICATOR DE ROTAȚIE (CHART TOP - TEXT BOTTOM)
        # =================================================================
        st.markdown("---")
        st.subheader("🔄 Indicator Rotație Sectoare (Nasdaq / Dow Jones)")
                
        rot_change = None
        try:
            # Raportul pe zilele comune; variația pe o lună calendaristică (nu 22 de rânduri)
            ratio = get_rotation_ratio()
            ratio_sma = ratio.rolling(window=50).mean()
                    
            current_ratio = ratio.iloc[-1]
            rot_ret = macro_calc.period_return(ratio, months=1)
            if rot_ret is None:
                raise ValueError("raportul Nasdaq/Dow nu acoperă o lună")
            rot_change = rot_ret * 100

            # --- PASUL A: GRAFICUL PE TOATĂ LĂȚIMEA (SUS) ---
            fig_rot = go.Figure()
                    
            # Linia Principală
            fig_rot.add_trace(go.Scatter(
                x=ratio.index, y=ratio.values, 
                mode='lines', name='Ratio Actual',
                line=dict(color='#BF91FF', width=2.5),
                fill='tozeroy', fillcolor='rgba(191, 145, 255, 0.05)'
            ))
                    
            # Linia de Trend (Media SMA 50)
            fig_rot.add_trace(go.Scatter(
                x=ratio.index, y=ratio_sma, 
                mode='lines', name='Trend (SMA 50)',
                line=dict(color='rgba(255, 255, 255, 0.4)', dash='dot', width=1.5)
            ))

            # Zoom Dinamic pe Axa Y
            y_min = ratio.min() * 0.98 
            y_max = ratio.max() * 1.02

            fig_rot.update_layout(
                height=400, # Am mărit puțin înălțimea pentru a profita de lățime
                margin=dict(l=0, r=0, t=10, b=0), 
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(
                    showgrid=True, gridcolor='#30363D',
                    range=[y_min, y_max],
                    tickformat=".3f"
                ),
                xaxis=dict(showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            # Afișăm graficul întâi
            plot_ro(fig_rot, width='stretch')

            # --- PASUL B: METRICILE ȘI EXPLICAȚIA (JOS) ---
            st.write("") # Mic spațiu între grafic și text
            c_inf1, c_inf2 = st.columns([1, 2]) # Organizăm detaliile pe două coloane sub grafic
                    
            with c_inf1:
                # Păstrăm logica de determinare a statusului
                rot_status = "🚀 CREȘTERE (Nasdaq)" if rot_change > 0 else "🏭 VALUE (Dow Jones)"
                delta_color = "#3FB950" if rot_change > 0 else "#F85149"
                        
                # Randăm cardul consolidat cu HTML/CSS
                st.markdown(f"""
                    <div style="background-color: #161B22; padding: 22px; border-radius: 12px; border: 1px solid #30363D; height: 100%;">
                        <div style="color: #8B949E; font-size: 12px; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 12px; font-weight: bold;">
                            Trend Rotație (1 lună)
                        </div>
                        <div style="font-size: 22px; font-weight: bold; color: var(--tk-text); margin-bottom: 18px;">
                            {rot_status}
                        </div>
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <div style="background: {delta_color}22; color: {delta_color}; padding: 4px 12px; border-radius: 6px; font-weight: bold; font-size: 15px; border: 1px solid {delta_color}44;">
                                {ro(rot_change):+.2f}%
                            </div>
                            <div style="color: #8B949E; font-size: 13px;">
                                Scor real: <span style="color: var(--tk-text); font-weight: bold; font-family: 'Courier New', monospace; font-size: 16px;">{ro(current_ratio):.4f}</span>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                        
            with c_inf2:
                if rot_change > 1.5:
                    st.success("🔥 **DOMINANȚĂ TECH:** Banii intră agresiv în sectoarele de creștere. Piața caută profituri mari și are apetit pentru risc.")
                elif rot_change < -1.5:
                    st.warning("🛡️ **MOD DEFENSIV:** Investitorii fug în Industriale și Value. Se caută siguranța dividendelor și a companiilor stabile.")
                else:
                    st.info("⚖️ **ECHILIBRU:** Rotația este neutră între 'Creștere' și 'Valoare'. Piața își caută o direcție clară.")
                st.caption("Variația raportului Nasdaq Composite / Dow Jones într-o lună calendaristică. Prag ±1,5%, orientativ, netestat. "
                           "Dow Jones e ponderat cu prețul, deci e doar un proxy pentru „value”.")
                            
        except Exception as e:  # rețea / Yahoo / istoric scurt
            logger.warning("rotație Nasdaq/Dow: %s", e)
            st.info("Indicatorul de rotație nu are date acum (Yahoo): N/A.")
                    
        # „Barometru Sentiment Global al Media” a fost scos (05.11, decizia lui Daniel): același scor FinBERT e
        # pilonul „Sentimentul știrilor” din verdictul de sus.
        
        # =================================================================
        # MODUL NOU: HARTA TERMICĂ A SECTOARELOR (MONEY FLOW)
        # =================================================================
        st.markdown("---")
        st.subheader("🗺️ Harta Termică a Sectoarelor (Money Flow)")
        st.markdown("Cele 11 ETF-uri sectoriale ale S&P 500 (XLK…XLC): în ce sectoare se mută capitalul.")
        heat_hz = st.radio("Orizont", ["Azi", "1 lună"], index=1, horizontal=True, key="sec_heat_hz",
                           help="Azi = ultima ședință față de precedenta (zgomotos: o zi nu e o tendință). "
                                "1 lună = randamentul pe o lună calendaristică, același folosit de pilonul „Rotația sectorială” din verdict.")
        heat_word = "zilei" if heat_hz == "Azi" else "lunii"

        with st.spinner("Se calculează fluxul de capital pe sectoare..."):
            if heat_hz == "Azi":
                df_sectors = get_sector_performance()
            else:
                df_sectors = (pd.DataFrame([{"Sector": k, "Variație %": v * 100} for k, v in sector_month.items()])
                              .sort_values("Variație %", ascending=True) if sector_month else pd.DataFrame())
            
            if not df_sectors.empty:
                # --- GRAFIC PLOTLY ORIZONTAL ---
                fig_sec_heat = go.Figure()
                # Colorăm dinamic: verde pentru plus, roșu pentru minus
                bar_colors = ['#F85149' if val < 0 else '#3FB950' for val in df_sectors['Variație %']]
                
                fig_sec_heat.add_trace(go.Bar(
                    x=df_sectors['Variație %'],
                    y=df_sectors['Sector'],
                    orientation='h',
                    marker_color=bar_colors,
                    text=[f"{ro(val):+.2f}%" for val in df_sectors['Variație %']],
                    textposition='auto',
                    textfont=dict(weight='bold'),
                    # alb doar în bară; eticheta de lângă barele scurte ia culoarea textului temei (altfel e albă pe alb)
                    insidetextfont=dict(color='white', weight='bold')
                ))
                
                fig_sec_heat.update_layout(
                    height=450,
                    margin=dict(l=0, r=0, t=10, b=0),
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=True, gridcolor='#30363D', ticksuffix="%"),
                    yaxis=dict(showgrid=False, tickfont=dict(size=13))
                )
                
                c_heat1, c_heat2 = st.columns([2, 1.2])
                with c_heat1:
                    plot_ro(fig_sec_heat, width='stretch')
                
                # --- INTERPRETAREA AI (RISK-ON vs RISK-OFF) ---
                with c_heat2:
                    st.markdown("#### 🧠 Analiza AI a Fluxului")
                    
                    # Extragem extremele
                    top_3 = df_sectors.tail(3).iloc[::-1] # Cele mai mari 3, liderii
                    bottom_3 = df_sectors.head(3) # Cele mai mici 3
                    
                    # 1. Filtrăm doar sectoarele care au crescut REA (pe verde) din TOATĂ piața
                    sectoare_verzi = df_sectors[df_sectors['Variație %'] > 0]
                    
                    # Definim categoriile macro
                    defensive = ['Utilități', 'Consum de Bază', 'Sănătate', 'Imobiliare']
                    aggressive = ['Tehnologie', 'Consum Discreționar', 'Comunicații', 'Financiar']
                    
                    # 2. Logică Financiară Avansată
                    top_sector = top_3.iloc[0] # Liderul absolut al perioadei alese
                    
                    if len(sectoare_verzi) == 0:
                        # Dacă absolut totul e pe roșu
                        st.markdown("<div style='background:rgba(248, 81, 73, 0.15); padding:15px; border-radius:10px; border-left: 5px solid #F85149;'><b>🩸 SELL-OFF GENERAL:</b><br>Niciun sector nu este pe plus. Banii ies complet din piața de acțiuni, semn de panică macro.</div>", unsafe_allow_html=True)
                    
                    elif top_sector['Sector'] == 'Energie' and top_sector['Variație %'] > 1.0 and len(sectoare_verzi) <= 3:
                        # Dacă energia explodează, iar restul pieței sângerează (Șoc de inflație)
                        st.markdown("<div style='background:rgba(210, 153, 34, 0.15); padding:15px; border-radius:10px; border-left: 5px solid #D29922;'><b>🛢️ TRADE DE INFLAȚIE (Commodity Shock):</b><br>Capitalul se refugiază agresiv în Energie în timp ce restul pieței sângerează. Acesta este un tipar clasic de frică față de inflație sau șocuri geopolitice.</div>", unsafe_allow_html=True)
                    
                    else:
                        # --- AICI ESTE MODIFICAREA NOUĂ (FINE-TUNING) ---
                        # Ne uităm STRICT la cine conduce piața AZI (Top 3 lideri care sunt și pe verde)
                        lideri_verzi = top_3[top_3['Variație %'] > 0]
                        
                        # Numărăm câți lideri sunt agresivi și câți defensivi
                        def_count = sum(1 for s in lideri_verzi['Sector'] if s in defensive)
                        agg_count = sum(1 for s in lideri_verzi['Sector'] if s in aggressive)
                        
                        # Cine are mai mulți lideri în Top 3 câștigă direcția zilei
                        if agg_count > def_count:
                            st.markdown("<div style='background:rgba(63, 185, 80, 0.15); padding:15px; border-radius:10px; border-left: 5px solid #3FB950;'><b>🚀 RISK-ON (Atac):</b><br>Banii domină sectoarele de creștere/agresive. Apetit mare pentru risc.</div>", unsafe_allow_html=True)
                        
                        elif def_count > agg_count:
                            st.markdown("<div style='background:rgba(248, 81, 73, 0.15); padding:15px; border-radius:10px; border-left: 5px solid #F85149;'><b>🛡️ RISK-OFF (Apărare):</b><br>Capitalul migrează spre dividende sigure și bunuri de necesitate. Instituțiile se protejează de o posibilă scădere.</div>", unsafe_allow_html=True)
                        
                        else:
                            # Dacă e 1 la 1 sau 0 la 0
                            st.markdown("<div style='background:#21262D; padding:15px; border-radius:10px; border-left: 5px solid #8B949E;'><b>🔄 ROTAȚIE MIXTĂ / TRANZIȚIE:</b><br>Fără o direcție macro clară. Se fac reglaje fine de portofoliu la nivel individual.</div>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Afișare Lideri cu procente
                    st.write(f"**🏆 Liderii {heat_word} (Top 3):**")
                    for _, row in top_3.iterrows():
                        st.write(f"<span style='color:#3FB950;'>▲</span> {row['Sector']} <span style='color:#8B949E; font-size:12px;'>({ro(row['Variație %']):+.2f}%)</span>", unsafe_allow_html=True)
                        
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Afișare Codași cu procente și culori inteligente
                    st.write(f"**🐢 Codașii {heat_word} (Bottom 3):**")
                    for _, row in bottom_3.iterrows():
                        if row['Variație %'] < 0:
                            # E pe minus -> Săgeată roșie
                            st.write(f"<span style='color:#F85149;'>▼</span> {row['Sector']} <span style='color:#8B949E; font-size:12px;'>({ro(row['Variație %']):+.2f}%)</span>", unsafe_allow_html=True)
                        else:
                            # E "codaș", dar e pe plus -> Săgeată galbenă (neutră)
                            st.write(f"<span style='color:#D29922;'>▲</span> {row['Sector']} <span style='color:#8B949E; font-size:12px;'>({ro(row['Variație %']):+.2f}%)</span>", unsafe_allow_html=True)
        st.caption("Clasificarea (risk-on / risk-off / rotație / trade de inflație) numără sectoarele agresive și defensive din primele 3 "
                   "ale perioadei alese. Reguli orientative, netestate; pe „Azi” reflectă o singură ședință.")
        # „Radar de Risc Sistemic” (HYG/IEF) a fost scos (05.11, decizia lui Daniel): graficul HYG/IEF cu regula
        # pilonului și spread-urile de credit sunt acum în „Stresul pieței”.
        # =================================================================
        # MODUL NOU: MATRICEA CROSS-ASSET (CORELAȚII MACRO)
        # =================================================================
        st.markdown("---")
        st.subheader("🕸️ Matricea Cross-Asset (Lichiditate & Refugiu)")
        st.markdown("Analizează cum interacționează marile clase de active în ultimele 3 luni. Corelațiile neobișnuite trădează mișcările tectonice din economie.")

        with st.spinner("Se construiește matricea quant..."):
            corr_matrix = get_cross_asset_correlation()
            
            if not corr_matrix.empty:
                c_cross1, c_cross2 = st.columns([1.5, 1])
                
                with c_cross1:
                    # Formatăm textul numerelor pentru a arăta curat (2 zecimale)
                    text_matrix = []
                    for i in range(len(corr_matrix)):
                        row_text = []
                        for j in range(len(corr_matrix)):
                            val = corr_matrix.iloc[i, j]
                            row_text.append(f"{ro(val):.2f}")
                        text_matrix.append(row_text)

                    # Desenăm Heatmap-ul (Verde = merg împreună, Roșu = opuse)
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale=CORR_SCALE,
                        zmin=-1, zmax=1, xgap=2, ygap=2,
                        text=text_matrix,
                        texttemplate="%{text}",
                        textfont=dict(color=CORR_TEXT),
                        hovertemplate="%{y} ↔ %{x}<br>Corelație: %{z:.2f}<extra></extra>"
                    ))
                    
                    fig_corr.update_layout(
                        height=400,
                        template="plotly_dark",
                        margin=dict(l=0, r=0, t=10, b=0),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
                    )
                    plot_ro(fig_corr, width='stretch')
                    
                with c_cross2:
                    st.markdown("#### 🧠 Radiografia Fluxului Global")
                    
                    # AI-ul trage concluzii automate pe baza intersecțiilor critice
                    try:
                        spy_tlt = corr_matrix.loc['Acțiuni (SPY)', 'Bonduri (TLT)']
                        spy_uup = corr_matrix.loc['Acțiuni (SPY)', 'Dolar (UUP)']
                        gld_uup = corr_matrix.loc['Aur (GLD)', 'Dolar (UUP)']
                        
                        # Regula 1: Risc vs Siguranță (SPY vs TLT)
                        if spy_tlt > 0.4:
                            st.error(f"🚨 **Șoc Inflaționist:** Acțiunile și Obligațiunile se mișcă împreună (Corelație +{ro(spy_tlt):.2f}). Diversificarea clasică (60/40) nu te protejează. Cash-ul este rege.")
                        elif spy_tlt < -0.4:
                            st.success(f"✅ **Piață Normală:** Corelația Acțiuni/Bonduri este negativă ({ro(spy_tlt):.2f}). Banii se mută ordonat între Risc și Siguranță.")
                        else:
                            st.warning(f"⚖️ **Tranziție:** Fără direcție clară între Acțiuni și Bonduri ({ro(spy_tlt):.2f}).")
                        
                        # Regula 2: Dolarul (Wrecking Ball)
                        if spy_uup < -0.5:
                            st.info(f"💵 **Dolarul dictează:** Dolarul puternic lovește acțiunile (Corelație {ro(spy_uup):.2f}). Urmărește moneda americană; dacă ea scade, bursa explodează.")
                            
                        # Regula 3: Aurul ca panică pură
                        if gld_uup > 0.3:
                            st.warning(f"🥇 **Frică Extremă:** Aurul crește ÎMPREUNĂ cu Dolarul (Corelație {ro(gld_uup):.2f}). Marile fonduri cumpără masiv orice activ de refugiu, ignorând matematica standard.")
                    except Exception as exc:  # înainte: except: gol
                        logger.warning("render_portfolio_tab: %s", exc)
                        st.write("Date insuficiente pentru diagnoza automată.")
                        
                    with st.expander("📖 Cum citești matricea?"):
                        st.write("""
                        * **Pătrate portocalii (+0,5 la +1,0):** activele se mișcă împreună. Dacă unul crește, crește și celălalt.
                        * **Pătrate albastre (−0,5 la −1,0):** corelație inversă: când unul urcă, celălalt scade. Aceasta e baza acoperirii riscului (hedging).
                        * **Pătrate gri (în jur de 0):** activele se mișcă independent.
                        * Randamente zilnice pe ultimele 3 luni, doar zilele comune (minimum 30). Pragurile regulilor (±0,4, −0,5, +0,3) sunt orientative, netestate.
                        """)
            else:
                st.info("Sincronizare date Cross-Asset...")
        # --- CALENDAR MACRO SUA (FRED releases/dates, cu cheie) ---
        st.markdown("---")
        st.subheader(f"📅 Calendar macro SUA (următoarele {fred_cal.CALENDAR_DAYS} zile)")
        cal_today = now_ro().date()
        cal_data, cal_msg = get_fred_calendar(cal_today)
        if cal_data is None:
            st.info(f"ℹ️ Calendarul nu poate fi afișat acum: {cal_msg}")
        else:
            cal_rows, cal_missing = fred_cal.upcoming(cal_data, cal_today)
            if not cal_rows:
                st.info("Nicio publicare urmărită în următoarele zile.")
            else:
                st.dataframe(
                    pd.DataFrame([{"Data": f"{r['date']:%d.%m.%Y}", "Ziua": r["weekday"],
                                   "Peste": "azi" if r["in_days"] == 0 else ("mâine" if r["in_days"] == 1 else f"{r['in_days']} zile"),
                                   "Publicare": r["label"], "Actualizează în tabelul de mai jos": r["updates"]} for r in cal_rows]),
                    hide_index=True, width='stretch',
                    column_config={
                        "Data": st.column_config.Column(help="Ziua publicării în SUA, după calendarul FRED. FRED nu dă ora; "
                                                             "majoritatea se publică dimineața la New York (după-amiaza în România). "
                                                             "Deciziile FOMC: calendarul oficial al Fed, comunicat la 14:00 ora New York "
                                                             "(21:00 ora României, cu excepția săptămânilor în care doar una dintre țări "
                                                             "a schimbat ora)."),
                        "Actualizează în tabelul de mai jos": st.column_config.Column(
                            help="Rândurile din „Barometrul Economiei Reale” care primesc valoarea nouă. FRED preia valoarea "
                                 "de obicei în aceeași zi sau a doua zi; tabelul de mai jos are cache."),
                    })
            st.caption("Sursa: FRED, calendarul publicărilor (releases/dates). Doar ziua, fără oră și fără estimări de consens. "
                       "Deciziile FOMC: calendarul oficial federalreserve.gov (ședințele programate; FRED le listează zilnic). "
                       + (f"Fără dată în interval: {', '.join(cal_missing)}. " if cal_missing else ""))
            if fred_cal.fomc_outdated(cal_today):
                st.warning("⚠️ Lista ședințelor FOMC din aplicație nu acoperă tot intervalul: trebuie adăugat programul "
                           "nou de pe federalreserve.gov (data/fred_calendar.py, FOMC_DECISIONS).")
            with st.expander("🔎 Toate publicările FRED din interval (verificarea potrivirii numelor)", expanded=False):
                st.caption("Dacă o publicare urmărită lipsește deși știi că apare în aceste zile, FRED a schimbat probabil numele: "
                           "caut-o aici și spune-mi numele exact.")
                st.write(", ".join(fred_cal.all_release_names(cal_data)) or "N/A")

        # =================================================================
        # MODUL NOU: TABEL MACROECONOMIE & SĂNĂTATEA ECONOMIEI REALE
        # =================================================================
        st.markdown("---")
        st.subheader("🏛️ Barometrul Economiei Reale (Date Oficiale FED)")
        st.markdown("Aceste date dictează politica Rezervei Federale (dobânzile). Piața reacționează violent la publicarea lor (NFP, CPI).")
        
        with st.spinner("Se interoghează baza de date a Rezervei Federale SUA..."):
            df_macro = get_fred_macro_data()
            
            if not df_macro.empty:
                col_tab, col_ai = st.columns([1.2, 1])
                
                with col_tab:
                    # Stilizare tabel: Evidențiem creșterile/scăderile
                    def color_macro_trend(val):
                        if isinstance(val, str):
                            if '+' in val: return f'color: {DF_UP}; font-weight: bold'
                            elif '-' in val: return f'color: {DF_DOWN}; font-weight: bold'
                        return ''
                    
                    # Format românesc cu punct la mii; numărătorii (NFP, ADP, JOLTS, retail, locuințe) fără zecimale.
                    # Valorile rămân numerice dedesubt (Styler), deci sortarea pe coloană rămâne corectă.
                    val_cols = ['Valoare Curentă', 'Lună Precedentă']
                    count_rows = df_macro.index[df_macro['_simbol'].map(macro_calc.fred_decimals) == 0]
                    st.dataframe(
                        df_macro[['Indicator', 'Valoare Curentă', 'Lună Precedentă', 'Evoluție (MoM)', 'Evoluție (An/An)']].style
                        .map(color_macro_trend, subset=['Evoluție (MoM)'])
                        .format(ro_spec(',.2f'), subset=val_cols)
                        .format(ro_spec(',.0f'), subset=pd.IndexSlice[count_rows, val_cols]),
                        width='stretch', hide_index=True
                    )
                    st.caption("Sursa: Federal Reserve Economic Data (FRED). MoM = față de luna precedentă. An/An = față de aceeași lună a anului trecut; pentru indicii de prețuri, aceasta este rata inflației.")
                
                with col_ai:
                    st.markdown("#### 🧠 Interpretare Strategică")
                    macro_insights = interpret_macro_data_ai(df_macro)
                    
                    if macro_insights:
                        for insight in macro_insights:
                            # Colorare automată în funcție de iconiță
                            bg_col = "rgba(248, 81, 73, 0.1)" if "🔥" in insight or "🚨" in insight else ("rgba(63, 185, 80, 0.1)" if "💪" in insight or "❄️" in insight else "#21262D")
                            b_col = "#F85149" if "🔥" in insight or "🚨" in insight else ("#3FB950" if "💪" in insight or "❄️" in insight else "#8B949E")
                            
                            st.markdown(f"""
                            <div style="background:{bg_col}; padding:15px; border-radius:10px; border-left: 4px solid {b_col}; margin-bottom: 10px; font-size:14px; line-height:1.5;">
                                {md_bold_html(insight)}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Piața se află în parametrii așteptați. Nicio anomalie macroeconomică detectată luna aceasta.")
            else:
                st.warning("Datele FRED sunt momentan indisponibile. Verifică conexiunea.")
        
        # --- TABELELE VECHI (în bloc, 09.11: o cerere pentru indici + mărfuri, topurile din datele Rezumatului) ---
        st.markdown("---")
        with st.spinner("Descărcăm datele acțiunilor..."):
            gm = get_global_market_data()

        def color_signed_txt(val):
            # celulele sunt text gata formatat („+1,23%”, „-0,50”, „N/A”): culoarea după semn
            txt = str(val)
            if txt.startswith("+"):
                return f'color: {DF_UP}'
            if txt.startswith("-"):
                return f'color: {DF_DOWN}'
            return ''

        signed = lambda v: f"{ro(v):+.2f}"
        signed_pct = lambda v: f"{ro(v):+.2f}%"
        quote_help = {
            "Preț": st.column_config.Column(help="Ultima închidere zilnică din Yahoo (neajustată). În timpul ședinței e "
                                                 "prețul curent, întârziat câteva minute."),
            "Variație": st.column_config.Column(help="Preț − închiderea precedentă a aceleiași serii, în unitățile instrumentului."),
            "Variație %": st.column_config.Column(help="(Preț ÷ închiderea precedentă − 1) × 100."),
            "Data": st.column_config.Column(help="Ziua ultimei închideri. Fiecare rând are ziua lui: bursele din SUA, Europa "
                                                 "și România și contractele futures pe mărfuri nu se închid la aceeași oră."),
        }

        col_m1, col_m2 = st.columns(2)
        for col_, title_, key_ in ((col_m1, "📊 Indici Principali", "indices"), (col_m2, "🛢️ Mărfuri (Commodities)", "commodities")):
            with col_:
                st.subheader(title_)
                st.dataframe(
                    text_frame(gm[key_], {"Preț": lambda v: format_ro(v, 2), "Variație": signed,
                                          "Variație %": signed_pct, "Data": lambda d: f"{d:%d.%m.%Y}"})
                    .style.map(color_signed_txt, subset=['Variație', 'Variație %']),
                    width='stretch', hide_index=True, column_config=quote_help
                )
        st.caption("Indici și mărfuri: o singură cerere Yahoo, închideri zilnice (cache 5 min). Mărfurile sunt contractele "
                   "futures cu scadența cea mai apropiată, în USD; BET prin ETF-ul TVBETETF, în lei.")

        # --- CURS VALUTAR FAȚĂ DE LEU ---
        st.markdown("---")
        st.subheader("💱 Curs valutar față de leu")
        with st.spinner("Citim cursurile BNR..."):
            fx_hist, fx_failed = get_bnr_fx_history()
            try:
                fx_live = get_fx_live()
            except Exception as exc:  # rețea / Yahoo
                logger.warning("curs live: %s", exc)
                fx_live = {}
        if fx_hist.empty:
            st.info("Cursurile BNR nu au putut fi citite acum. Se reîncearcă la următoarea deschidere a paginii. "
                    f"Detalii: {', '.join(fx_failed) or 'N/A'}.")
        else:
            df_fx = bnr_data.fx_table(fx_hist)
            df_fx.insert(4, "Live (Yahoo)", [
                (fx_live.get(c) * u) if (fx_live.get(c) and u == 1) else None
                for c, u in zip(df_fx["Cod"], df_fx["Unitate"])])
            df_fx["Unitate"] = [f"{u} {c}" if c != "XAU" else "1 g" for u, c in zip(df_fx["Unitate"], df_fx["Cod"])]
            pct_fmt = lambda v: "N/A" if v is None or pd.isna(v) else f"{ro(v):+.2f}%"
            # Text gata formatat: st.dataframe arată celulele nule ca „None” chiar cu Styler.format(na_rep=...).
            st.dataframe(
                text_frame(df_fx.drop(columns=["Cod"]), {
                    "Curs BNR": lambda v: format_ro(v, 4), "Live (Yahoo)": lambda v: format_ro(v, 4),
                    "Data": lambda d: f"{d:%d.%m.%Y}",
                    "Zi %": pct_fmt, "1 lună %": pct_fmt, "De la 1 ian. %": pct_fmt, "1 an %": pct_fmt,
                }).style.map(sign_color, subset=["Zi %", "1 lună %", "De la 1 ian. %", "1 an %"]),
                width='stretch', hide_index=True,
                column_config={
                    "Unitate": st.column_config.Column(help="Câți lei costă această cantitate de valută (cum publică BNR: 100 de unități la yen și forint)."),
                    "Curs BNR": st.column_config.Column(help="Cursul oficial de referință BNR, publicat în zilele bancare puțin după ora 13:00. "
                                                             "E cursul folosit în contabilitate, la declarații fiscale și de bănci ca reper."),
                    "Live (Yahoo)": st.column_config.Column(help="Cursul de piață interbancar de la Yahoo, întârziat câteva minute; doar pentru valutele principale. "
                                                                 "Diferă de BNR prin moment (BNR e fixat o dată pe zi) și sursă."),
                    "Data": st.column_config.Column(help="Ziua bancară a ultimului curs BNR."),
                    "Zi %": st.column_config.Column(help="Față de cursul BNR din ziua bancară precedentă."),
                    "1 lună %": st.column_config.Column(help="Față de ultimul curs BNR de la sau înainte de aceeași zi de acum o lună."),
                    "De la 1 ian. %": st.column_config.Column(help="Față de ultimul curs BNR din anul precedent."),
                    "1 an %": st.column_config.Column(help="Față de ultimul curs BNR de la sau înainte de aceeași zi de acum un an."),
                })
            st.caption("ℹ️ Variație pozitivă = valuta s-a apreciat față de leu (costă mai mulți lei): pozițiile tale în acea valută valorează mai mult în RON. "
                       "Sursa: BNR (bnr.ro, fișierele XML oficiale)."
                       + (f" Nu s-au putut citi: {', '.join(fx_failed)}; orizonturile neacoperite apar N/A." if fx_failed else ""))

        st.markdown("---")
        top_help = {
            "Instrument": st.column_config.Column(help="Simbolul Yahoo."),
            "Preț": st.column_config.Column(help="Închiderea din ultima ședință a pieței (neajustată), în moneda cotației."),
            "Monedă": st.column_config.Column(help="Moneda cotației, după sufixul bursei. `.L` = pence (GBp), nu lire."),
            "Variație %": st.column_config.Column(help="Închiderea ultimei ședințe față de închiderea precedentă, în procente."),
        }
        for key_, title_ in (("us", "🇺🇸 Top Mișcări SUA (Blue Chips)"), ("eu", "🇪🇺 Top Mișcări EUROPA")):
            st.subheader(title_)
            c_g, c_l = st.columns(2)
            for col_, label_, frame_ in ((c_g, "**🚀 Top Creșteri (Gainers)**", gm[f"{key_}_gain"]),
                                         (c_l, "**🔻 Top Scăderi (Losers)**", gm[f"{key_}_lose"])):
                with col_:
                    st.markdown(label_)
                    if frame_.empty:
                        st.info("Niciun titlu în această listă pentru ultima ședință (sau datele Yahoo lipsesc acum).")
                    else:
                        st.dataframe(
                            text_frame(frame_, {"Preț": price_ro, "Variație %": signed_pct})
                            .style.map(color_signed_txt, subset=['Variație %']),
                            width='stretch', hide_index=True, column_config=top_help
                        )
            sess_ = gm[f"{key_}_session"]
            st.caption(f"Ședința: {sess_:%d.%m.%Y}. " if sess_ is not None else "Ședința: N/A. ")
            st.caption("Doar titlurile tranzacționate în ședință (volum > 0), maximum 10 pe fiecare parte, fără prag de valoare; "
                       "aceleași date ca în „6. Rezumatul Zilei” (cache comun de 15 min)."
                       + (" ⚠️ Yahoo a dat prețuri pentru mai puțin de 80% din listă: topul poate fi incomplet." if gm[f"{key_}_partial"] else ""))
            st.markdown("---")
        _pg_timer.caption(f"⏱️ Pagina a rulat complet în {format_ro(time.perf_counter() - _pg_t0, 1)} s, la {now_ro():%H:%M:%S}.")

    # ==================================================
    # 5. IMPORT DATE (GOOGLE SHEETS) - BVB EXTINS & GLOBAL FIX
    # ==================================================
    elif sectiune == "5. Import Date":
        st.title("📂 Analiză Date (Cloud Sheets)")
        st.caption("Foile BVB și GLOBAL din portofoliu_db, așa cum sunt calculate în foaie (prețul vine din GOOGLEFINANCE "
                   "în foaie; aplicația nu recalculează nimic aici). Celulele goale, „-” și erorile de foaie (#DIV/0!, #N/A) "
                   "apar ca N/A, nu ca 0. Coloanele se sortează numeric.")

        if st.button("🔄 Reîncarcă Datele"):
            # golește doar funcțiile care citesc din Google Sheets, nu și datele de piață
            for _fn in (load_sheet_values, load_bvb_fundamentals, load_ledger_values, load_manual_price_values,
                        load_iv_history_values, list_company_files, build_bvb_dataset):
                _fn.clear()
            st.rerun()

        tab_bvb, tab_global = st.tabs(["🇷🇴 BVB (Local)", ":material/public: Internațional (Global)"])

        # --- TAB BVB: indicatorii pe rânduri în foaie → companiile pe rânduri aici ---
        with tab_bvb:
            st.subheader("Date BVB")
            bvb_values, bvb_err = read_sheet_values("BVB")
            bvb_df, bvb_units, bvb_dropped = sheet_view.bvb_view(bvb_values) if bvb_values else (pd.DataFrame(), {}, [])
            if bvb_err:
                st.error(f"Foaia BVB nu poate fi citită: {bvb_err}")
            elif bvb_df.empty:
                st.info("Foaia BVB este goală.")
            else:
                def _bvb_fmt(col):
                    if col not in bvb_units:
                        return lambda v: v                               # coloană text (golurile sunt deja „N/A”)
                    unit = bvb_units[col]
                    if unit == "%":
                        return ro_spec(".2f", "%")
                    if unit == " lei" or "eps" in col.lower() or "câștig pe acțiune" in col.lower():
                        return ro_spec(".4f", unit)
                    return ro_spec(",.2f", unit)
                # Coloanele text: celula goală devine textul „N/A” (st.dataframe afișează „None” la orice celulă nulă,
                # oricum ar formata Styler-ul). Coloanele numerice rămân numerice, ca sortarea să meargă (decizie 13.11).
                bvb_df = sheet_view.text_columns_na(bvb_df, bvb_units)
                st.dataframe(
                    bvb_df.style.format({c: _bvb_fmt(c) for c in bvb_df.columns}),
                    height=600, width='stretch',
                )
                st.caption(f"{len(bvb_df) - 2} companii, {len(bvb_df.columns)} indicatori. Ultimele două rânduri: "
                           f"„{sheet_view.AVERAGE_ROW}” și „{sheet_view.MEDIAN_ROW}” (din companiile cu valoare; mai puțin "
                           "sensibilă la extreme: un P/E de 200 trage media, nu și mediana).")
                if bvb_dropped:
                    st.warning("Indicatori care apar de două ori în foaie (s-a afișat prima apariție): "
                               + ", ".join(bvb_dropped))

        # --- TAB GLOBAL: o companie pe rând, moneda citită din fiecare celulă ---
        with tab_global:
            st.subheader("Date Internaționale")
            g_values, g_err = read_sheet_values("GLOBAL")
            g_df, g_units, g_cur = sheet_view.global_view(g_values) if g_values else (pd.DataFrame(), {}, pd.DataFrame())
            if g_err:
                st.error(f"Foaia GLOBAL nu poate fi citită: {g_err}")
            elif g_df.empty:
                st.info("Foaia GLOBAL este goală.")
            else:
                g_style = g_df.style
                for col in g_df.columns:
                    if col not in g_units:
                        g_style = g_style.format(lambda v: "N/A" if v is None else v, subset=[col])
                        continue
                    if g_units[col] == "%":
                        spec = ".2f"
                    elif col.strip().lower() in ("nr actiuni", "nr acțiuni"):
                        spec = ",.0f"
                    elif col == "Capitalizare (mld.)":
                        spec = ",.1f"
                    else:
                        spec = ",.2f"
                    g_style = g_style.format(ro_spec(spec, g_units[col]), subset=[col])
                    for code, rows in sheet_view.currency_groups(g_cur, col).items():
                        g_style = g_style.format(ro_spec(spec, f" {code}"), subset=pd.IndexSlice[rows, [col]])
                st.dataframe(
                    g_style, height=600, width='stretch', hide_index=True,
                    column_config={
                        "Capitalizare (mld.)": st.column_config.Column(
                            help="Capitalizarea din foaie, în miliarde, în moneda scrisă în celulă."),
                        "Industrie": st.column_config.Column(
                            help="În foaie e scrisă doar la prima companie din grup; aici e completată pe fiecare rând."),
                    },
                )
                counts = g_cur.stack().value_counts() if not g_cur.empty else pd.Series(dtype=int)
                st.caption(f"{len(g_df)} companii. Moneda e cea scrisă în fiecare celulă din foaie "
                           f"({', '.join(f'{k}: {v} celule' for k, v in counts.items()) or 'nicio monedă scrisă'}); "
                           "valorile nu sunt convertite între monede.")
                mixed = sheet_view.mixed_currency_rows(g_df, g_cur)
                if mixed:
                    st.warning("Companii cu monede diferite pe același rând (de verificat în foaie): " + "; ".join(
                        f"{name}: {', '.join(f'{c} în {m}' for c, m in odd.items())}, restul în {main}" for name, main, odd in mixed))

    # ==================================================
    # 6. REZUMATUL ZILEI (NOU & OPTIMIZAT)
    # ==================================================
    elif sectiune == "6. Rezumatul Zilei":
        st.title("🗞️ Rezumatul Zilei")
        st.caption("Ultima ședință a fiecărei piețe, din prețurile de închidere neajustate (Yahoo). Pragurile textelor "
                   "(1σ, 2σ) sunt descriptive, netestate. Fără recomandări: datele și baza lor.")
        # Timpul rulării complete (Etapa 5), scris la final; fără st.fragment (vezi Piață Globală).
        _rz_t0 = time.perf_counter()
        _rz_timer = st.empty()
        rz_top = st.container()        # „3 lucruri de știut azi” (etapa E): se completează după ce sunt calculate toate

        try:
            with st.spinner("Citim ultima ședință..."):
                brief = get_daily_briefing_data()
        except Exception as exc:  # rețea / Yahoo
            logger.warning("rezumatul zilei: %s", exc)
            brief = None
        try:
            mk_closes_rz = get_market_closes()
        except Exception as exc:  # rețea / Yahoo
            logger.warning("rezumat, regim: %s", exc)
            mk_closes_rz = pd.DataFrame()

        def _pct(v, d=2):
            return "N/A" if v is None or pd.isna(v) else format_ro(v * 100, d, "%") if v < 0 else "+" + format_ro(v * 100, d, "%")

        def _sig(v):
            return "N/A" if v is None or pd.isna(v) else (("+" if v >= 0 else "") + format_ro(v, 1, "σ"))

        STATE_TEXT_RZ = {"deschisă": "bursa e deschisă", "închisă": "bursa e închisă", "weekend": "weekend, bursa e închisă"}

        def _session_header(market, label, session):
            stt = briefing_calc.market_state(market, now_ro())
            if session is None:
                st.info(f"{label}: nicio cotație de la Yahoo acum.")
                return stt
            sd = pd.Timestamp(session)
            st.markdown(f"**Ultima ședință: {fred_cal.WEEKDAYS_RO[sd.weekday()]} {sd:%d.%m.%Y}** · {label}: {STATE_TEXT_RZ[stt['state']]} "
                        f"(ședința {stt['opens']}–{stt['closes']}, ora României; fără sărbătorile legale)")
            if briefing_calc.session_is_partial(session, stt):
                st.warning("⏳ Ședința e în desfășurare: prețurile, variațiile și volumele de azi sunt parțiale.")
            return stt

        def _regime_metrics(sym, name):
            reg = markets_calc.market_regime(mk_closes_rz[sym] if sym in mk_closes_rz.columns else None)
            r1, r2, r3 = st.columns(3)
            r1.metric(f"Trend {name}", "N/A" if reg["trend_n"] is None else f"{reg['trend_n']}/3",
                      help=MARKET_HELP["trend"])
            with r2.container(key=ui_theme.signed_key(f"mom-{name}", reg["momentum"])):
                st.metric("Momentum 12-1", _pct(reg["momentum"], 1),
                          help=MARKET_HELP["momentum"])
            r3.metric("Scor moment", "N/A" if reg["score"] is None else format_ro(reg["score"], 0),
                      help=MARKET_HELP["moment_score"])

        def _breadth_bar(b, scope):
            if not b["n"]:
                st.info("Niciun simbol din univers nu a tranzacționat în ultima ședință.")
                return
            st.markdown(compact_html(f"""
            <div style="display:flex; justify-content:space-between; margin-bottom:8px; font-size:14px;">
                <span style="color:#3FB950; font-weight:bold;">🟢 Cresc: {b['up']} ({format_ro(b['up_pct'], 1, '%')})</span>
                <span style="color:#8B949E; font-weight:bold;">⚪ Nemodificate: {b['flat']}</span>
                <span style="color:#F85149; font-weight:bold;">🔴 Scad: {b['down']} ({format_ro(b['down_pct'], 1, '%')})</span>
            </div>
            <div style="display:flex; height:14px; border-radius:7px; overflow:hidden; background-color:#30363D; margin-bottom:5px;">
                <div style="width:{b['up_pct']}%; background-color:#3FB950;"></div>
                <div style="width:{b['flat_pct']}%; background-color:#8B949E;"></div>
                <div style="width:{b['down_pct']}%; background-color:#F85149;"></div>
            </div>"""), unsafe_allow_html=True)
            st.caption(f"{scope}: {b['n']} simboluri tranzacționate în ședință"
                       + (f"; {b['not_traded']} fără tranzacții (excluse: variația lor ar fi din altă zi)" if b["not_traded"] else "")
                       + ". Nemodificat = variație exact 0.")

        def _moves_table(df, currency, show_value=True):
            # currency=None (Europa): moneda pe rând (df["currency"]) și valoarea deja în lei (df["value_ron"])
            mixed = currency is None
            unit = "" if mixed else (" lei" if currency == "RON" else f" {currency}")
            rate = None if mixed else pday_calc.rate_for(currency, rz_fx)   # valoarea în lei (prețul rămâne în moneda bursei)
            values_ron = (list(df["value_ron"]) if mixed else
                          [v * rate if (rate is not None and v is not None and pd.notna(v)) else None for v in df["value"]])
            view = pd.DataFrame({
                "Simbol": [s.replace(".RO", "") for s in df.index], "Preț": df["close"].values,
                **({"Monedă": [c or "N/A" for c in df["currency"]]} if mixed else {}),
                "Variație": (df["pct"] * 100).values, "σ": df["sigma"].values,
                **({"Valoare": [briefing_calc.money_short(v if (v is not None and pd.notna(v)) else None, "RON")
                                for v in values_ron]} if show_value else {}),
            })
            color = lambda v: "" if v is None or pd.isna(v) else (f"color: {DF_UP}" if v > 0 else f"color: {DF_DOWN}" if v < 0 else "")
            st.dataframe(
                view.style.format({"Preț": ro_spec(",.2f", unit), "Variație": ro_spec("+.2f", "%"), "σ": ro_spec("+.1f", "σ")})
                .map(color, subset=["Variație"]),
                width='stretch', hide_index=True,
                column_config={"σ": st.column_config.Column(help=("Variația zilei ÷ abaterea standard a variațiilor zilnice din "
                                                                  "ultimele 60 de ședințe (fără ziua curentă; minimum 20). "
                                                                  "") + " Sub 1σ obișnuită · 1–2σ mai mare decât de obicei · 2–3σ neobișnuită (~5% din zile) · peste 3σ extremă (praguri orientative, netestate)."),
                               "Valoare": st.column_config.Column(help="Preț × volum în ședință (valoarea tranzacționată), "
                                                                        "în lei la cursul BNR."),
                               "Variație": st.column_config.Column(help="Față de închiderea ședinței precedente, preț neajustat.")})

        def _news_list(items, src_key):
            if not items:
                return False
            st.markdown(compact_html("".join(
                f"""<div style="margin-bottom:10px; border-bottom:1px solid #30363D; padding-bottom:5px;">
                <a href="{html.escape(safe_link(it.get('link')) or '#', quote=True)}" style="color:#58A6FF; text-decoration:none; font-weight:600;" target="_blank" rel="noopener noreferrer">{html.escape(str(it.get('title', '')))}</a>
                <div style="font-size:12px; color:#8B949E;">{html.escape(str(it.get(src_key, '')))} • {html.escape(str(it.get('date_str', '')))}{future_note(it)}</div></div>"""
                for it in items)), unsafe_allow_html=True)
            return True

        # === Portofoliul meu azi (etapa B): ședința fiecărei piețe din universul întreg, nu doar din deținerile tale ===
        _rz_sessions = {}
        if brief is not None:
            _rz_sessions["bvb"] = briefing_calc.session_date(brief["bvb_close"].drop(columns=[BRIEFING_BET_PROXY], errors="ignore"))
            _rz_sessions["us"] = briefing_calc.session_date(brief["us_close"].drop(columns=BRIEFING_US_INDICES + BRIEFING_US_SECTORS,
                                                                                   errors="ignore"))
            if not brief["eu_close"].empty:      # Europa: din universul urmărit (79), nu doar din deținerile tale
                _rz_sessions["eu"] = briefing_calc.session_date(brief["eu_close"].drop(columns=BRIEFING_EU_INDICES, errors="ignore"))
        rz_fx, rz_fx_dates = rezumat_fx()          # toate sumele din Rezumatul Zilei în lei, la cursul BNR
        rz_pday = render_portfolio_today(_rz_sessions, rz_fx, rz_fx_dates) or {}
        rz_soon = render_calendar_line()

        tab_bvb, tab_eu, tab_us, tab_news = st.tabs(["🇷🇴 BVB", "🇪🇺 Europa", "🇺🇸 SUA", ":material/newspaper: Știri"])

        # === BVB ===
        with tab_bvb:
            if brief is None or brief["bvb_close"].empty:
                st.info("Cotațiile BVB nu au putut fi citite acum de la Yahoo. Se reîncearcă la următoarea deschidere a paginii.")
            else:
                bvb_cols = [c for c in brief["bvb_close"].columns if c != BRIEFING_BET_PROXY]
                bvb_t, bvb_session = briefing_calc.session_table(brief["bvb_close"][bvb_cols],
                                                                 brief["bvb_volume"].reindex(columns=bvb_cols))
                _session_header("BVB", "BVB", bvb_session)
                if brief["partial"].get("bvb"):
                    st.warning("⚠️ Yahoo a trimis acum doar o parte din cotații (sub 80% din simboluri, și după reîncercare). Ce lipsește nu e inclus în calcule; datele nu se păstrează în cache, deci se recitesc la următoarea deschidere a paginii.")
                bet = briefing_calc.daily_move(brief["bvb_close"].get(BRIEFING_BET_PROXY))
                bvb_b = briefing_calc.breadth_counts(bvb_t)
                bvb_turn = briefing_calc.turnover_vs_median(brief["bvb_close"][bvb_cols], brief["bvb_volume"].reindex(columns=bvb_cols))
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("BET (proxy: ETF TVBETETF)", "N/A" if bet["last"] is None else format_ro(bet["last"], 2, " lei"),
                          None if bet["pct"] is None else _pct(bet["pct"]),
                          help="Prețul ETF-ului care urmărește BET, nu nivelul indicelui (Yahoo nu are BET). "
                               + (f"Ultima cotație: {bet['date']:%d.%m.%Y}." if bet["date"] is not None else ""))
                with m2.container(key=ui_theme.signed_key("bet-sigma", bet["sigma"])):
                    st.metric("Mișcarea BET în σ", _sig(bet["sigma"]), help=MARKET_HELP["sigma"]
                              + f"\n\n**Azi:** {briefing_calc.sigma_label(bet['sigma'])}, față de ultimele {bet['n']} ședințe.")
                m3.metric("Lățime (cresc / scad)", f"{bvb_b['up']} / {bvb_b['down']}" if bvb_b["n"] else "N/A",
                          help=MARKET_HELP["breadth"] + "\n\n**Univers:** BET Plus, BET-FI, BETAeRO (tools/univers_bvb.txt), "
                               "companiile care au tranzacționat în ultima ședință.")
                m4.metric("Valoare tranzacționată", briefing_calc.money_short(bvb_turn["today"], "RON"),
                          None if bvb_turn["ratio"] is None else f"×{format_ro(bvb_turn['ratio'], 2)} față de mediană",
                          delta_color="off",
                          delta_arrow=briefing_calc.ratio_arrow(bvb_turn["ratio"]),
                          help=MARKET_HELP["turnover"] + f"\n\n**Mediana:** din {bvb_turn['n']} ședințe dinainte.")
                st.markdown("#### 🎯 Regimul BET (regula „Moment”)")
                _regime_metrics(BRIEFING_BET_PROXY, "BET")
                st.markdown("---")
                st.markdown("#### ⚖️ Lățimea pieței")
                bvb_universe = briefing_bvb_universe()
                bvb_missing = briefing_calc.missing_symbols(bvb_universe, bvb_t)
                _breadth_bar(bvb_b, f"Universul urmărit BVB ({len(bvb_universe)} simboluri, {len(bvb_t)} cu date Yahoo)")
                if bvb_missing:
                    st.caption((f"Necitite acum de la Yahoo ({len(bvb_missing)}), excluse din calcule: " if brief["partial"].get("bvb") else
                                f"Fără niciun preț în Yahoo în ultimele 6 luni ({len(bvb_missing)}), excluse din toate calculele: ")
                               + ", ".join(s.replace(".RO", "") for s in bvb_missing) + ".")
                st.markdown("---")
                bvb_up, bvb_down = briefing_calc.top_moves(bvb_t, n=briefing_calc.TOP_N, min_value=briefing_calc.MIN_VALUE_RON)
                st.caption(f"Topurile: doar companiile cu valoare tranzacționată ≥ {briefing_calc.money_short(briefing_calc.MIN_VALUE_RON, 'RON')} "
                           "în ședință (sub prag, o singură tranzacție mică mișcă prețul).")
                c_up, c_dn = st.columns(2)
                with c_up:
                    st.markdown(f"**🚀 Top {briefing_calc.TOP_N} creșteri**")
                    if bvb_up is not None and not bvb_up.empty:
                        _moves_table(bvb_up, "RON")
                    else:
                        st.info("Nicio creștere peste prag.")
                with c_dn:
                    st.markdown(f"**🔻 Top {briefing_calc.TOP_N} scăderi**")
                    if bvb_down is not None and not bvb_down.empty:
                        _moves_table(bvb_down, "RON")
                    else:
                        st.info("Nicio scădere peste prag.")
                st.markdown("---")
                st.subheader("📊 Top lichiditate (valoare tranzacționată)")
                liq = briefing_calc.liquidity_top(bvb_t, n=10)
                if liq is not None and not liq.empty:
                    _moves_table(liq, "RON")
                else:
                    st.info("Nicio companie din univers nu a tranzacționat în ultima ședință.")
                full = briefing_calc.full_table(bvb_t, briefing_bvb_sections())
                if full is not None and not full.empty:
                    n_traded = int(full["traded"].sum())
                    with st.expander(f"📋 Toate companiile urmărite ({len(full)}; {n_traded} tranzacționate în ședință)"):
                        st.caption("Fără prag de valoare: la valori mici, o singură tranzacție mișcă prețul. "
                                   "La titlurile fără tranzacții în ultima ședință variația, σ și valoarea sunt N/A, "
                                   "iar „Ultima cotație” arată data ultimului preț din Yahoo.")
                        tr = full["traded"].astype(bool)
                        fv = pd.DataFrame({
                            "Piață": full["section"].fillna("N/A").values,
                            "Simbol": [s.replace(".RO", "") for s in full.index],
                            "Preț": full["close"].values,
                            "Variație": (full["pct"] * 100).where(tr).values,
                            "σ": full["sigma"].where(tr).values,
                            "Valoare": [briefing_calc.money_short(v, "RON") if t else "N/A"
                                        for v, t in zip(full["value"], tr)],
                            "Ultima cotație": [d.strftime("%d.%m.%Y") for d in full["date"]],
                        })
                        f_color = lambda v: "" if v is None or pd.isna(v) else (f"color: {DF_UP}" if v > 0 else f"color: {DF_DOWN}" if v < 0 else "")
                        st.dataframe(
                            fv.style.format({"Preț": ro_spec(",.2f", " lei"), "Variație": ro_spec("+.2f", "%"),
                                             "σ": ro_spec("+.1f", "σ")}, na_rep="N/A")
                            .map(f_color, subset=["Variație"]),
                            width='stretch', hide_index=True, height=min(38 + 35 * len(fv), 600),
                            column_config={"σ": st.column_config.Column(help=("Variația zilei ÷ abaterea standard a variațiilor "
                                                                             "din ultimele 60 de ședințe (fără ziua curentă; minimum 20).") + " Sub 1σ obișnuită · 1–2σ mai mare decât de obicei · 2–3σ neobișnuită (~5% din zile) · peste 3σ extremă (praguri orientative, netestate)."),
                                           "Valoare": st.column_config.Column(help="Preț × volum în ședință."),
                                           "Piață": st.column_config.Column(help="Secțiunea din tools/univers_bvb.txt "
                                                                                "(componența BET Plus, BET-FI, BETAeRO la 04.11.2026).")})

        # === EUROPA (etapa C): toate sumele în lei la cursul BNR; prețurile în moneda bursei ===
        with tab_eu:
            if brief is None or brief["eu_close"].empty:
                st.info("Cotațiile Europa nu au putut fi citite acum de la Yahoo. Se reîncearcă la următoarea deschidere a paginii.")
            else:
                eu_cols = [c for c in brief["eu_close"].columns if c not in BRIEFING_EU_INDICES]
                eu_close, eu_vol = brief["eu_close"][eu_cols], brief["eu_volume"].reindex(columns=eu_cols)
                eu_t, eu_session = briefing_calc.session_table(eu_close, eu_vol)
                _session_header("XETRA", "Europa (reper: XETRA)", eu_session)
                if brief["partial"].get("eu"):
                    st.warning("⚠️ Yahoo a trimis acum doar o parte din cotații (sub 80% din simboluri, și după reîncercare). Ce lipsește nu e inclus în calcule; datele nu se păstrează în cache, deci se recitesc la următoarea deschidere a paginii.")
                i_cols = st.columns(4)
                for col, (sym, name, kind) in zip(i_cols, (("^STOXX", "STOXX Europe 600", "indice de preț"),
                                                           ("^GDAXI", "DAX", "indice de performanță (cu dividendele reinvestite)"),
                                                           ("^FCHI", "CAC 40", "indice de preț"),
                                                           ("^FTSE", "FTSE 100", "indice de preț"))):
                    mv = briefing_calc.daily_move(brief["eu_close"].get(sym))
                    col.metric(name, "N/A" if mv["last"] is None else format_ro(mv["last"], 2),
                               None if mv["pct"] is None else f"{_pct(mv['pct'])} · {_sig(mv['sigma'])}",
                               help=MARKET_HELP["sigma"]
                                    + f"\n\n**Azi:** {briefing_calc.sigma_label(mv['sigma'])}, față de ultimele {mv['n']} ședințe. {kind.capitalize()}."
                                    + (f" Ultima cotație: {mv['date']:%d.%m.%Y}." if mv["date"] is not None else ""))

                eu_rates = {s: pday_calc.rate_for(briefing_calc.quote_currency(s), rz_fx) for s in eu_cols}
                eu_t = briefing_calc.with_ron_value(eu_t, eu_rates)
                eu_close_ron, eu_no_fx = briefing_calc.close_in_ron(eu_close, eu_rates)
                eu_turn = briefing_calc.turnover_vs_median(eu_close_ron, eu_vol.reindex(columns=eu_close_ron.columns))
                eu_b = briefing_calc.breadth_counts(eu_t)
                eu_cur = sorted({c for c in (briefing_calc.quote_currency(s) for s in eu_cols) if c})

                def _eu_fx(c):
                    base = "GBP" if c in pday_calc.PENCE_CODES else c
                    r = pday_calc.rate_for(c, rz_fx)
                    if not r:
                        return f"{c}: curs indisponibil"
                    d = rz_fx_dates.get(base)
                    return (f"1 {c} = {format_ro(r, 4)} lei" + (" (GBP ÷ 100)" if base != c else "")
                            + (f", {d:%d.%m.%Y}" if d is not None else ""))

                eu_fx_txt = "; ".join(_eu_fx(c) for c in eu_cur)
                m1, m2 = st.columns(2)
                m1.metric("Lățime (cresc / scad)", f"{eu_b['up']} / {eu_b['down']}" if eu_b["n"] else "N/A",
                          help=MARKET_HELP["breadth"] + "\n\n**Univers:** DAX 40, CAC 40, Londra, Milano "
                               "(tools/univers_urmarit_ue.txt), companiile care au tranzacționat în ultima ședință.")
                m2.metric("Valoare tranzacționată (universul urmărit)", briefing_calc.money_short(eu_turn["today"], "RON"),
                          None if eu_turn["ratio"] is None else f"×{format_ro(eu_turn['ratio'], 2)} față de mediană",
                          delta_color="off",
                          delta_arrow=briefing_calc.ratio_arrow(eu_turn["ratio"]),
                          help=MARKET_HELP["turnover"] + "\n\n**Detalii:** fiecare titlu în lei la cursul BNR al "
                               f"monedei lui ({eu_fx_txt}), față de mediana celor {eu_turn['n']} ședințe dinainte (minimum 10; "
                               "același curs pe toate ședințele, deci raportul nu depinde de curs). Doar universul, nu toată bursa.")
                if eu_no_fx:
                    st.caption("Fără curs BNR, excluse din sumele în lei: " + ", ".join(eu_no_fx) + ".")

                st.markdown("#### 🎯 Regimul STOXX 600 (regula „Moment”)")
                _regime_metrics("^STOXX", "STOXX 600")
                st.markdown("---")
                st.markdown("#### ⚖️ Lățimea pieței")
                eu_universe = universes.symbols("eu")
                eu_missing = briefing_calc.missing_symbols(eu_universe, eu_t)
                _breadth_bar(eu_b, f"Universul urmărit Europa ({len(eu_universe)} simboluri, {len(eu_t)} cu date Yahoo)")
                st.caption("Ședința e ultima zi cu preț la cel puțin 20% din titluri. Bursele au sărbători diferite (Londra, "
                           "de exemplu): într-o zi închisă doar acolo, titlurile ei apar fără tranzacții și ies din calcule.")
                if eu_missing:
                    st.caption((f"Necitite acum de la Yahoo ({len(eu_missing)}), excluse din calcule: " if brief["partial"].get("eu") else
                                f"Fără niciun preț în Yahoo în ultimele 6 luni ({len(eu_missing)}), excluse din toate calculele: ")
                               + ", ".join(eu_missing) + ".")
                st.markdown("---")
                eu_up, eu_down = briefing_calc.top_moves(eu_t, n=briefing_calc.TOP_N)
                st.caption("Topurile: fără prag de valoare (toate titlurile sunt componente ale indicilor mari; decizia din "
                           "05.11.2026). Prețul în moneda bursei (Londra în pence, GBp); valoarea în lei.")
                c_eu1, c_eu2 = st.columns(2)
                with c_eu1:
                    st.markdown(f"**🚀 Top {briefing_calc.TOP_N} creșteri (universul urmărit)**")
                    if eu_up is not None and not eu_up.empty:
                        _moves_table(eu_up, None)
                    else:
                        st.info("Nicio creștere în ultima ședință.")
                with c_eu2:
                    st.markdown(f"**🔻 Top {briefing_calc.TOP_N} scăderi (universul urmărit)**")
                    if eu_down is not None and not eu_down.empty:
                        _moves_table(eu_down, None)
                    else:
                        st.info("Nicio scădere în ultima ședință.")
                st.markdown("---")
                st.subheader("📊 Top lichiditate (valoare tranzacționată, în lei)")
                eu_liq = briefing_calc.liquidity_top_ron(eu_t, n=10)
                if eu_liq is not None and not eu_liq.empty:
                    _moves_table(eu_liq, None)
                else:
                    st.info("Nicio companie din univers nu a tranzacționat în ultima ședință.")
                eu_full = briefing_calc.full_table(eu_t, universes.sections("eu"))
                if eu_full is not None and not eu_full.empty:
                    tr = eu_full["traded"].astype(bool)
                    with st.expander(f"📋 Toate companiile urmărite ({len(eu_full)}; {int(tr.sum())} tranzacționate în ședință)"):
                        st.caption("La titlurile fără tranzacții în ultima ședință variația, σ și valoarea sunt N/A, iar "
                                   "„Ultima cotație” arată data ultimului preț din Yahoo.")
                        fv = text_frame(pd.DataFrame({
                            "Piață": eu_full["section"].fillna("N/A").values,
                            "Simbol": list(eu_full.index),
                            "Preț": eu_full["close"].values,
                            "Monedă": [c or "N/A" for c in eu_full["currency"]],
                            "Variație": (eu_full["pct"] * 100).where(tr).values,
                            "σ": eu_full["sigma"].where(tr).values,
                            "Valoare": [briefing_calc.money_short(v, "RON") if (t and pd.notna(v)) else "N/A"
                                        for v, t in zip(eu_full["value_ron"], tr)],
                            "Ultima cotație": [d.strftime("%d.%m.%Y") for d in eu_full["date"]],
                        }), {"Preț": lambda v: format_ro(v, 2),
                             "Variație": lambda v: ("+" if v > 0 else "") + format_ro(v, 2, "%"),
                             "σ": lambda v: ("+" if v >= 0 else "") + format_ro(v, 1, "σ")})
                        s_color = lambda v: (f"color: {DF_UP}" if str(v).startswith("+") and str(v) != "+0,00%"
                                             else f"color: {DF_DOWN}" if str(v).startswith("-") else "")
                        st.dataframe(fv.style.map(s_color, subset=["Variație"]), width='stretch', hide_index=True,
                                     height=min(38 + 35 * len(fv), 600),
                                     column_config={"σ": st.column_config.Column(help=("Variația zilei ÷ abaterea standard a "
                                                                                      "variațiilor din ultimele 60 de ședințe "
                                                                                      "(fără ziua curentă; minimum 20).") + " Sub 1σ obișnuită · 1–2σ mai mare decât de obicei · 2–3σ neobișnuită (~5% din zile) · peste 3σ extremă (praguri orientative, netestate)."),
                                                    "Valoare": st.column_config.Column(help="Preț × volum în ședință, în lei la cursul BNR."),
                                                    "Piață": st.column_config.Column(help="Secțiunea din tools/univers_urmarit_ue.txt.")})

        # === SUA ===
        with tab_us:
            if brief is None or brief["us_close"].empty:
                st.info("Cotațiile SUA nu au putut fi citite acum de la Yahoo. Se reîncearcă la următoarea deschidere a paginii.")
            else:
                us_cols = [c for c in brief["us_close"].columns if c not in BRIEFING_US_INDICES + BRIEFING_US_SECTORS]
                us_t, us_session = briefing_calc.session_table(brief["us_close"][us_cols], brief["us_volume"].reindex(columns=us_cols))
                _session_header("NYSE", "Wall Street", us_session)
                if brief["partial"].get("us"):
                    st.warning("⚠️ Yahoo a trimis acum doar o parte din cotații (sub 80% din simboluri, și după reîncercare). Ce lipsește nu e inclus în calcule; datele nu se păstrează în cache, deci se recitesc la următoarea deschidere a paginii.")
                i_cols = st.columns(3)
                for col, (sym, name) in zip(i_cols, (("^GSPC", "S&P 500"), ("^IXIC", "Nasdaq Composite"), ("^DJI", "Dow Jones"))):
                    mv = briefing_calc.daily_move(brief["us_close"].get(sym))
                    col.metric(name, "N/A" if mv["last"] is None else format_ro(mv["last"], 2),
                               None if mv["pct"] is None else f"{_pct(mv['pct'])} · {_sig(mv['sigma'])}",
                               help=MARKET_HELP["sigma"]
                                    + f"\n\n**Azi:** {briefing_calc.sigma_label(mv['sigma'])}, față de ultimele {mv['n']} ședințe. Indice de preț (fără dividende).")

                st.markdown("#### 🌡️ Volatilitatea așteptată (VIX)")
                try:
                    vix_c = get_vix_curve_closes()
                except Exception as exc:  # rețea / Yahoo
                    logger.warning("rezumat, VIX: %s", exc)
                    vix_c = pd.DataFrame()
                vt = stress_calc.vix_term(vix_c.get("^VIX"), vix_c.get("^VIX3M")) if not vix_c.empty else None
                vix_s = vix_c["^VIX"].dropna() if "^VIX" in vix_c.columns else pd.Series(dtype=float)
                v1, v2, v3 = st.columns(3)
                v1.metric("VIX", "N/A" if vix_s.empty else format_ro(vix_s.iloc[-1], 2),
                          help=MARKET_HELP["vix"])
                v2.metric("Percentila VIX pe 1 an", "N/A" if vix_s.empty else format_ro(stress_calc.percentile_rank(vix_s), 0, "%"),
                          help=MARKET_HELP["vix_pct"] + f"\n\n**Eșantion:** {len(vix_s)} închideri.")
                v3.metric("VIX / VIX3M", "N/A" if vt is None else f"{format_ro(vt['ratio'], 2)} ({vt['state']})",
                          help=MARKET_HELP["vix_term"])

                st.markdown("---")
                st.markdown("#### ⚖️ Lățimea pieței")
                try:
                    br_c = get_breadth_closes()
                except Exception as exc:  # rețea / Yahoo
                    logger.warning("rezumat, lățime: %s", exc)
                    br_c = pd.DataFrame()
                b1, b2 = st.columns(2)
                for col, (a_sym, b_sym, label) in zip((b1, b2), markets_calc.BREADTH_PAIRS):
                    rd = briefing_calc.relative_day(br_c.get(a_sym), br_c.get(b_sym)) if not br_c.empty else {"diff_pp": None}
                    with col.container(key=ui_theme.signed_key(f"latime-{a_sym}-{b_sym}", rd["diff_pp"])):
                        st.metric(f"{a_sym} − {b_sym} (o zi)",
                                  "N/A" if rd["diff_pp"] is None else ("+" if rd["diff_pp"] >= 0 else "") + format_ro(rd["diff_pp"], 2, " pp"),
                                  help=MARKET_HELP["rsp_spy"] + f"\n\n**Pereche:** {label} ({a_sym} − {b_sym}, prețuri ajustate).")
                _breadth_bar(briefing_calc.breadth_counts(us_t), f"Universul urmărit ({len(us_t)} simboluri, nu S&P 500)")
                us_turn = briefing_calc.turnover_vs_median(brief["us_close"][us_cols], brief["us_volume"].reindex(columns=us_cols))
                usd_rz = rz_fx.get("USD")
                st.metric("Valoare tranzacționată (universul urmărit)",
                          briefing_calc.money_short(us_turn["today"] * usd_rz if (usd_rz and us_turn["today"] is not None) else None, "RON"),
                          None if us_turn["ratio"] is None else f"×{format_ro(us_turn['ratio'], 2)} față de mediană",
                          delta_color="off",
                          delta_arrow=briefing_calc.ratio_arrow(us_turn["ratio"]),
                          help=MARKET_HELP["turnover"] + "\n\n**Detalii:** în lei la cursul BNR"
                               + (f" ({format_ro(usd_rz, 4)} lei/USD, {rz_fx_dates['USD']:%d.%m.%Y})" if usd_rz and "USD" in rz_fx_dates else " (indisponibil)")
                               + f", față de mediana celor {us_turn['n']} ședințe dinainte (raportul nu depinde de curs).")

                st.markdown("---")
                st.markdown("#### 🎯 Regimul S&P 500 (regula „Moment”)")
                _regime_metrics("^GSPC", "S&P 500")

                st.markdown("---")
                st.markdown("#### 🧩 Sectoarele S&P 500 (ETF-urile SPDR, o zi)")
                sec_cols = [s for s in BRIEFING_US_SECTORS if s in brief["us_close"].columns]
                if sec_cols:
                    sec_df, sec_session = briefing_calc.sector_moves(brief["us_close"][sec_cols],
                                                                     brief["us_volume"].reindex(columns=sec_cols),
                                                                     brief["us_close"].get("^GSPC"))
                    sv = text_frame(pd.DataFrame({
                        "Sector": sec_df["name"].values, "ETF": list(sec_df.index),
                        "Variație": (sec_df["pct"] * 100).values, "σ": sec_df["sigma"].values,
                        "Față de S&P 500": sec_df["vs_index_pp"].values,
                    }), {"Variație": lambda v: ("+" if v > 0 else "") + format_ro(v, 2, "%"),
                         "σ": lambda v: ("+" if v >= 0 else "") + format_ro(v, 1, "σ"),
                         "Față de S&P 500": lambda v: ("+" if v > 0 else "") + format_ro(v, 2, " pp")})
                    s_color = lambda v: (f"color: {DF_UP}" if str(v).startswith("+") and not str(v).startswith("+0,00")
                                         else f"color: {DF_DOWN}" if str(v).startswith("-") else "")
                    st.dataframe(sv.style.map(s_color, subset=["Variație", "Față de S&P 500"]), width='stretch', hide_index=True,
                                 column_config={
                                     "Variație": st.column_config.Column(help="Față de închiderea precedentă, preț neajustat: în ziua "
                                                                              "ex-dividend scăderea include dividendul."),
                                     "σ": st.column_config.Column(help=("Variația zilei ÷ abaterea standard a variațiilor zilnice ale "
                                                                       "ETF-ului din ultimele 60 de ședințe (fără ziua curentă; minimum 20). "
                                                                       "") + " Sub 1σ obișnuită · 1–2σ mai mare decât de obicei · 2–3σ neobișnuită (~5% din zile) · peste 3σ extremă (praguri orientative, netestate)."),
                                     "Față de S&P 500": st.column_config.Column(help="Variația ETF-ului minus variația S&P 500 în "
                                                                                     "aceeași ședință, în puncte procentuale. Pozitiv = "
                                                                                     "sectorul a făcut mai bine decât indicele.")})
                    if sec_session is not None and us_session is not None and pd.Timestamp(sec_session) != pd.Timestamp(us_session):
                        st.caption(f"Atenție: ultima ședință a ETF-urilor sectoriale ({pd.Timestamp(sec_session):%d.%m.%Y}) diferă "
                                   "de cea a universului.")
                    st.caption("Cele 11 ETF-uri SPDR Select Sector, câte unul pe sector GICS din S&P 500 (ponderate cu "
                               "capitalizarea). Sortate după variația zilei.")
                else:
                    st.info("ETF-urile sectoriale nu au putut fi citite acum de la Yahoo.")

                st.markdown("---")
                us_up, us_down = briefing_calc.top_moves(us_t, n=briefing_calc.TOP_N)
                c_us1, c_us2 = st.columns(2)
                with c_us1:
                    st.markdown(f"**🚀 Top {briefing_calc.TOP_N} creșteri (universul urmărit)**")
                    if us_up is not None and not us_up.empty:
                        _moves_table(us_up, "USD")
                    else:
                        st.info("Nicio creștere în ultima ședință.")
                with c_us2:
                    st.markdown(f"**🔻 Top {briefing_calc.TOP_N} scăderi (universul urmărit)**")
                    if us_down is not None and not us_down.empty:
                        _moves_table(us_down, "USD")
                    else:
                        st.info("Nicio scădere în ultima ședință.")


        # === ȘTIRI (decizia lui Daniel, 05.11 seara): toate știrile Rezumatului într-un singur tab ===
        with tab_news:
            rz_news = render_holdings_news(rz_pday.get("holdings"))
            st.markdown("---")
            st.subheader("🇷🇴 Top 5 știri financiare (România)")
            raw_news = fetch_news_data()
            ro_sources = ["Ziarul Financiar", "Biziday", "Economica", "Bursa", "Profit.ro", "StartupCafe", "Financial Intelligence", "Wall-Street"]
            ro_news = [n for n in raw_news if any(src.lower() in n['source'].lower() for src in ro_sources)]
            if not ro_news:
                ro_news = filter_news(raw_news, "Financiar") + filter_news(raw_news, "Energie")
            seen_ro, unique_ro_news = set(), []
            for n in ro_news:
                if n['title'] not in seen_ro:
                    unique_ro_news.append(n)
                    seen_ro.add(n['title'])
            if not _news_list(unique_ro_news[:5], "source"):
                st.info("Nu s-au găsit știri locale recente.")
            st.markdown("---")
            st.subheader("🇺🇸 Top 10 știri Wall Street")
            if not _news_list(get_us_index_news(), "publisher"):
                st.info("Nu s-au putut încărca știrile din SUA.")

        # === 3 lucruri de știut azi (etapa E): reguli fixe din analytics/day_notes.py, afișate sus ===
        rz_rows = rz_pday.get("rows")
        rz_by_sym = {}
        for r in ([] if rz_rows is None else rz_rows.itertuples(index=False)):
            rz_by_sym[r.Symbol] = {k: (None if pd.isna(getattr(r, k)) else float(getattr(r, k)))
                                   for k in ("Pct", "Sigma", "Contribution")}
        rz_tot = rz_pday.get("totals")
        rz_moves, rz_sectors = [], []
        if brief is not None:
            for nm, key, sym in (("BET", "bvb_close", BRIEFING_BET_PROXY), ("STOXX 600", "eu_close", "^STOXX"),
                                 ("S&P 500", "us_close", "^GSPC")):
                rz_moves.append((nm, briefing_calc.daily_move(brief[key].get(sym))))
            sec_cols_rz = [s for s in BRIEFING_US_SECTORS if s in brief["us_close"].columns]
            if sec_cols_rz:
                sec_rz, _ = briefing_calc.sector_moves(brief["us_close"][sec_cols_rz], brief["us_volume"].reindex(columns=sec_cols_rz),
                                                       brief["us_close"].get("^GSPC"))
                rz_sectors = [{"name": r.name, "etf": s, "pct": None if pd.isna(r.pct) else float(r.pct),
                               "vs_index_pp": None if pd.isna(r.vs_index_pp) else float(r.vs_index_pp)}
                              for s, r in zip(sec_rz.index, sec_rz.itertuples(index=False))]
        rz_things = day_notes.three_things(
            portfolio=day_notes.portfolio_note(rz_tot, rz_by_sym),
            market=day_notes.market_note(rz_moves),
            events=day_notes.events_note(rz_soon),
            alerts=day_notes.alerts_note((rz_tot or {}).get("alerts"), rz_by_sym),
            sector=day_notes.sector_note(rz_sectors),
            news=day_notes.news_note(rz_news))
        with rz_top:
            st.markdown("### 🧭 3 lucruri de știut azi")
            if rz_things:
                st.markdown(compact_html(
                    '<div style="border:1px solid #30363D; border-radius:8px; padding:10px 14px; margin-bottom:6px;">'
                    + "".join(f'<div style="margin:4px 0;"><b>{i}.</b> {md_bold_html(t["text"])}</div>'
                              for i, t in enumerate(rz_things, 1))
                    + "</div>"), unsafe_allow_html=True)
            else:
                st.info("Nu sunt destule date acum pentru rezumat (cotațiile sau registrul nu au putut fi citite).")
            st.caption("Text din reguli fixe, descriptiv (analytics/day_notes.py): 1) portofoliul în ultima ședință; 2) indicele "
                       "cu cea mai mare mișcare în σ dintre BET (prin ETF-ul TVBETETF), STOXX 600 și S&P 500; 3) primul disponibil dintre calendarul "
                       "macro SUA de azi/mâine, pozițiile tale peste 2σ, sectorul S&P 500 cu abaterea cea mai mare (≥ 1 pp) și "
                       "cea mai nouă știre despre pozițiile tale (lista în tabul „Știri”). Detaliile sunt mai jos.")
            st.markdown("---")
        _rz_timer.caption(f"⏱️ Pagina a rulat complet în {format_ro(time.perf_counter() - _rz_t0, 1)} s, la {now_ro():%H:%M:%S}.")

    # ==================================================
    # 7. SCANNER VOLUM (RVOL) - NOU
    # ==================================================
    elif sectiune == "7. Scanner Volum":
        st.title("📡 Scanner Volum Relativ (RVOL)")
        st.markdown("""
        Acest modul identifică **anomaliile de volum**. 
        Un RVOL (Relative Volume) mai mare de **1,5** indică un interes instituțional sau o știre importantă.
        """)
        
        # Slider pentru sensibilitate (Default 1.5)
        threshold = st.slider("Arată doar acțiunile cu Volum de 'X' ori mai mare decât media:", 
                            min_value=1.2, max_value=5.0, value=scanner_calc.RVOL_DEFAULT, step=0.1)
        st.caption(f"Lichiditate: la BVB intră doar titlurile cu valoare tranzacționată ≥ "
                   f"{briefing_calc.money_short(briefing_calc.MIN_VALUE_RON, 'RON')} în ședința evaluată (coloana „Valoare” = preț × volum, în moneda din coloana „Monedă”; ca în "
                   f"Rezumatul Zilei). La SUA și Europa nu există prag de lichiditate: listele au doar companii mari.")

        # Listele de scanare: din listele unice (data/universes.py), câte un grup pe secțiune + „toate” pe piață.
        # BVB rămâne separat de Europa; ETF-ul BET (TVBETETF.RO) se scanează împreună cu BVB, ca înainte.
        tickers_map = {f"🇷🇴 BVB – toate ({len(universes.symbols('bvb')) + 1})": [BRIEFING_BET_PROXY] + universes.symbols("bvb")}
        for mk, flag in (("us", "🇺🇸 SUA"), ("eu", "🇪🇺 Europa")):
            tickers_map[f"{flag} – toate ({len(universes.symbols(mk))})"] = universes.symbols(mk)
            for sec, syms in universes.groups(mk).items():
                tickers_map[f"{flag} – {sec} ({len(syms)})"] = syms
        
        # Funcție internă de calcul RVOL + AI Isolation Forest
        def get_rvol_data(ticker_list, include_partial=False):
            from ai_engine import detect_volume_anomaly_ai # Importăm funcția nouă AI
            try:
                # Descărcăm date pe 3 luni pentru a avea destul istoric de "învățare" pt ML
                data = yf.download(ticker_list, period="3mo", group_by='ticker', progress=False, auto_adjust=True)
                closes, volumes = field_frame(data, "Close", ticker_list), field_frame(data, "Volume", ticker_list)
                results = []
                states = {}   # starea bursei, o dată pe bursă: ședința de azi e încă în desfășurare?
                
                for t in ticker_list:
                    try:
                        if t not in closes.columns or t not in volumes.columns: continue
                        hk = scanner_calc.hours_key(t)
                        if hk not in states:
                            states[hk] = briefing_calc.market_state(hk, now_ro())
                        df_t = pd.DataFrame({"Close": closes[t], "Volume": volumes[t]})
                        # Preț și volum aliniate pe aceeași zi; media pe 20 de ședințe cu volum (fără zilele cu 0)
                        # În timpul ședinței: implicit ultima ședință încheiată; opțional bara în curs, marcată „parțial”
                        rs = scanner_calc.rvol_session(df_t, lambda d, s=states[hk]: briefing_calc.session_is_partial(d, s),
                                                       include_partial)
                        if rs is None: continue
                        curr_p, curr_vol, avg_vol_20 = rs["close"], rs["volume"], rs["avg_volume"]
                        # Lichiditate: doar la BVB, valoare ≥ 50.000 lei în ședința evaluată (ca în Rezumat); SUA/UE fără prag
                        if not scanner_calc.liquidity_ok(t, curr_p, curr_vol, avg_vol_20): continue
                        
                        rvol = rs["rvol"]
                        change_pct = rs["change_pct"]
                        
                        # Isolation Forest pe aceeași ședință ca RVOL, antrenat doar pe zilele dinaintea ei (nevalidat)
                        is_anomaly = detect_volume_anomaly_ai(df_t[df_t.index <= rs["date"]])
                        
                        results.append({
                            "Simbol": t.replace('.RO', ''),
                            "Ședința": f"{pd.Timestamp(rs['date']):%d.%m}" + (" ⏳ parțial" if rs["partial"] else ""),
                            "Preț": curr_p,
                            "Variație %": change_pct,
                            "Monedă": briefing_calc.quote_currency(t) or "N/A",
                            "Volum": curr_vol,
                            "Valoare": curr_p * curr_vol,
                            "Volum Mediu (20z)": avg_vol_20,
                            "RVOL": rvol,
                            "Anomalie model (nevalidat)": "da" if is_anomaly else ("N/A" if is_anomaly is None else "-"),
                            # Praguri fixe, netestate istoric (analytics/scanner.py: status_label)
                            "Status": scanner_calc.status_label(rvol, change_pct)
                        })
                    except Exception as exc:  # înainte: except: gol
                        logger.debug("get_rvol_data: %s", exc)
                        continue
                    
                return pd.DataFrame(results)
            except Exception as exc:  # înainte: except: gol
                logger.warning("get_rvol_data: %s", exc)
                return pd.DataFrame()
                        

        # --- SELECTOR DE PIAȚĂ (DROPDOWN în loc de TABURI pentru eficiență) ---
        market_choice = st.selectbox("Alege Piața/Sectorul de scanat:", list(tickers_map.keys()))
        
        # Extragem tickerii pentru selecția făcută
        selected_tickers = tickers_map[market_choice]
        
        include_partial = st.checkbox("Include ședința în curs (volum parțial)", value=False, key="scanner_partial",
                                      help="Implicit, RVOL se calculează pe ultima ședință ÎNCHEIATĂ: în timpul zilei volumul e "
                                           "parțial, iar împărțit la media pe 20 de ședințe întregi ar ieși prea mic. Bifat: "
                                           "se folosește bara de azi, marcată „⏳ parțial” în coloana „Ședința”.")
        col_scan_btn, col_info = st.columns([1, 3])
        
        with col_scan_btn:
            run_scan = st.button(f"🔎 Scanează {len(selected_tickers)} companii", type="primary")
            
        with col_info:
            st.caption(f"Se vor analiza volumele pentru: {', '.join(selected_tickers[:5])} ... și altele.")

        if run_scan:
            with st.spinner(f"Analizăm {market_choice}... (Poate dura 10-20 secunde)"):
                df_res = get_rvol_data(selected_tickers, include_partial)
                
                if not df_res.empty:
                    # Filtrare după Threshold-ul ales de user
                    df_filtered = df_res[df_res['RVOL'] >= threshold].copy()
                    
                    # Sortare descrescătoare după RVOL
                    df_filtered = df_filtered.sort_values(by="RVOL", ascending=False)
                    
                    if not df_filtered.empty:
                        # --- FUNCȚIE DE COLORARE PROFESIONALĂ (ACTUALIZATĂ CU AI) ---
                        def style_scanner_rows(row):
                            # Setăm culoarea de bază a rândului
                            if "BREAKOUT" in row['Status']:
                                styles = ['background-color: rgba(63, 185, 80, 0.3); font-weight: bold'] * len(row)
                            elif "ACUMULARE" in row['Status']:
                                styles = ['background-color: rgba(63, 185, 80, 0.1)'] * len(row)
                            elif "PANIC" in row['Status']:
                                styles = ['background-color: rgba(248, 81, 73, 0.2)'] * len(row)
                            else:
                                styles = [''] * len(row)
                            return styles

                        # --- AFIȘARE TABEL FĂRĂ INDEX ȘI CU STIL ---
                        st.success(f"Găsit: {len(df_filtered)} companii cu volum neobișnuit în {market_choice}.")
                        
                        st.dataframe(
                            df_filtered.style.apply(style_scanner_rows, axis=1).format({
                                "Preț": price_ro,   # 4 zecimale sub 10 (HAI 0,2350, nu 0,23)
                                "Variație %": ro_spec('+.2f', '%'),
                                "Volum": ro_spec(',.0f'),
                                "Valoare": ro_spec(',.0f'),
                                "Volum Mediu (20z)": ro_spec(',.0f'),
                                "RVOL": ro_spec('.2f', 'x')
                            }),
                            width='stretch', 
                            height=600,
                            hide_index=True  # <--- ACEASTA ESTE LINIA CARE ELIMINĂ NUMERELE DIN STÂNGA
                        )
                        
                        st.caption("🟢 **Verde aprins:** BREAKOUT (RVOL > 2 și variație > +1,5%) | 🟢 **Verde pal:** ACUMULARE "
                                   "(RVOL > 1,2 și variație pozitivă) | 🔴 **Roșu:** PANIC SELL (RVOL > 2 și variație < −1,5%). "
                                   "Etichete descriptive cu praguri fixe, **netestate istoric**: nu s-a măsurat ce face prețul după ele.")
                        st.caption("„Anomalie model (nevalidat)”: Isolation Forest pe randament, volum și volatilitatea pe 5 zile, "
                                   "antrenat pe ~3 luni dinaintea ședinței evaluate. Pragul e fixat ca ~5% din zilele de antrenare "
                                   "să fie anomalii, prin construcție; modelul nu a fost validat pe date nevăzute, deci nu e un "
                                   "semnal mai puternic decât RVOL. N/A = istoric prea scurt.")
                    else:
                        st.info(f"Nicio acțiune din {market_choice} nu depășește pragul de {threshold}x azi.")
                else:
                    st.warning("Eroare la preluarea datelor. Yahoo Finance ar putea limita cererile.")
    # ==================================================
    # 8. WATCHLIST (logica în analytics/watchlist.py)
    # ==================================================
    elif sectiune == "8. Watchlist":
        st.title("Lista de Urmărire (Watchlist)")
        st.markdown("Monitorizează acțiunile pe care vrei să le cumperi când prețul scade.")
        if st.session_state.get("wl_msg"):
            st.success(st.session_state.pop("wl_msg"))

        # --- FORMULAR ADĂUGARE ---
        with st.expander("➕ Adaugă Alertă Nouă", expanded=False):
            with st.form("wl_form"):
                c1, c2, c3 = st.columns([1, 1, 2])
                s_wl = c1.text_input("Simbol (ex: TSLA)").upper()
                # 4 zecimale: la BVB sunt acțiuni sub 1 leu (0,4950); formatul implicit are 2
                p_wl = c2.number_input("Preț Țintă (Target)", min_value=0.0, step=0.01, format="%.4f")
                n_wl = c3.text_input("Notă (ex: Suport major, aștept earnings)")
                st.caption("Simbolul se caută ca în Analiză: întâi cel scris, apoi cu .RO. La BVB scrie-l cu .RO când "
                           "există și o acțiune americană cu același simbol (EL = Estée Lauder, EL.RO = Electrica). "
                           "Ținta e în moneda cotației, cea din coloana Monedă (la Londra poate fi GBp = pence, GBP, USD sau EUR).")
                
                if st.form_submit_button("Adaugă în Listă"):
                    if s_wl.strip() and p_wl > 0:
                        ok, msg = add_to_watchlist(s_wl, p_wl, n_wl)
                        if ok:
                            st.session_state["wl_msg"] = msg  # afișat după rerun (altfel mesajul dispare)
                            st.rerun()
                        st.warning(msg)
                    else:
                        st.warning("Introdu un simbol și un preț valid.")

        # --- AFIȘARE TABEL ---
        df_wl = load_watchlist()
        for problem in df_wl.attrs.get("problems", []):
            st.warning(problem)
        
        if not df_wl.empty:
            # 1. Ultimul preț pentru fiecare simbol (close_frame, cache 15 minute)
            tickers_list = df_wl['Symbol'].unique().tolist()
            with st.spinner("Actualizăm prețurile..."):
                quotes = get_watchlist_quotes(tickers_list)
            with st.spinner("Calculăm momentul..."):
                moments = get_watchlist_moments(tickers_list)

            # Scorul fundamental: la buton (decizia lui Daniel, 09.11 seara), ~8 cereri Yahoo pe acțiune SUA/UE.
            # Rezultatele stau 24 h în cache pe simbol; butonul doar pornește afișarea în sesiunea curentă.
            fund_cells, fund_times, fund_failed, fund_note = {}, [], [], None
            b1, b2, _ = st.columns([1.4, 1.4, 2])
            if b1.button("📊 Calculează scorurile fundamentale", key="wl_fund_btn"):
                st.session_state["wl_fund_on"] = True
            if st.session_state.get("wl_fund_on") and b2.button("🔄 Recalculează (golește cache-ul de 24 h)", key="wl_fund_clear"):
                _watchlist_global_fundamental.clear()
            if st.session_state.get("wl_fund_on"):
                ro_syms = [s for s in tickers_list if str(s).upper().endswith(".RO")]
                other_syms = [s for s in tickers_list if s not in ro_syms]
                if ro_syms:
                    with st.spinner("Citesc fișierele din „Analiză companii” (prima dată durează până la ~30 de secunde)..."):
                        bvb_cells, fund_note = watchlist_bvb_fundamentals(
                            ro_syms, {s: (quotes.get(s) or {}).get("price") for s in ro_syms})
                    fund_cells.update(bvb_cells)
                bar = st.progress(0.0, text="Scoruri fundamentale SUA/UE...")
                for k, sym in enumerate(other_syms, 1):
                    bar.progress(k / max(len(other_syms), 1), text=f"Scoruri fundamentale SUA/UE: {sym} ({k} din {len(other_syms)})")
                    try:
                        res = _watchlist_global_fundamental(sym)
                        fund_cells[sym] = res["na"] or watchlist_calc.global_fundamental_cell(res["score"], res["label"])
                        fund_times.append(res["at"])
                    except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError,
                            yf.exceptions.YFException) as e:
                        logger.warning(f"Watchlist, scor fundamental {sym}: {e}")
                        fund_cells[sym] = "N/A (Yahoo)"
                        fund_failed.append(sym)
                bar.empty()

            # 2. Construim tabelul final
            today_ro = now_ro()
            states = {}   # starea fiecărei burse (BVB, NYSE, XETRA), calculată o dată
            display_rows = []
            for index, row in df_wl.iterrows():
                sym = row['Symbol']
                note = row['Notes']
                quote = quotes.get(sym) or {}
                hk = scanner_calc.hours_key(sym)
                if hk not in states:
                    states[hk] = briefing_calc.market_state(hk, today_ro)
                partial = briefing_calc.session_is_partial(quote.get("date"), states[hk])
                
                # Preț sau țintă lipsă rămân lipsă („N/A”), nu 0; distanța e față de țintă, ca în cardul din analiză.
                curr, target, dist_pct, _ = watchlist_status(quote.get("price"), row['TargetPrice'])
                display_rows.append({
                    "Simbol": sym,
                    "Monedă": watchlist_currency(sym),
                    "Ultimul preț": curr,
                    "Data prețului": watchlist_calc.price_date_text(quote.get("date"), today_ro, partial),
                    "Preț Țintă 🎯": target,
                    "Distanță (%)": dist_pct,
                    "Status": watchlist_calc.status_label(curr, target),
                    "Moment": moments.get(sym, "N/A"),
                    "Scor fundamental": fund_cells.get(sym, watchlist_calc.NOT_COMPUTED),
                    "Data țintei": row['Date'].strftime("%d.%m.%Y") if pd.notna(row['Date']) else "N/A",
                    "Notă": note,
                    "_zone": target_zone(curr, target),
                })
            
            df_res = pd.DataFrame(display_rows)
            no_price = [s for s in tickers_list if s not in quotes]
            if no_price:
                st.caption(f"Fără preț Yahoo: {', '.join(no_price)}. La BVB simbolul are sufixul .RO (ex: TLV.RO): "
                           "șterge rândul și adaugă-l din nou din formular, care găsește singur simbolul corect.")
            pence = df_res.loc[df_res["Monedă"].str.startswith("GBp"), "Simbol"].tolist()
            if pence:
                st.caption(f"În pence (GBp): {', '.join(pence)}. Ținta acestor rânduri se scrie tot în pence (1 GBP = 100 GBp).")

            # 3. Stilizare și Afișare: verde la/sub țintă, galben aproape (aceleași culori ca în cardul din Analiză)
            zone_style = {"sub": 'background-color: rgba(63, 185, 80, 0.2); font-weight: bold',
                          "aproape": 'background-color: rgba(210, 153, 34, 0.18)'}
            def highlight_zone(row):
                return [zone_style.get(row['_zone'], '')] * len(row)

            st.dataframe(
                df_res.style.apply(highlight_zone, axis=1)
                .format({"Ultimul preț": price_ro, "Preț Țintă 🎯": price_ro, "Distanță (%)": ro_spec('+.2f', '%')}, na_rep="N/A"),
                width='stretch',
                height=500,
                column_config={
                    "_zone": None,
                    "Monedă": st.column_config.Column(help="Moneda cotației. SUA (fără sufix) → USD, BVB (.RO) → RON; la celelalte burse "
                                                           "moneda vine de la Yahoo, pentru că aceeași bursă are linii în monede diferite "
                                                           "(la Londra: GBp = pence, GBP, USD, EUR). „(dedusă)” = Yahoo n-a răspuns, moneda "
                                                           "e ghicită după sufix. Ținta e în aceeași monedă ca prețul."),
                    "Ultimul preț": st.column_config.Column(help="Ultima închidere Yahoo (preț real, neajustat cu dividende). În timpul "
                                                                 "ședinței e prețul de azi, cu întârziere de ~15 minute."),
                    "Data prețului": st.column_config.Column(help="Ziua prețului. ⏳ = ședință în curs (prețul se mai schimbă). "
                                                                  f"⚠️ = mai vechi de {watchlist_calc.STALE_BDAYS} zile lucrătoare "
                                                                  "(titlu netranzacționat sau suspendat; sărbătorile nu sunt scăzute)."),
                    "Preț Țintă 🎯": st.column_config.Column(help="Prețul la care vrei să cumperi, scris de tine în foaia watchlist."),
                    "Distanță (%)": st.column_config.Column(help="(Ultimul preț − țintă) ÷ țintă × 100. Plus = prețul e peste țintă "
                                                                 "(cât mai are de scăzut, raportat la țintă); minus = sub țintă."),
                    "Status": st.column_config.Column(help="✅ prețul e la sau sub țintă; 🟡 peste țintă cu mai puțin de "
                                                           f"{NEAR_TARGET_PCT:.0f}%; ⏳ peste. Alertă față de ținta ta, nu o recomandare: "
                                                           "verifică fundamentalele în Analiză înainte de a cumpăra."),
                    "Moment": st.column_config.Column(help="Sub-scorul tehnic 0–100, același ca în Analiză Companie: media trendului "
                                                           "(câte din: preț > SMA 50, preț > SMA 200, SMA 50 > SMA 200) și a momentumului "
                                                           "12-1 (−30% → 0, +30% → 100), pe prețuri ajustate; RSI nu intră. BVB: eticheta pe "
                                                           "pragurile testate 2012–2025 (≥ 80 FAVORABIL, 40–80 NEUTRU, < 40 NEFAVORABIL). "
                                                           "Celelalte piețe: fără verdict (testul pe SUA nu a găsit putere de predicție). "
                                                           "N/A = istoric sub 50 de ședințe sau fără date. Nu se adună cu scorul fundamental."),
                    "Scor fundamental": st.column_config.Column(help="Scorul companiei, același ca în Analiză Companie. BVB: media "
                                                                     "percentilelor E/P, B/P, randament dividend și Piotroski simplificat față "
                                                                     "de companiile din fișierele „Analiză companii”, cu treapta testată. SUA/UE: "
                                                                     "evaluare, calitate, sănătate (rentabilitate la financiare), praguri netestate. "
                                                                     "— = necalculat (butonul de deasupra tabelului); N/A (ETF) = nu se aplică; "
                                                                     "N/A (Yahoo) = date incomplete acum, se reîncearcă la următorul calcul."),
                    "Data țintei": st.column_config.Column(help="Când ai pus ținta (coloana Date din foaie). O țintă veche poate să nu mai "
                                                                "corespundă fundamentalelor actuale. N/A = rând adăugat înainte de coloana Date."),
                },
                hide_index=True 
            )
            st.caption("Status = prețul față de ținta ta (o alertă de preț, nu o recomandare). Distanța = (preț − țintă) ÷ țintă: "
                       f"plus = peste țintă. „Aproape” = sub {NEAR_TARGET_PCT:.0f}% peste țintă, același prag ca în cardul „Țintă intrare”. "
                       "Prețul e ultima închidere Yahoo; în timpul ședinței, prețul de azi cu întârziere (⏳).")
            
            if fund_times:
                st.caption(f"Scorurile fundamentale SUA/UE din cache-ul de 24 h, calculate între {min(fund_times):%d.%m %H:%M} și "
                           f"{max(fund_times):%d.%m %H:%M}: evaluarea folosește prețul de atunci. „🔄 Recalculează” le reface.")
            if fund_failed:
                st.caption(f"Fără scor acum (Yahoo a răspuns incomplet): {', '.join(fund_failed)}. Se reîncearcă la următorul calcul.")
            if fund_note:
                st.caption(f"Scorul BVB indisponibil: {fund_note}")

            # Buton ștergere
            with st.expander("🗑️ Șterge din listă"):
                del_sym = st.selectbox("Alege simbol de șters:", tickers_list)
                if st.button("Șterge"):
                    ok, msg = remove_from_watchlist(del_sym)
                    if ok:
                        st.session_state["wl_msg"] = msg  # afișat după rerun
                        st.rerun()
                    st.error(msg)

        elif not df_wl.attrs.get("problems"):
            st.info("Nu ai nicio acțiune în Watchlist. Folosește formularul de sus.")

    elif sectiune == "10. Idei și Strategii":
        render_ideas_page()

    elif sectiune == "11. Fonduri de Pensii":
        render_pilon2_page()

    elif sectiune == "12. Fonduri SUA":
        render_sec_funds_page()

    elif sectiune == "9. Glosar Economic":
        glosar.render(st)

if __name__ == "__main__":
    main()

