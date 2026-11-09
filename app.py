import streamlit as st
import logging
logger = logging.getLogger(__name__)
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import feedparser
from data.news import filter_company_news, future_note, holdings_news, mark_future, news_card_html
from data import fred_calendar as fred_cal
from data import sheet_view
from data import universes
from ui.card_help import FUND_HELP, fund_help
import os
import time
from datetime import datetime, timedelta
from textblob import TextBlob
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
from scipy.stats import norm
from collections import deque
from threading import Lock

# Module proprii: funcții de calcul pure, acoperite de teste (vezi tests/)
from analytics.technical import atr_trailing_stop, rsi_wilder, macd as macd_lines
from analytics.macro import yoy_pct, real_rate, market_of, monthly_spread, same_day_spread, tbill_discount_to_bey
from analytics import macro as macro_calc
from analytics import markets as markets_calc
from analytics import rates as rates_calc
from analytics import stress as stress_calc
from analytics import briefing as briefing_calc
from analytics import day_notes
from analytics import portfolio_day as pday_calc
from analytics import scanner as scanner_calc
from analytics.portfolio import value_positions, portfolio_curve as build_portfolio_curve, positions_detail
from analytics import portfolio as pf_calc
from analytics import fundamentals as fund
from analytics import ledger as ledger_calc
from analytics import simulation
from analytics import options as opt_calc
from data import portfolio_sheet
from data import iv_history
from data import broker_import
import zipfile
from analytics.risk import beta_benchmark, beta_weekly, beta_corr_weekly, jensen_alpha, performance_risk, common_day_correlation
from analytics.risk import CORR_HIGH, CORR_MID, CORR_BANDS, correlation_band
from analytics.audit import get_sector_benchmarks, generate_advanced_audit_v2
from analytics.scoring import technical_subscore, technical_components, moment_label, rsi_description, explain_moment
from analytics import global_fundamental as gfund
from analytics.bvb_fundamental import current_indicators, cross_section, verdict_banner, score_rank, explain_fundamental, piotroski_value_text, final_verdict, GN_LIMIT, TEST_NOTE
from data.company_files import load_company, is_bank_template, ticker_from_title, period_label, FIELD_LABELS as COMPANY_FIELD_LABELS
from analytics.peers import PEERS, METRICS as PEER_METRICS, peer_region, peer_list, peer_medians, versus_median
from analytics.peers import BVB_REGION, bvb_sector, bvb_sector_peers, bvb_regional_peers, peer_symbols, sheet_peer_row, format_peer_value, split_financial_peers
from data.bvb_sheet import parse_bvb_sheet, bvb_symbol, reprice as reprice_bvb, unmapped_info_keys
from data import bnr as bnr_data
from data.helpers import format_ro, num, close_frame, field_frame, slice_window, now_ro, struct_time_utc_to_ro, smart_to_float, parse_ecb_csv, parse_ecb_series, parse_eurostat_monthly, positive_or_none, entry_target_view, scale_number, watchlist_status, ro, ro_spec, md_bold_html, text_frame

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

# Wrapper pentru a putea rula cod asincron în interiorul Streamlit (care e sincron)
def get_fast_live_prices(tickers):
    """Prețuri curente: dict simbol -> preț, sau None pentru simbolurile care nu au putut fi citite."""
    if not tickers: return {}
    try:
        return asyncio.run(get_all_portfolio_prices(tickers))
    except (RuntimeError, httpx.HTTPError) as e:
        logger.warning(f"prețuri live indisponibile: {e}")
        return {t: None for t in tickers}

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

# --- CSS MODERNIZAT (UI PREMIUM) ---
st.markdown("""
    <style>
    /* Stil general aplicație */
    .stApp { background-color: #0E1117; }
    
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
        background-color: #21262D;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #30363D;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetricLabel"] { font-size: 14px; color: #8B949E; }
    div[data-testid="stMetricValue"] { font-size: 24px; font-weight: 600; color: #FFFFFF; }

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
        display: flex; justify-content: space-between; font-size: 10px; color: #8B949E; margin-top: 5px;
    }

    /* Sentiment Tags */
    .impact-poz { color: #3FB950; font-weight: bold; background: rgba(63, 185, 80, 0.1); padding: 2px 6px; border-radius: 4px; }
    .impact-neg { color: #F85149; font-weight: bold; background: rgba(248, 81, 73, 0.1); padding: 2px 6px; border-radius: 4px; }
    .impact-neu { color: #8B949E; font-weight: bold; background: rgba(139, 148, 158, 0.1); padding: 2px 6px; border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)

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
    Se aplică axelor, etichetelor și casetelor de hover; nu schimbă datele graficului."""
    fig.update_layout(separators=",.")
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

def calculate_investment_rating_pro(info, inst_pct, rvol, spread_val, mos_val, spread_label=None):
    score = 50
    details = []
    
    # 1. ANALIZA SMART MONEY (None = date lipsă: pilonul nu se punctează)
    if inst_pct is None:
        details.append("ℹ️ **Smart Money:** acționariatul nu este disponibil. Pilon neinclus.")
    elif inst_pct > 70:
        score += 15
        details.append("✅ **Smart Money:** Deținere de elită (>70%). Suport instituțional masiv.")
    elif inst_pct > 50:
        score += 10
        details.append("✅ **Smart Money:** Majoritate instituțională. Stabilitate ridicată.")
    elif inst_pct < 20:
        score -= 15
        details.append("⚠️ **Smart Money:** Deținere instituțională slabă. Risc de volatilitate retail.")

    # 2. ANALIZA EVALUARE
    if mos_val is None:
        details.append("ℹ️ **Evaluare:** DCF indisponibil sau neaplicabil acestui emitent. Pilon neinclus.")
    elif mos_val > 25:
        score += 15
        details.append(f"✅ **Evaluare:** Marjă de siguranță excelentă ({ro(mos_val):.1f}%). Preț subevaluat.")
    elif mos_val < -10:
        score -= 15
        details.append(f"🚨 **Evaluare:** Supraevaluare semnificativă. Risc ridicat de corecție.")

    # 3. ANALIZA MACRO
    if spread_val is None:
        details.append(f"ℹ️ **Macro:** {spread_label or 'curba randamentelor pieței indisponibilă'}. Pilon neinclus.")
    elif spread_val < 0:
        score -= 20
        details.append(f"🚨 **Macro:** Curba 10 ani − 3 luni inversată ({ro(spread_val):+.2f} pp; {spread_label}). "
                       "Risc de recesiune pe piața acțiunii.")
    else:
        score += 5
        details.append(f"✅ **Macro:** Curba 10 ani − 3 luni normală ({ro(spread_val):+.2f} pp; {spread_label}).")

    # 4. SĂNĂTATE FINANCIARĂ
    roe = num(info, 'returnOnEquity')
    if roe is not None and roe > 0.20:
        score += 10
        details.append(f"🚀 **Eficiență:** ROE excepțional ({ro(roe*100):.1f}%).")
    
    debt = num(info, 'debtToEquity')
    if debt is not None and debt > 150:
        score -= 10
        details.append("🚩 **Datorii:** Grad de îndatorare ridicat.")
    
    return max(0, min(100, score)), details

def get_score_highlights(data):
    highlights = []
    
    # Verificare Profitabilitate
    if data['roe'] > 0.15:
        highlights.append("✅ Profitabilitate: ROE excelent susține creșterea organică.")
    
    # Verificare Marjă de Siguranță
    if data['margin_of_safety'] < 0.10:
        highlights.append("⚠️ Evaluare: Marjă de siguranță redusă sub nivelul ideal de 20%.")
        
    # Verificare Balene (Instituționali)
    if data['inst_ownership'] > 0.50:
        highlights.append("✅ Smart Money: Suport instituțional solid detectat.")
        
    return highlights

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


BVB_HELP = {
    "scor": ("Media percentilelor a patru indicatori față de celelalte companii BVB din fișierele tale "
             "(P/E, P/BV, randament dividend, Piotroski). 100 = cea mai bună pe toți, 0 = cea mai slabă. "
             "Treimea de sus = FAVORABIL, mijlocul = NEUTRU, treimea de jos = NEFAVORABIL. "
             "Test 2017–2026: treimea de sus a bătut media BVB cu +6,0% la 6 luni (56% cazuri pozitive), "
             "treimea de jos a rămas cu −7,0% în urmă (33% pozitive)."),
    "pe": ("P/E TTM = capitalizare (preț curent × acțiuni din ultimul raport) / profit net pe ultimele 12 luni, "
           "construit din cifrele cumulate din fișier. Mai mic = mai ieftin. Test: ieftinul a bătut scumpul (IC +0,17)."),
    "pbv": ("P/BV = capitalizare / capital propriu din ultimul raport. Mai mic = mai ieftin raportat la activul net. "
            "Test: IC +0,13, mai slab decât P/E."),
    "dy": ("Randament dividend = profitul distribuit din ultimul an / capitalizarea de azi. "
           "Test: dividendul mare a bătut dividendul mic (IC +0,13)."),
    "pio": ("Piotroski simplificat: 6 criterii, ultimul raport față de același raport de acum un an: ROA pozitiv, "
            "ROA în creștere, lichiditate curentă în creștere, datorii pe termen lung/active în scădere, fără emisiune "
            "de acțiuni, rotația activelor în creștere. Fără criteriile de flux de numerar (nu sunt în fișiere). "
            "Cel mai puternic indicator în test (IC +0,28)."),
    "gn": ("GN = P/E × P/BV, pragul lui Graham 22,5. Test: GN ≤ 22,5 → +2,9% față de media BVB la 6 luni "
           "(50% pozitive); GN > 22,5 → −3,6% (39% pozitive)."),
    "moment": ("Media dintre trend (preț peste SMA 50 și SMA 200, SMA 50 peste SMA 200) și momentum 12-1 luni. "
               "Test BVB 2012–2025: ≥ 80 → +6,1% față de media BVB la 6 luni (54% pozitive); "
               "< 40 → −7,2% (32% pozitive). RSI nu intră: în test „supravândut” a continuat să scadă."),
    "rsi": "RSI 14 (Wilder), doar descriptiv. În testul pe BVB nu a dat semnal de intrare.",
    "roe": "ROE TTM = profit net pe 12 luni / capital propriu. În test NU a prezis randamentul (IC ≈ 0): informație, fără puncte.",
    "marja": "Marja netă TTM = profit net / cifră de afaceri pe 12 luni. În test NU a prezis randamentul: informație.",
    "indatorare": ("Datorii TOTALE (inclusiv furnizori, provizioane, venituri în avans) / capital propriu, din ultimul raport "
                   "din fișier. Diferă de cardul „Datorii/Capital” din Yahoo, care ia doar datoria financiară. "
                   "În test NU a prezis randamentul: informație de risc."),
}


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
    t1.metric("Moment", "N/A" if moment is None else f"{ro(moment):.0f}/100", m_label, delta_color="off", delta_arrow="off", help=BVB_HELP["moment"])
    trend = comp["trend"]
    t2.metric("Trend SMA 50/200", "N/A" if trend is None else f"{ro(trend * 3):.0f} din 3" if len(close.dropna()) >= 200 else f"{ro(trend):.0f} din 1",
              help="Câte din: preț > SMA 50, preț > SMA 200, SMA 50 > SMA 200 (sub 200 de ședințe: doar preț > SMA 50).")
    mom_raw = comp["raw"]["momentum_12_1"]
    t3.metric("Momentum 12-1 luni", _bvb_fmt(mom_raw, "pct"),
              help="Randamentul de acum 12 luni până acum o lună (ultima lună exclusă), pe date calendaristice.")
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
            "dy": None if pd.isna(me["DY"]) else float(me["DY"]), "dy_year": me.get("raport_anual"), "price": price}
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
                   else f"{ro(res['score']):.1f}/100 · {n_ok} din {len(res['pillars'])} piloni cu date · descriptiv")
    moment_detail = "N/A" if moment is None else f"tehnic {ro(moment):.0f}/100, descriptiv"

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

    pillar_help = {
        "evaluare": ("Media componentelor cu date. Nefinanciare: DCF invers (creșterea FCF pe care o cere prețul; "
                     "≤ 0% → 100, ≥ 20% → 0) și randamentul profitului E/P față de randamentul AAA (criteriul lui "
                     "Graham: ≤ 1× → 0, ≥ 2× → 100). GN = P/E × P/BV apare doar informativ. "
                     "Bănci și asigurători: P/BV față de P/BV justificat (ROE − g) / (ke − g); raport ≤ 0,7 → 100, ≥ 1,3 → 0."),
        "calitate": "Piotroski F-Score: criterii îndeplinite / criterii evaluabile (minimum 6 din 9), ultimul an fiscal față de cel precedent.",
        "sanatate": ("Media: zona Altman (sigură 100, gri 50, dificultate 0), datorie netă / EBITDA (≤ 1 → 100, ≥ 4 → 0), "
                     "acoperirea dobânzii EBIT / dobânzi (≤ 1,5 → 0, ≥ 8 → 100)."),
        "rentabilitate": "ROE față de costul capitalului propriu ke = rf + beta × 5%: −5 pp → 0, +5 pp → 100.",
    }
    cols = st.columns(len(res["pillars"]))
    for col, p in zip(cols, res["pillars"]):
        col.metric(p["label"], "N/A" if p["score"] is None else f"{ro(p['score'] * 100):.0f}/100",
                   help=pillar_help[p["key"]] + " Test istoric SUA 2022–2026: fără putere de predicție; "
                   "pragurile sunt orientative.")

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
    st.caption("„Decizie Master AI” de mai jos reia acest scor și momentul, fără să le recalculeze.")
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

def run_monte_carlo_sim(portfolio_curve, days_ahead=252, simulations=1000):
    """
    Rulează o simulare stocastică pentru a prezice evoluția portofoliului.
    """
    if portfolio_curve is None or len(portfolio_curve) < 10:
        return None, 0
    
    # Calculăm randamentele logaritmice pentru stabilitate matematică
    returns = np.log(portfolio_curve / portfolio_curve.shift(1)).dropna()
    mu = returns.mean()
    var = returns.var()
    # Drift-ul reprezintă direcția medie ajustată cu volatilitatea
    drift = mu - (0.5 * var)
    stdev = returns.std()
    
    # Generăm șocuri aleatorii (Z ~ N(0,1))
    # Matrice de (zile x simulări)
    daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(days_ahead, simulations)))
    
    # Proiecția prețului
    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = portfolio_curve.iloc[-1]
    
    for t in range(1, days_ahead):
        price_paths[t] = price_paths[t-1] * daily_returns[t]
        
    return price_paths, portfolio_curve.iloc[-1]

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

def get_company_news_rss(symbol, names=None, limit=7):
    """Știrile Yahoo (RSS) pe simbol. Cu `names` (numele companiei din `info`), se păstrează doar
    știrile care menționează emitentul (data/news.py): fluxul pe simbol aduce des știri despre alte
    companii. Fără `names` (indici: ^GSPC, ^IXIC) lista rămâne nefiltrată."""
    rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
    news_list = []
    try:
        feed = feedparser.parse(rss_url)
        if not feed.entries: return []
        for entry in feed.entries[:30]:
            dt = parse_date(entry)
            news_list.append({
                "title": entry.title,
                "link": entry.link,
                "summary": getattr(entry, "summary", ""),
                "publisher": "Yahoo Finance",
                "date_str": dt.strftime("%Y-%m-%d %H:%M")
            })
    except Exception as exc:  # înainte: except: gol
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

def calculate_margin_of_safety(current_price, fair_value):
    """Calculează marja de siguranță între prețul actual și valoarea intrinsecă."""
    if fair_value <= 0: return 0, "N/A"
    
    # Diferența procentuală
    mos = (fair_value - current_price) / fair_value
    
    if mos > 0.30:
        verdict = "🚀 Deep Value: Acțiunea este sever subevaluată. Marjă de siguranță excelentă."
    elif mos > 0.10:
        verdict = "✅ Fair Value: Preț rezonabil. Există o marjă de siguranță acceptabilă."
    elif mos > -0.10:
        verdict = "⚖️ Evaluare Corectă: Prețul pieței reflectă valoarea reală. Fără marjă de siguranță."
    else:
        verdict = "⚠️ Supraevaluare: Prețul este mult peste valoarea intrinsecă. Risc mare de corecție."
        
    return mos * 100, verdict

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
            colorscale=[[0.0, '#3987e5'], [0.5, '#383835'], [1.0, '#d95926']],
            zmin=-1, zmax=1, xgap=2, ygap=2,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(color='#ffffff'),
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
                        <h4 style="margin: 10px 0; color: white;">{t1} <span style="color: {color};">↔</span> {t2}</h4>
                        <div style="background: {bg_light}; padding: 5px 10px; border-radius: 5px; display: inline-block;">
                            <span style="color: {color}; font-family: monospace; font-size: 18px; font-weight: bold;">{ro(score):.2f}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            col_idx += 1
    st.caption(f"Corelația randamentelor zilnice pe perechi. Peste {ro(CORR_HIGH):.2f}: slabă; {ro(CORR_MID):.2f}–{ro(CORR_HIGH):.2f}: moderată; "
               f"sub {ro(CORR_MID):.2f} (inclusiv negativă): bună. Praguri orientative, netestate.")


def render_benchmark_comparison(portfolio_curve, bench_ticker="SPY", bench_name="S&P 500"):
    """Compară performanța portofoliului cu benchmark-ul folosind procente în tooltip."""
    if portfolio_curve is None or portfolio_curve.empty:
        return
    
    try:
        start_date = portfolio_curve.index[0]
        spy_data = yf.download(bench_ticker, start=start_date, progress=False, auto_adjust=True)['Close']
        
        if isinstance(spy_data, pd.DataFrame):
            spy_data = spy_data.iloc[:, 0]
        
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


def get_daily_briefing_data():
    """Închiderile NEAJUSTATE și volumele pe ~6 luni (Yahoo), pentru Rezumatul Zilei: dict bvb_close, bvb_volume
    (universul BVB + ETF-ul BET), us_close, us_volume (universul SUA + indicii + ETF-urile sectoriale), eu_close,
    eu_volume (universul Europa + indicii) și partial = {piață: True dacă Yahoo a dat sub 80% din simboluri}.
    Fiecare piață are cererea și cache-ul ei: o citire parțială nu se păstrează și nu le strică pe celelalte.
    Prețul neajustat dă variația zilei cum o raportează bursa; 6 luni ajung pentru σ pe 60 de ședințe.
    Fără niciun preț pe nicio piață → excepție."""
    out = {"partial": {}}
    for key, syms in (("bvb", briefing_bvb_universe() + [BRIEFING_BET_PROXY]),
                      ("us", BRIEFING_US_UNIVERSE + BRIEFING_US_INDICES + BRIEFING_US_SECTORS),
                      ("eu", universes.symbols("eu") + BRIEFING_EU_INDICES)):
        try:
            data, partial = _briefing_market(tuple(syms)), False
        except _PartialBriefing as exc:
            data, partial = exc.data, True
            logger.warning("rezumat %s: citire parțială (%d din %d simboluri fără preț)", key,
                           len(briefing_calc.symbols_without_prices(data["close"], syms)), len(syms))
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
          "minimum 20). Peste 2σ = mișcare neobișnuită pentru acest titlu (prag netestat)."),
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
        view = pd.concat([eq_view, bond_view], ignore_index=True) if len(bond_view) else eq_view
        st.dataframe(
            text_frame(view, {
                "Ședința": lambda d: f"{pd.Timestamp(d):%d.%m.%Y}",
                "Variație": lambda v: _pp(v), "σ": lambda v: format_ro(v, 1, "σ"),
                "P/L (lei)": _lei, "Contribuție": _cpp, "Valoare (lei)": _val,
            }),
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
        <a href="{html.escape(str(it['link']), quote=True)}" style="color:#58A6FF; text-decoration:none; font-weight:600;" target="_blank">{html.escape(it['title'])}</a>
        <div style="font-size:12px; color:#8B949E;">{html.escape(str(it['source']))} • {it['time']:%d.%m %H:%M}</div></div>"""
        for it in items))


def render_holdings_news(holdings):
    """„Despre acțiunile tale” (tabul „📰 Știri” din Rezumatul Zilei): știrile din ultimele 48 de ore despre acțiunile
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

# --- FUNCȚII WATCHLIST (CORECȚIE BUG) ---
def load_watchlist():
    """Citește datele din foaia 'watchlist'."""
    ws = connect_to_gsheets("watchlist")  # Direct, fără sheet.spreadsheet.worksheet()
    if ws:
        try:
            data = ws.get_all_records()
            return pd.DataFrame(data)
        except Exception as e:
            logger.warning("load_watchlist: %s", e)
            return pd.DataFrame()
    return pd.DataFrame()

def add_to_watchlist(symbol, target, note):
    """Adaugă o intrare nouă în watchlist."""
    ws = connect_to_gsheets("watchlist")
    if ws:
        try:
            ws.append_row([symbol, float(target), note])
            # load_watchlist nu are cache: nu e nimic de golit (înainte: st.cache_data.clear() global)
            return True
        except Exception as e:
            st.error(f"Eroare salvare: {e}")
            return False
    return False

def remove_from_watchlist(symbol):
    """Șterge un simbol din watchlist (căutând după nume)."""
    ws = connect_to_gsheets("watchlist")
    if ws:
        try:
            cell = ws.find(symbol)
            if cell:
                ws.delete_rows(cell.row)
                # load_watchlist nu are cache: nu e nimic de golit
                return True
        except Exception as exc:  # înainte: except: gol
            logger.warning("remove_from_watchlist: %s", exc)
            pass
    return False

        # load_portfolio nu are cache: nu e nimic de golit

@st.cache_data(ttl=300)
def get_portfolio_history_data(tickers):
    if not tickers: return pd.DataFrame()
    data = yf.download(tickers, period="5y", group_by='ticker', auto_adjust=True)
    return data

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
        # Descărcăm istoricul bulk pentru grafic (cu rate limiter)
        _yf_limiter.wait_if_needed()
        hist_data = yf.download(tickers, period="5y", group_by='ticker', progress=False, auto_adjust=True)

    # close_frame tratează toate formele întoarse de yf.download (un simbol sau mai multe)
    closes = close_frame(hist_data, tickers)

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
    "xirr": "**Cum se calculează:** rata anuală care, aplicată fiecărei depuneri de la data ei până azi, dă exact valoarea de azi a portofoliului "
            "(rata internă de rentabilitate pe fluxuri datate).\n\n**Ce înseamnă:** randamentul tău anual real, ținând cont de momentul în care ai depus fiecare sumă. "
            "Se compară cu dobânda unui depozit sau a unui titlu de stat. Banii depuși recent cântăresc puțin, cei vechi mult.",
    "twr": "**Cum se calculează:** randamentele zilnice ale portofoliului, înlănțuite, cu fiecare depunere scoasă din calcul în ziua ei: "
           "randament zilnic = valoare azi ÷ (valoare ieri + depunerea de azi) − 1.\n\n**Ce înseamnă:** cât a crescut un leu investit la început, "
           "indiferent de când și cât ai mai depus. Măsoară calitatea alegerilor de investiții și e direct comparabil cu un indice (BET, S&P 500).",
    "twr_annual": "**Cum se calculează:** (1 + TWR total) ridicat la puterea 1 ÷ numărul de ani, minus 1.\n\n**Ce înseamnă:** TWR-ul exprimat ca medie pe an. "
                  "Dacă e sub XIRR, ai depus mai mult înaintea perioadelor bune; dacă e peste, înaintea celor slabe.",
    "period": "Prima și ultima zi pentru care există valoare calculată. TWR și indicatorii de risc sunt măsurați pe acest interval.",
    "vol": "**Cum se calculează:** deviația standard a randamentelor zilnice × √252 (zile de tranzacționare pe an).\n\n"
           "**Ce înseamnă:** cât de mult fluctuează portofoliul într-un an obișnuit. La 10%, două treimi din ani ar trebui să se încheie la ±10% față de medie. "
           "Aici e subestimată, pentru că obligațiunile și numerarul nu au variație zilnică.",
    "sharpe": "**Cum se calculează:** (randamentul mediu zilnic − rata fără risc zilnică) ÷ deviația standard zilnică × √252.\n\n"
              "**Ce înseamnă:** cât randament peste titlurile de stat primești pentru fiecare unitate de risc. Sub 0: mai slab decât titlurile de stat. "
              "0–0,5: slab. 0,5–1: acceptabil. Peste 1: bun.",
    "sortino": "**Cum se calculează:** ca Sharpe, dar la numitor intră doar variația zilelor sub rata fără risc (deviația în jos).\n\n"
               "**Ce înseamnă:** randamentul raportat doar la riscul care doare, scăderile. Creșterile bruște nu sunt penalizate. Se citește ca Sharpe, dar valorile sunt de regulă mai mari.",
    "max_dd": "**Cum se calculează:** cea mai mare scădere procentuală de la un vârf la minimul de după el, pe curba randamentelor (fără efectul depunerilor).\n\n"
              "**Ce înseamnă:** cea mai rea perioadă prin care a trecut portofoliul. E pierderea pe care trebuie să fii pregătit să o vezi din nou.",
    "var_hist": "**Cum se calculează:** percentila 5 a randamentelor zilnice din istoric.\n\n**Ce înseamnă:** într-o zi din 20, pierderea a fost mai mare decât atât. "
                "Suma în RON aplică procentul la valoarea de azi a portofoliului.",
    "cvar": "**Cum se calculează:** media randamentelor din cele mai rele 5% din zile.\n\n**Ce înseamnă:** cât pierzi, în medie, într-o zi proastă "
            "(una dintre cele mai rele 20). E mai relevant decât VaR, pentru că arată cât de adâncă e coada, nu doar unde începe.",
    "var_param": "**Cum se calculează:** media zilnică − 1,645 × deviația standard, adică percentila 5 dacă randamentele ar fi distribuite normal.\n\n"
                 "**Ce înseamnă:** estimarea teoretică a aceleiași pierderi. Piețele au zile extreme mai dese decât distribuția normală, "
                 "deci compară-l cu cel istoric: diferența dintre ele arată cât de neobișnuite sunt cozile.",
}
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
    pct = lambda v, d=1: "N/A" if v is None else f"{v * 100:.{d}f}%"
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
    with st.expander("📥 Importă un export de la broker (Tradeville, XTB)"):
        st.caption(
            "Încarcă raportul de activitate Tradeville sau exportul XTB (.xlsx), exact cum l-ai descărcat. "
            "Rândurile deja existente în foaie sunt recunoscute după ID și sărite, deci poți încărca oricând exportul complet. "
            "Nimic nu se scrie până nu apeși butonul de adăugare."
        )
        files = st.file_uploader("Fișiere .xlsx", type=["xlsx"], accept_multiple_files=True, key="ledger_import_files")
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
                sheets = broker_import.read_workbook(uploaded)
            except (zipfile.BadZipFile, KeyError, ValueError, OSError) as e:
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
@st.cache_data(ttl=300)
def get_global_market_data():
    indices = {
        'S&P 500': '^GSPC', 'Dow Jones': '^DJI', 'Nasdaq': '^IXIC', 
        'DAX (GER)': '^GDAXI', 'FTSE 100 (UK)': '^FTSE','Bursa RO (BET) 🇷🇴': 'TVBETETF.RO'
    }
    commodities = {
        'Aur (Gold)': 'GC=F', 'Argint (Silver)': 'SI=F', 
        'Petrol (WTI)': 'CL=F', 'Petrol (Brent)': 'BZ=F', 'Copper': 'HG=F', 'Gaz Natural': 'NG=F'
    }
    
    # Listele unice (data/universes.py): SUA = tot ce se tranzacționează în SUA (inclusiv ADR), Europa = bursele europene.
    us_stocks = universes.symbols("us")
    eu_stocks = universes.symbols("eu")

    all_symbols = list(indices.values()) + list(commodities.values()) + us_stocks + eu_stocks
    tickers = yf.Tickers(' '.join(all_symbols))
    
    def process_tickers(symbol_dict, is_list=False):
        data = []
        source = symbol_dict if is_list else symbol_dict.items()
        for item in source:
            name = item if is_list else item[0]
            sym = item if is_list else item[1]
            try:
                t = tickers.tickers[sym]
                info = t.fast_info
                price = info.last_price
                prev = info.previous_close
                # Fără închiderea precedentă variația e necunoscută (N/A), nu 0.
                if prev and price is not None:
                    change = price - prev
                    pct = (change / prev) * 100
                else: change = None; pct = None
                
                data.append({
                    'Instrument': name, 'Simbol': sym, 'Preț': price, 'Variație': change, 'Variație %': pct
                })
            except Exception as exc:  # înainte: except: gol
                logger.debug("process_tickers: %s", exc)
                continue
        return pd.DataFrame(data)

    df_indices = process_tickers(indices)
    df_commodities = process_tickers(commodities)
    df_us = process_tickers(us_stocks, is_list=True)
    if not df_us.empty:
        us_gainers = df_us.sort_values(by='Variație %', ascending=False).head(10)
        us_losers = df_us.sort_values(by='Variație %', ascending=True).head(10)
    else: us_gainers = us_losers = pd.DataFrame()
        
    df_eu = process_tickers(eu_stocks, is_list=True)
    if not df_eu.empty:
        eu_gainers = df_eu.sort_values(by='Variație %', ascending=False).head(10)
        eu_losers = df_eu.sort_values(by='Variație %', ascending=True).head(10)
    else: eu_gainers = eu_losers = pd.DataFrame()

    return df_indices, df_commodities, us_gainers, us_losers, eu_gainers, eu_losers

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
        'ADPCHGS': 'ADP Private Payrolls',
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
            col_data = df_item[code].dropna()
            
            if len(col_data) >= 2:
                val_curr = col_data.iloc[-1]
                val_prev = col_data.iloc[-2]
                
                if code in ['PAYEMS', 'ADPCHGS', 'JTSJOL']:
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

# --- MAIN APP ---
def main():
    st.sidebar.title("Navigare")
    sectiune = st.sidebar.radio("Mergi la:", [
        "1. Agregator Știri", 
        "2. Analiză Companie", 
        "3. Portofoliu", 
        "4. Piață Globală", 
        "5. Import Date", 
        "6. Rezumatul Zilei",
        "7. Scanner Volum",
        "8. Watchlist" 
    ])
    st.sidebar.markdown("---")

    # Versiunile instalate pe server: necesare ca requirements.txt să fixeze exact ce funcționează
    with st.sidebar.expander("ℹ️ Despre aplicație"):
        import platform
        from importlib import metadata as _md
        rows = [f"Python {platform.python_version()}"]
        for pkg in ("streamlit", "yfinance", "pandas", "numpy", "scipy", "plotly", "curl_cffi", "httpx",
                    "gspread", "google-auth", "google-auth-oauthlib", "h2", "feedparser", "pandas-datareader", "scikit-learn",
                    "prophet", "torch", "transformers", "lxml", "requests", "matplotlib", "textblob", "html5lib"):
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
        st.sidebar.header("Căutare")
        sym = st.sidebar.text_input("Simbol (ex: AAPL, TLV):", "AAPL").upper()
        
        st.sidebar.markdown("### Indicatori Grafic")
        show_sma20 = st.sidebar.checkbox("SMA 20", value=True)
        show_sma50 = st.sidebar.checkbox("SMA 50", value=True)
        show_sma200 = st.sidebar.checkbox("SMA 200", value=True)
        show_rsi = st.sidebar.checkbox("RSI 14", value=True)
        show_macd = st.sidebar.checkbox("MACD", value=True)

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
            # None = indisponibilă; consumatorii (SWOT, rating, Decizie Master AI) nu o punctează.
            curve_market = market_of(real_sym, info.get('currency'))
            spread, spread_label = get_market_curve(curve_market)
            # Intrările sintezei „Decizie Master AI” calculate doar în anumite ramuri: inițializare explicită.
            regime_msg, z_val_swot, s_score_val, c_news_ai = None, None, None, []
            rvol, iv_rank, iv_days = None, None, 0
            
        if hist is None or hist.empty:
            st.error("Simbol invalid sau date indisponibile.")
        else:
            # 1. Informații Generaley
            st.markdown(f"## {info.get('longName') or real_sym}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Sector", info.get('sector') or bvb_sector(real_sym) or 'N/A')
            c2.metric("Industrie", info.get('industry') or 'N/A')
            c3.metric("Capitalizare", format_num(info.get('marketCap')))             
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
                                <p style="color:#8B949E; margin:0; font-size:11px; text-transform:uppercase; letter-spacing:1px;">Țintă Intrare</p>
                                <h2 style="color:white; margin:5px 0; font-size:22px;">{target_text} <span style="font-size:12px; color:#8B949E;">{info.get('currency', 'USD')}</span></h2>
                            </div>
                            <div style="flex: 1; min-width: 100px; text-align: center; border-left: 1px solid #30363D; border-right: 1px solid #30363D;">
                                <p style="color:#8B949E; margin:0; font-size:11px; text-transform:uppercase; letter-spacing:1px;">Preț Live</p>
                                <h2 style="color:{t_color}; margin:5px 0; font-size:28px;">{ro(curr_price):.2f}</h2>
                            </div>
                            <div style="flex: 1; min-width: 100px; text-align: right;">
                                <p style="color:#F85149; margin:0; font-size:11px; text-transform:uppercase; letter-spacing:1px;">Stop-Loss (Exit)</p>
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
            
            if not subset.empty and len(hist) >= 2:
                curr_price = subset['Close'].iloc[-1]
                start_price = subset['Close'].iloc[0]
                diff_val = curr_price - start_price
                diff_pct = (diff_val / start_price) * 100
                prev_close = hist['Close'].iloc[-2]
                day_val = curr_price - prev_close
                day_pct = (day_val / prev_close) * 100
            else:
                curr_price = 0; diff_val = 0; diff_pct = 0; day_val = 0; day_pct = 0

            with col_price_info:
                 m1, m2 = st.columns(2)
                 m1.metric(f"Interval ({time_opt})", f"{ro(curr_price):.2f} {info.get('currency', '')}", f"{ro(diff_val):.2f} ({ro(diff_pct):.2f}%)")
                 m2.metric("Evoluție Azi", f"{ro(curr_price):.2f}", f"{ro(day_val):.2f} ({ro(day_pct):.2f}%)")

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
                fig.add_trace(go.Bar(x=subset.index, y=subset['MACD']-subset['Signal'], name='Hist'), row=current_row, col=1)

            fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False, hovermode="x unified", paper_bgcolor='#0E1117', plot_bgcolor='#0E1117')
            plot_ro(fig, width='stretch')

            # --- SEPARATORUL SOLICITAT (ADAUGĂ ACEASTĂ LINIE) ---
            st.markdown("---")

            # Beta și alpha (un singur calcul, folosit de scorul fundamental, DCF, audit și scoruri)
            risk_stats = resolve_beta_alpha(real_sym, info, hist)
            beta_val, alpha_val = risk_stats["beta"], risk_stats["alpha"]

            # 2b. Decizie BVB în doi pași (doar .RO cu fișier în „Analiză companii”)
            bvb_decision = render_bvb_decision(real_sym, hist)
            # 2c. Scor fundamental pentru acțiunile din afara BVB (descriptiv: testul pe SUA nu a găsit putere de predicție)
            global_decision = render_global_decision(real_sym, info, hist, risk_stats)
            # Sinteza „Decizie Master AI” refolosește exact acest rezultat (nu recalculează)
            score_decision = bvb_decision or global_decision

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
                    div_rate = info.get('dividendRate')
                    div_display = f"{ro(div_rate):.2f} ({ro((div_rate/curr_price*100)):.2f}%)" if (div_rate and curr_price) else "N/A"
                    div_help = fund_help(info, "dividendRate")
                    bvb_dy = (bvb_decision or {}).get("dy")
                    if bvb_dy is not None and (bvb_decision or {}).get("price"):
                        # BVB: din fișierul „Analiză companii” (Yahoo nu trimite dividendul); are prioritate.
                        dps = bvb_dy * bvb_decision["price"]
                        div_display = f"{ro(dps):.4f} ({ro(bvb_dy * 100):.2f}%)" if dps < 1 else f"{ro(dps):.2f} ({ro(bvb_dy * 100):.2f}%)"
                        div_help = (f"{FUND_HELP['dividendRate']} Sursă: fișierul „Analiză companii” {bvb_symbol(real_sym)}: "
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
                    st.metric("Beta", format_num(beta_val), help=f"Sursă: {risk_stats['beta_label']}")
                    st.metric("Alpha (1Y)", format_num(alpha_val, True), help=f"Alpha Jensen. {risk_stats['alpha_label']}")
            
            # ==================================================
            # MODUL NOU: DATE FINANCIARE VIZUALE (STIL XTB)
            # ==================================================
            st.markdown("---")
            st.subheader("📊 Evoluție Financiară (Venit vs. Profit Net)")
            
            with st.spinner("Extragem datele contabile..."):
                try:
                    t_fin = yf.Ticker(real_sym)
                    
                    # Preluăm datele (yfinance le aduce cu datele pe coloane, deci facem Transpose .T)
                    df_ann = t_fin.financials.T
                    df_qtr = t_fin.quarterly_financials.T
                    
                    # Curățăm și ordonăm cronologic (cele mai vechi primele)
                    if not df_ann.empty and not df_qtr.empty:
                        df_ann = df_ann.sort_index(ascending=True)
                        df_qtr = df_qtr.sort_index(ascending=True)
                        
                        # Creăm Tab-uri exact ca în XTB
                        tab_anual, tab_trimestrial = st.tabs(["🗓️ Anual", "📅 Trimestrial"])
                        
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
                                textfont=dict(color='#FFAB00', weight='bold')
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
                            fig.update_yaxes(title_text="Marjă (%)", showgrid=False, ticksuffix="%", color="#FFAB00", secondary_y=True)
                            
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
                if p_own is None:
                    p_col.metric(p_title, "N/A")
                elif p_diff is None:
                    p_col.metric(p_title, p_fmt.format(ro(p_own)), "fără mediană de comparație", delta_color="off")
                else:
                    p_col.metric(p_title, p_fmt.format(ro(p_own)),
                                 f"{ro(p_diff * 100):+.0f}% față de mediana {p_fmt.format(ro(p_med))} (n={p_n})", delta_color="off")

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
                    lambda row: ["font-weight: bold; background-color: #21262D" if row.name < 2 else "" for _ in row], axis=1),
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
            cf1.metric("Venituri Totale", format_num(rev))
            cf2.metric("Profit Net", format_num(net_inc))
            cf3.metric("Cheltuieli (Est.)", format_num(exp))
            cf4.metric("Numerar Disponibil", format_num(cash))
            
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
                st.metric("Preț Țintă (Mediu)", f"{ro(target):,.2f} {info.get('currency','USD')}" if target else "N/A")

            with col_an_right:
                st.markdown("""<div class="fin-card"><h4>🆚 Raportări vs Așteptări</h4></div>""", unsafe_allow_html=True)
                if earn_df is not None and not earn_df.empty:
                    def style_surprise(val):
                        color = '#3FB950' if val > 0 else '#F85149' if val < 0 else '#8B949E'
                        return f'color: {color}; font-weight: bold'
                    e_disp = earn_df[['epsEstimate', 'epsActual', 'epsDifference', 'surprisePercent']].copy()
                    e_disp.columns = ['Estimare', 'Realizat', 'Diferență', 'Surpriză %']
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
                            <p style="color:#8B949E; margin:0; font-size:11px; text-transform:uppercase;">Volum Relativ (RVOL)</p>
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
                        <p style='color:#8B949E; margin:0; font-size:11px; text-transform:uppercase;'>Director Executiv (CEO)</p>
                        <h3 style='margin:10px 0; color:white; font-size:18px;'>{ceo_name}</h3>
                    </div>
                    """, unsafe_allow_html=True)

                with c_total:
                    # Calculăm culoarea în funcție de dominanță
                    i_col = "#3FB950" if inst_percent > 50 else "#D29922"
                    st.markdown(f"""
                    <div style='background:#161B22; padding:20px; border-radius:15px; border-top: 4px solid {i_col}; text-align:center; height:130px;'>
                        <p style='color:#8B949E; margin:0; font-size:11px; text-transform:uppercase;'>Dețineri Instituționale Totale</p>
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
                        f'<div style="color:{"white" if color == "#30363D" else color}; font-size:40px; line-height:46px; font-weight:700; flex:1; display:flex; align-items:center; justify-content:center; gap:8px;">{html.escape(value)}{unit_html}</div>'
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
                        return "color: #8B949E"
                    return "color: #3FB950" if v > price_f else "color: #F85149"

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
                    <p style="color:#8B949E; margin:0; font-size:11px; text-transform:uppercase;">{az_title}</p>
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
                    <p style="color:#8B949E; margin:0; font-size:11px; text-transform:uppercase;">Criterii trecute</p>
                    <h1 style="color:{pio_color}; margin:10px 0; font-size:40px;">{pio_value}</h1>
                    <p style="color:#C9D1D9; font-size:12px; margin:0;">{pio_label}</p>
                </div>
                """, unsafe_allow_html=True)
                nd_ebitda = lev["net_debt_to_ebitda"]
                st.metric("Datorie netă / EBITDA", "N/A" if nd_ebitda is None else f"{ro(nd_ebitda):.2f}x",
                          help="Câți ani de EBITDA ar acoperi datoria netă. Negativ = numerar net. N/A când EBITDA e negativ sau lipsește. "
                               f"Datorie netă {format_amount(lev['net_debt'])}, EBITDA {format_amount(lev['ebitda'])}.")
                int_cov = lev["interest_coverage"]
                st.metric("Acoperirea dobânzii", "N/A" if int_cov is None else f"{ro(int_cov):.1f}x",
                          help="EBIT împărțit la cheltuielile cu dobânzile raportate. Sub 1,5x profitul operațional abia acoperă dobânda. "
                               f"EBIT {format_amount(lev['ebit'])}, dobânzi {format_amount(lev['interest_expense'])}. "
                               + ("N/A aici: dobânzile raportate depășesc 25% din datorie, deci rândul include și alte costuri financiare."
                                  if lev["interest_unreliable"] else
                                  "Atenție: la unele companii „dobânzile” raportate includ și alte costuri financiare."))
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
                    <p style="color:#8B949E; margin:0; font-size:11px; text-transform:uppercase;">Scor Sănătate Financiară</p>
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
                        <p style="color:#8B949E; margin:0; font-size:11px; text-transform:uppercase;">Margin of Safety</p>
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
                    <p style="color:#8B949E; margin:0; font-size:11px; text-transform:uppercase;">Cash-to-Income Ratio</p>
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
                    <p style='margin:0; color:#8B949E; font-size: 11px; text-transform: uppercase; font-weight: bold; letter-spacing:1px;'>Radar AI: Clasificare Regim (K-Means)</p>
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
                st.write("📊 **Analiză Sentiment AI (Context 24h)**")
                if c_news_ai:
                    with st.spinner("FinBERT procesează narațiunea pieței..."):
                        s_score = analyze_sentiment_ai(c_news_ai)
                        
                        # --- LOGICĂ EXTINSĂ PE 5 PRAGURI ---
                        if s_score is None:      # FinBERT indisponibil: fără scor, nu 0 „neutru”
                            c_ai, status_text = "#8B949E", "Indisponibil: FinBERT nu a putut analiza știrile"
                        elif s_score >= 0.25:
                            c_ai, status_text = "#3FB950", "🚀 Euforie media"
                        elif s_score >= 0.05:
                            c_ai, status_text = "#3FB950", "📈 Optimism moderat"
                        elif s_score <= -0.25:
                            c_ai, status_text = "#F85149", "🚨 Panică extremă"
                        elif s_score <= -0.05:
                            c_ai, status_text = "#F85149", "📉 Îngrijorare / Pesimism"
                        else:
                            c_ai, status_text = "#8B949E", "⚖️ Zgomot neutru"
                        
                        # UI PRO: Indicator vizual de încredere
                        confidence = "RIDICATĂ" if len(c_news_ai) > 5 else "MODERATĂ"
                        
                        st.markdown(f"""
                        <div style='background:#161B22; padding:20px; border-radius:15px; border-left: 5px solid {c_ai}; text-align:center;'>
                            <h1 style='color:{c_ai}; margin:0; font-size:48px;'>{"N/A" if s_score is None else f"{ro(s_score):.2f}"}</h1>
                            <div style='color:{c_ai}; font-weight:bold; font-size:18px; margin-top:5px;'>{status_text}</div>
                            <p style='color:#8B949E; font-size:11px; margin-top:10px;'>Încredere Model: {confidence} ({len(c_news_ai)} surse)</p>
                        </div>
                        
                        <div style='margin-top: 15px; padding: 15px; background: #21262D; border-radius: 10px; border-left: 3px solid #58A6FF;'>
                            <p style='color:#8B949E; font-size: 11px; margin:0 0 8px 0; text-transform: uppercase; font-weight: bold;'>Ghid Intervale (Scală -1 la +1)</p>
                            <div style='color:#C9D1D9; font-size: 13px; line-height: 1.8;'>
                                <div><span style='color:#3FB950;'>■</span> <b>+0,25 la +1,00:</b> Euforie media</div>
                                <div><span style='color:#3FB950; opacity:0.7;'>■</span> <b>+0,05 la +0,24:</b> Optimism moderat</div>
                                <div><span style='color:#8B949E;'>■</span> <b>-0,05 la +0,05:</b> Zgomot neutru</div>
                                <div><span style='color:#F85149; opacity:0.7;'>■</span> <b>-0,24 la -0,06:</b> Îngrijorare / Pesimism</div>
                                <div><span style='color:#F85149;'>■</span> <b>-1,00 la -0,25:</b> Panică extremă</div>
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
            
            # Verdict Final (rating vechi): mutat în arhiva de la „Decizie Master AI”.
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
                                <p style="margin:0; color:#8B949E; text-transform:uppercase; font-size:11px; letter-spacing:1px; font-weight:bold;">Expunere netă la gamma (GEX, estimare) · {opt_data['gex_n']} contracte</p>
                                <h1 style="margin:10px 0; color:{opt_data['gex_color']}; font-size:36px;">{gex_txt}</h1>
                            </div>
                            <div style="text-align:right;">
                                <span style="background:{opt_data['gex_color']}; color:white; padding:8px 16px; border-radius:25px; font-weight:bold; font-size:13px;">
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
                              help="Σ open interest put / Σ open interest call, pe expirarea aleasă. Poziții deschise acumulate. "
                                   "Pragurile <0,7 optimism / >1,1 pesimism sunt convenții de piață, netestate aici. N/A = fără open interest la call-uri.")
                    m2.metric("Put/Call (volum)", _f(vol_pc, ".2f"),
                              help="Σ volum put / Σ volum call, pe expirarea aleasă, în ședința curentă. Se mișcă mai repede decât open interest-ul. N/A = fără volum la call-uri.")

                    m3, m4 = st.columns(2)
                    mp = opt_data['max_pain']
                    mp_display = f"{ro(mp):.2f} USD" if mp is not None else "N/A"
                    mp_help = ("Strike-ul la care opțiunile acestei expirări ar valora cel mai puțin la expirare, "
                               "calculat din open interest. Este un reper, nu o țintă de preț.")
                    if mp is not None:
                        diff_mp = ((mp / spot_opt) - 1) * 100
                        m3.metric("Preț Max Pain", mp_display, f"{ro(diff_mp):.1f}% față de preț", help=mp_help)
                    else:
                        m3.metric("Preț Max Pain", mp_display, help=mp_help)
                    m4.metric("Expirare", opt_data['expiration'], f"{opt_data['days']} zile", delta_color="off",
                              help="Expirarea cu 20–45 de zile rămase cea mai apropiată de 30. Expirările de câteva zile au IV deformat de timpul scurt.")

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
                                   help="(IV de azi − minim) / (maxim − minim) pe ultimul an, din istoricul IV la bani salvat zilnic în foaia IV_istoric. "
                                        f"Sub {opt_calc.MIN_IV_OBS} de zile de istoric nu se calculează (N/A).")
                    c_rank2.metric("IV Percentile", _f(rk["percentile"], ".1f") + ("%" if rk["percentile"] is not None else ""),
                                   obs_txt, delta_color="off",
                                   help="Procentul zilelor din ultimul an cu IV la bani mai mic decât azi. Cere același istoric ca IV Rank.")
                    c_rank3.metric("IV / volatilitate realizată 30z", _f(iv_hv, ".2f"),
                                   f"realizată {ro(hv30*100):.1f}%" if hv30 else None, delta_color="off",
                                   help="IV la bani / volatilitatea realizată pe ultimele 30 de zile calendaristice (randamente log, √252). "
                                        "Peste 1: piața plătește pentru mai multă mișcare decât a avut acțiunea recent. Nu este IV Rank.")

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
                              help="Variația prețului opțiunii la +1 USD în prețul acțiunii. Black-Scholes cu IV la bani, r și q din antet.")
                    g2.metric("Gamma", f"{ro(gc['gamma']):.4f}",
                              help="Variația deltei la +1 USD în prețul acțiunii; aceeași pentru call și put.")
                    g3.metric("Theta call / put (pe zi)", f"{ro(gc['theta_day']):.3f} / {ro(gp['theta_day']):.3f}",
                              help="Pierderea de valoare într-o zi calendaristică, pe o acțiune suport (un contract = 100). Theta anual / 365.")
                    g4.metric("Vega (la 1 pp de IV)", f"{ro(gc['vega_pt']):.3f}",
                              help="Variația prețului opțiunii la +1 punct procentual de volatilitate implicită; aceeași pentru call și put.")
                else:
                    st.info("Greeks N/A: lipsesc IV la bani sau rata fără risc (vezi antetul secțiunii).")
                em = opt_data['expected_move']
                if em is not None:
                    st.metric("Mișcare așteptată până la expirare (±1σ)", f"±{ro(em):.2f} ({ro(em / spot_opt * 100):.1f}%)",
                              help="Preț × IV la bani × √(zile/365). Intervalul de ±1 abatere standard implicat de opțiuni "
                                   "(~68% probabilitate sub ipoteza de normalitate, care subestimează cozile).")

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
            # 👑 DECIZIE MASTER AI: SINTEZA ÎN TREI PAȘI (fundamental, moment, context)
            # Pașii nu se adună. Fundamentalul și momentul vin neschimbate din secțiunea de scor de sus;
            # contextul e o listă de alerte fără puncte. Metoda veche rămâne în arhivă, neschimbată.
            # =================================================================
            st.markdown("---")
            st.subheader("👑 Decizie Master AI")
            from analytics.decision import characterize, context_flags, count_alerts
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

            if score_decision is None:
                st.info("Fără scor fundamental pentru acest simbol (ETF, fond, indice sau emitent .RO fără fișier în "
                        f"„{COMPANY_FOLDER_NAME}”). Sinteza arată doar contextul; metoda veche e în arhivă.")
            else:
                d = score_decision
                alert_color = "#F85149" if any(f["level"] == "alerta" for f in ctx_flags) else (
                    "#D29922" if n_alerts else "#3FB950")
                phrase = characterize(is_bvb=d["is_bvb"], fund_label=d["fund_label"], fund_score=d["fund_score"],
                                      moment_score=d["moment"], pillars=d["pillars"], n_alerts=n_alerts,
                                      history_text=d["history_text"])
                st.markdown(compact_html(f"""
                    <div style="background:#161B22; border:1px solid #30363D; border-radius:12px; padding:16px; margin:8px 0 12px 0;">
                        <div style="display:flex; flex-wrap:wrap; gap:32px; align-items:baseline;">
                            <div><span style="color:#8B949E; font-size:12px; text-transform:uppercase;">Pasul 1 · Fundamental</span><br>
                                <span style="color:{d['fund_color']}; font-weight:bold; font-size:22px;">{html.escape(d['fund_label'])}</span>
                                <span style="color:#C9D1D9; font-size:15px;">&nbsp;{html.escape(d['fund_detail'])}</span><br>
                                <span style="color:#8B949E; font-size:12px;">{html.escape(d['fund_basis'])}</span></div>
                            <div><span style="color:#8B949E; font-size:12px; text-transform:uppercase;">Pasul 2 · Moment</span><br>
                                <span style="color:{d['moment_color']}; font-weight:bold; font-size:22px;">{html.escape(d['moment_label'])}</span>
                                <span style="color:#C9D1D9; font-size:15px;">&nbsp;{html.escape(d['moment_detail'])}</span><br>
                                <span style="color:#8B949E; font-size:12px;">{html.escape(d['moment_basis'])}</span></div>
                            <div><span style="color:#8B949E; font-size:12px; text-transform:uppercase;">Pasul 3 · Context</span><br>
                                <span style="color:{alert_color}; font-weight:bold; font-size:22px;">{n_alerts} {'alertă' if n_alerts == 1 else 'alerte'}</span><br>
                                <span style="color:#8B949E; font-size:12px;">fără puncte</span></div>
                        </div>
                        <div style="color:#C9D1D9; font-size:14px; margin-top:12px; line-height:1.5;">{html.escape(phrase)}</div>
                    </div>"""), unsafe_allow_html=True)
                if d["pillars"]:
                    p_cols = st.columns(len(d["pillars"]))
                    for col, (p_name, p_score) in zip(p_cols, d["pillars"]):
                        col.metric(p_name, "N/A" if p_score is None else f"{ro(p_score):.0f}/100",
                                   help="Pilonul scorului fundamental din secțiunea „Scor fundamental” de sus; aceeași valoare.")

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

            with st.expander("🗄️ Metoda veche (arhivă, netestată): Verdict Final și Master AI pe 10 piloni"):
                st.caption("Calculele vechi, păstrate neschimbate pentru comparație. Adună tehnicul cu fundamentalul, "
                           "folosesc praguri netestate și pot contrazice sinteza de mai sus.")
                # ==================================================
                # CALCUL RATING FINAL (CONCLUZIA)
                # ==================================================
                st.markdown("##### 🎯 Verdict Final: Rating de Investiție (arhivă)")

                # Pregătim variabilele (inițializare explicită — nu mai depindem de locals())

                s_inst = inst_percent  # None = acționariat indisponibil
                s_mos = mos_val
                s_rvol = rvol if rvol is not None else 1.0
            
                # Apelăm funcția PRO folosind noul spread 10Y-3M
                final_score, highlights = calculate_investment_rating_pro(info, s_inst, s_rvol, spread, s_mos, spread_label)
            
                r_color = "#3FB950" if final_score > 70 else ("#D29922" if final_score > 40 else "#F85149")
                r_label = "STRONG BUY" if final_score > 80 else ("ACCUMULATE" if final_score > 60 else "AVOID/WATCH")
                # Fără acționariat, DCF, ROE și datorii, scorul ar fi doar 50 + macro: nu e un verdict.
                if s_inst is None and s_mos is None and num(info, 'returnOnEquity') is None and num(info, 'debtToEquity') is None:
                    r_color, r_label = "#8B949E", "DATE INSUFICIENTE"

                # Afișare UI
                c_res1, c_res2 = st.columns([1, 2])
                with c_res1:
                    st.markdown(f"""
                        <div style="background:#161B22; padding:30px; border-radius:15px; border:2px solid {r_color}; text-align:center;">
                            <p style="color:#8B949E; margin:0; font-size:11px; text-transform:uppercase;">Scor Investiție</p>
                            <h1 style="color:{r_color}; margin:15px 0; font-size:54px;">{final_score}</h1>
                            <div style="background:{r_color}22; color:{r_color}; padding:5px; border-radius:5px; font-weight:bold;">{r_label}</div>
                        </div>
                    """, unsafe_allow_html=True)

                with c_res2:
                    st.markdown("#### 🔍 Argumente pentru acest Scor:")
                    for item in highlights:
                        st.write(item)

                    # --- AFISARE SCOR SĂNĂTATE FINANCIARĂ UNIFORMIZAT ---
                    # Stabilim pictograma în funcție de nota de sănătate (h_score)
                    if h_score is None:
                        st.write("⚪ **Sănătate Financiară:** indisponibilă (lipsesc datoriile, ROE și lichiditatea).")
                    else:
                        h_icon = "🟢" if h_score >= 8 else ("🟡" if h_score >= 5 else "🔴")
                        st.write(f"{h_icon} **Sănătate Financiară:** Scorul de stabilitate al bilanțului este **{h_score}/10**.")


                # =================================================================
                # 👑 VERDICT FINAL MASTER AI: DECIZIA DE INVESTIȚIE (RADIOGRAFIE)
                # =================================================================
                st.markdown("##### 👑 Decizie Master AI (arhivă)")
            
                # --- PROTECȚIE VARIABILE (inițializare explicită — fără locals()) ---
                s_inst = inst_percent  # None = acționariat indisponibil
                s_mos = mos_val
                s_rvol = rvol if rvol is not None else 1.0
                s_score_final = s_score_val if s_score_val is not None else 0
                # Pilonul de opțiuni intră în Master AI doar cu P/C pe open interest și IV la bani calculate; altfel lipsește (fără puncte)
                opt_final = opt_data if (opt_data and opt_data.get('oi_pc_ratio') is not None and opt_data.get('iv') is not None) else None
            
                # Extragem datele specifice modulelor adiacente
                z_score_val = z_val_swot
                cash_ratio = q_ratio
                ai_regime = regime_msg or "Neutru"

                # Curba pieței acțiunii, aceeași ca în rating (None = indisponibilă, fără puncte)
                curr_spread = spread

                # --- RULĂM MOTORUL DE SINTEZĂ GLOBALĂ ---
                from ai_engine import calculate_master_ai_score
                m_score, m_action, m_col, m_advice, m_reasons = calculate_master_ai_score(
                    info, hist, h_score, s_mos, s_inst, s_rvol, s_score_final, opt_final, 
                    curr_spread, z_score_val, cash_ratio, ai_regime
                )

                # --- AFIȘARE VIZUALĂ DE IMPACT (Dashboard Bloomberg-style) ---
                col_m1, col_m2 = st.columns([1, 1.8])
            
                with col_m1:
                    st.markdown(f"""
                        <div style="background:#161B22; padding:30px; border-radius:15px; border:3px solid {m_col}; text-align:center; height: 100%; box-shadow: 0 4px 15px rgba(0,0,0,0.5);">
                            <p style="color:#8B949E; margin:0; font-size:12px; text-transform:uppercase; letter-spacing: 1px;">Scor Algoritmic Integrat</p>
                            <h1 style="color:{m_col}; margin:15px 0; font-size:80px; text-shadow: 0 0 10px {m_col}44;">{int(m_score)}<span style="font-size: 20px; color:#8B949E; text-shadow: none;">/100</span></h1>
                            <div style="background:{m_col}; color:white; padding:12px; border-radius:8px; font-weight:bold; font-size:18px; letter-spacing: 1px;">
                                {m_action}
                            </div>
                            <p style="color:#8B949E; font-size:13px; margin-top:15px; line-height:1.4;">
                                {m_advice}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)

                with col_m2:
                    st.markdown("#### 🧠 Radiografia Deciziei (De ce să faci asta?)")
                    st.markdown("<p style='color:#8B949E; font-size:13px; margin-bottom:15px;'>Algoritmul a scanat toți cei 10 piloni (DCF, Tehnic, Opțiuni, Macro, Sentiment, Instituții, Bilanț, Cash-Flow, Faliment, Volum) și a identificat următoarele:</p>", unsafe_allow_html=True)
                
                    # Afișăm lista curățată și sortată (Roșu sus, Verde jos)
                    for reason in m_reasons:
                        # caseta e HTML: **text** din Markdown nu se interpretează acolo, deci devine <b>text</b>
                        reason = md_bold_html(reason)
                        if "🚨" in reason:
                            st.markdown(f"<div style='background:rgba(248, 81, 73, 0.15); padding:12px; border-radius:8px; margin-bottom:8px; border-left:4px solid #F85149; color:#FFD8D8;'>{reason}</div>", unsafe_allow_html=True)
                        elif "⚠️" in reason:
                            st.markdown(f"<div style='background:rgba(210, 153, 34, 0.1); padding:12px; border-radius:8px; margin-bottom:8px; border-left:4px solid #D29922;'>{reason}</div>", unsafe_allow_html=True)
                        elif "✅" in reason:
                            st.markdown(f"<div style='background:rgba(63, 185, 80, 0.05); padding:12px; border-radius:8px; margin-bottom:8px; border-left:4px solid #3FB950;'>{reason}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div style='background:#21262D; padding:12px; border-radius:8px; margin-bottom:8px; border-left:4px solid #8B949E;'>{reason}</div>", unsafe_allow_html=True)
            
            
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
                        text_news = f"Acoperirea în presă este neutră, fără șocuri reputaționale majore în ultimele 24 de ore. Subiectul momentului: «{top_news}»."
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
                                
    # ==================================================
    # 3. PORTOFOLIU (MODIFICAT PENTRU MOBIL)
    # ==================================================
    elif sectiune == "3. Portofoliu":
        st.title("💼 Portofoliu Personal")
        
        # Formularul „Adaugă Tranzacție Nouă” (scria în foaia Portofoliu) a fost scos la 04.11.2026, cu acordul
        # lui Daniel: tranzacțiile intră doar prin importul exporturilor de la broker (deduplicare pe ID).

        # Toată pagina folosește pozițiile din registrul `Tranzactii` (exporturile brokerilor);
        # foaia Portofoliu nu mai alimentează pagina.
        ledger_positions = render_ledger_results()
        df_pf = ledger_calc.tab_frame(ledger_positions)
        st.markdown("---")

        if df_pf.empty:
            st.info("Nu există acțiuni sau ETF-uri deschise în registrul `Tranzactii` (sau registrul nu a putut fi citit).")
        else:
            st.markdown("### Perioadă Analiză")
            hist_range = st.select_slider("Perioadă analiză", label_visibility="collapsed", options=["1Z", "1S", "1L", "3L", "6L", "1A", "3A", "5A"], value="1A", key="range_slider")
            
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
                c_kpi1.metric(f"Total Investit ({currency_symbol})", f"{format_ro(total_invested)} {currency_symbol}")
                c_kpi2.metric(f"Valoare Curentă ({currency_symbol})", f"{format_ro(total_current)} {currency_symbol}")
                c_kpi3.metric(f"Profit/Pierdere ({currency_symbol})", f"{format_ro(total_profit_val)} {currency_symbol}", f"{format_ro(total_profit_pct)}%")

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
                                help="**Cum se calculează:** percentila 5 a valorii finale din cele 1000 de scenarii, fiecare construit din randamente zilnice "
                                     "trase la întâmplare din istoricul portofoliului.\n\n**Ce înseamnă:** în 95% din scenarii portofoliul a terminat anul peste această valoare.")
                    c_mc2.metric("Variația în scenariul pesimist", f"{ro(mc['change_p5_pct']):+.2f}%",
                                help="**Cum se calculează:** (valoarea din scenariul pesimist ÷ valoarea de azi − 1) × 100.\n\n"
                                     "**Ce înseamnă:** cât ai pierde (minus) sau câștiga (plus) într-un an slab, de felul celor care apar o dată la 20. "
                                     "Poate fi pozitivă dacă perioada aleasă a fost foarte bună.")
                    c_mc3.metric("Probabilitatea de profit", f"{ro(mc['prob_profit']):.1f}%",
                                help="**Cum se calculează:** câte scenarii din 1000 au terminat peste valoarea de azi.\n\n"
                                     "**Ce înseamnă:** depinde direct de randamentul mediu al perioadei alese. Comparați cu varianta „fără tendință” de mai jos: "
                                     "acolo iese puțin sub 50%, pentru că volatilitatea singură trage valoarea mediană în jos.")
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
                            help="Cea mai mare scădere de la un vârf la minimul de după el, pe curba TWR din fereastra aleasă." + n_txt)
                c_r2.metric("Sharpe Ratio", rat_t(rk.get("sharpe")),
                            help="(media zilnică − rata fără risc zilnică) / deviația standard zilnică × √252. "
                                 "Sub 0: mai slab decât titlurile de stat; peste 1: bun. " + rf_txt + n_txt)
                c_r3.metric("VaR istoric (95%, 1 zi)", amt_t(rk.get("var95_hist")),
                            help="Percentila 5 a randamentelor zilnice, aplicată la valoarea de azi: într-o zi din 20 pierderea a fost "
                                 f"mai mare. Procent: {pct_t(rk.get('var95_hist'))}." + n_txt)
                c_r4.metric("Volatilitate Anualizată", pct_t(rk.get("vol_ann")),
                            help="Deviația standard a randamentelor zilnice × √252." + n_txt)

                st.write("")
                c_b1, c_b2, c_b3, c_b4 = st.columns(4)
                c_b1.metric("Benchmark", bench_name, help="Indicele pieței în aceeași monedă cu pozițiile.")
                bc_txt = (f" {bc['n']} randamente săptămânale comune, {bc['start']:%d.%m.%Y}–{bc['end']:%d.%m.%Y}." if bc
                          else " Sub 52 de săptămâni comune: N/A.")
                corr_v = bc["corr"] if bc else None
                c_b2.metric("Corelație cu Piața", rat_t(corr_v),
                            help=f"Corelația randamentelor săptămânale TWR cu {bench_name}. 1,00 = mișcare identică." + bc_txt)
                c_b3.metric("Beta Portofoliu", rat_t(bc["beta"] if bc else None),
                            help=f"cov(portofoliu, {bench_name}) / var({bench_name}), pe randamente săptămânale. "
                                 "Peste 1 = mai sensibil decât piața." + bc_txt)
                c_b4.metric("Scor Diversificare", "N/A" if corr_v is None else f"{ro((1 - abs(corr_v)) * 100):.1f}%",
                            help="(1 − |corelația cu piața|) × 100. Cât de independent se mișcă portofoliul față de indice.")

                st.write("")
                c_s1, c_s2, c_s3 = st.columns(3)
                sortino_val = rk.get("sortino")
                c_s1.metric("Sortino Ratio", rat_t(sortino_val),
                            help="Ca Sharpe, dar la numitor intră doar variația zilelor sub rata fără risc. " + rf_txt + n_txt)
                c_s2.metric("CVaR (95%, 1 zi)", amt_t(rk.get("cvar95")),
                            help="Media pierderilor din cele mai rele 5% din zile, aplicată la valoarea de azi. "
                                 f"Procent: {pct_t(rk.get('cvar95'))}." + n_txt)
                c_s3.metric("VaR parametric (95%)", amt_t(rk.get("var95_param")),
                            help="media − 1,645 × deviația standard (presupune distribuție normală; subestimează cozile). "
                                 f"Procent: {pct_t(rk.get('var95_param'))}." + n_txt)
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
                        color = '#3FB950' if val >= 0 else '#F85149'
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
                                        help="**Cum se calculează:** media randamentelor zilnice ale alocării × 252, pe tot istoricul comun.\n\n"
                                             "**Ce înseamnă:** cât ar fi adus această alocare în trecut. E calculat pe aceleași date pe care a fost aleasă, "
                                             "deci e optimist prin construcție; nu e o estimare pentru viitor.")
                            
                            c_opt2.metric("Volatilitate (Risc)", f"{ro(opt_res['expected_volatility']):.2f}%",
                                        help="**Cum se calculează:** deviația standard a randamentelor zilnice ale alocării × √252.\n\n"
                                             "**Ce înseamnă:** cât de mult ar fi fluctuat alocarea într-un an obișnuit.")
                            
                            c_opt3.metric("Sharpe Ratio", "N/A" if (opt_res['sharpe_ratio'] is None or rf_opt is None) else f"{ro(opt_res['sharpe_ratio']):.2f}",
                                        help="**Cum se calculează:** (randament anual − rata fără risc) ÷ volatilitate.\n\n"
                                             "**Ce înseamnă:** randamentul peste titlurile de stat pentru fiecare unitate de risc. E valoarea maximă posibilă pe aceste date, "
                                             "cu plafonul pe poziție; în practică iese mai mic.")

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
        m_score_txt = "N/A" if m_score is None else f"{m_score:.0f}"

        # --- 3. AFIȘARE BANNER ---
        st.markdown(f"""
            <div style="background:linear-gradient(90deg, #161B22 0%, #21262D 100%); padding:30px; border-radius:15px; border-left: 10px solid {m_col}; margin-bottom:20px; box-shadow: 0 4px 15px rgba(0,0,0,0.5);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4 style="color:#8B949E; margin:0; text-transform:uppercase; letter-spacing:1px;">Mediul de piață SUA (verdict macro, netestat)</h4>
                        <h1 style="color:{m_col}; margin:10px 0; font-size:38px;">{m_label}</h1>
                        <p style="color:#C9D1D9; font-size:16px;">{m_desc}</p>
                    </div>
                    <div style="text-align:center; min-width: 120px;">
                        <div style="font-size:12px; color:#8B949E;">SCOR MACRO</div>
                        <div style="font-size:56px; font-weight:bold; color:{m_col};">{m_score_txt}</div>
                        <div style="font-size:14px; color:#8B949E;">/ 100</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

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
                pd.DataFrame(reg_rows).style.format({
                    "Ultima închidere": lambda v: format_ro(v, 2), "Data": lambda d: "N/A" if d is None or pd.isna(d) else f"{d:%d.%m.%Y}",
                    "Momentum 12-1": pct1, "Scor moment": lambda v: "N/A" if v is None or pd.isna(v) else f"{ro(v):.0f}",
                    "De la maximul pe 52 săpt.": pct1}, na_rep="N/A"),
                hide_index=True, width='stretch',
                column_config={
                    "Trend": st.column_config.Column(help="Câte din 3 condiții sunt îndeplinite: închiderea peste media pe 50 de ședințe, "
                                                          "peste media pe 200, media pe 50 peste cea pe 200. 3/3 = trend ascendent complet."),
                    "Momentum 12-1": st.column_config.Column(help="Randamentul de acum 12 luni până acum o lună (fără ultima lună), pe date calendaristice."),
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
                pd.DataFrame(ret_rows).style.format({"Randament local": pct1, "Efect curs față de leu": pct1, "Randament în lei": pct1}, na_rep="N/A"),
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
                     help="Volatilitatea implicită pe 30 de zile ÷ cea pe 3 luni (S&P 500), aceeași zi. "
                          f"Sub {ro(stress_calc.VIX_TERM_TENSE):.2f}: normal (piața se teme mai puțin de imediat decât de viitor); "
                          f"{ro(stress_calc.VIX_TERM_TENSE):.2f}–{ro(stress_calc.VIX_TERM_STRESS):.2f}: tensiune; peste 1: stres acut "
                          "(apare în corecții și vânzări forțate). Praguri uzuale de piață, netestate."
                          + ("" if vterm is None else f" VIX {ro(vterm['vix']):.2f}, VIX3M {ro(vterm['vix3m']):.2f}, {vterm['date']:%d.%m.%Y}."))
        for col, code, name in ((c_st2, "BAMLH0A0HYM2", "Spread high yield SUA"), (c_st3, "BAMLC0A0CM", "Spread investment grade SUA")):
            cs = credit[code]
            col.metric(name, "N/A" if cs["value"] is None else f"{ro(cs['value']):.2f} pp",
                       None if cs["d1m"] is None else f"{ro(cs['d1m']):+.2f} pp într-o lună", delta_color="inverse",
                       help=f"ICE BofA option-adjusted spread (FRED {code}): randamentul obligațiunilor corporative peste titlurile de stat. "
                            "Crește când investitorii cer mai mult pentru riscul de neplată (stres); scade când creditul e ieftin. "
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
            get_global_market_data.clear()
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
                                <div style="color: #8B949E; font-size: 11px; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 12px; font-weight: bold;">
                                    Trend Rotație (1 lună)
                                </div>
                                <div style="font-size: 22px; font-weight: bold; color: #FFFFFF; margin-bottom: 18px;">
                                    {rot_status}
                                </div>
                                <div style="display: flex; align-items: center; justify-content: space-between;">
                                    <div style="background: {delta_color}22; color: {delta_color}; padding: 4px 12px; border-radius: 6px; font-weight: bold; font-size: 15px; border: 1px solid {delta_color}44;">
                                        {ro(rot_change):+.2f}%
                                    </div>
                                    <div style="color: #8B949E; font-size: 13px;">
                                        Scor real: <span style="color: #FFFFFF; font-weight: bold; font-family: 'Courier New', monospace; font-size: 16px;">{ro(current_ratio):.4f}</span>
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

            else:
                st.warning("Date indisponibile sau eroare conexiune Yahoo.")
        
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
                    textfont=dict(color='white', weight='bold')
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
                        colorscale=[[0.0, '#3987e5'], [0.5, '#383835'], [1.0, '#d95926']], 
                        zmin=-1, zmax=1, xgap=2, ygap=2,
                        text=text_matrix,
                        texttemplate="%{text}",
                        textfont=dict(color='#ffffff'),
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
                            if '+' in val: return 'color: #3FB950; font-weight: bold'
                            elif '-' in val: return 'color: #F85149; font-weight: bold'
                        return ''
                    
                    st.dataframe(
                        df_macro[['Indicator', 'Valoare Curentă', 'Lună Precedentă', 'Evoluție (MoM)', 'Evoluție (An/An)']].style
                        .map(color_macro_trend, subset=['Evoluție (MoM)'])
                        .format({
                            'Valoare Curentă': ro_spec('.2f'),
                            'Lună Precedentă': ro_spec('.2f')
                        }),
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
                                {insight}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Piața se află în parametrii așteptați. Nicio anomalie macroeconomică detectată luna aceasta.")
            else:
                st.warning("Datele FRED sunt momentan indisponibile. Verifică conexiunea.")
        
        # --- TABELELE VECHI ---
        st.markdown("---")
        with st.spinner("Descărcăm datele acțiunilor..."):
            df_ind, df_comm, us_gain, us_lose, eu_gain, eu_lose = get_global_market_data()

        def color_change_val(val):
            if val is None or pd.isna(val): return ''
            color = '#3FB950' if val >= 0 else '#F85149'
            return f'color: {color}'

        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.subheader("📊 Indici Principali")
            st.dataframe(
                df_ind.style.map(color_change_val, subset=['Variație', 'Variație %'])
                .format({'Preț': ro_spec('.2f'), 'Variație': ro_spec('.2f'), 'Variație %': ro_spec('.2f', '%')}),
                width='stretch', hide_index=True
            )
            
        with col_m2:
            st.subheader("🛢️ Mărfuri (Commodities)")
            st.dataframe(
                df_comm.style.map(color_change_val, subset=['Variație', 'Variație %'])
                .format({'Preț': ro_spec('.2f'), 'Variație': ro_spec('.2f'), 'Variație %': ro_spec('.2f', '%')}),
                width='stretch', hide_index=True
            )

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
                }),
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
        
        st.subheader("🇺🇸 Top Mișcări SUA (Blue Chips)")
        c_us1, c_us2 = st.columns(2)
        
        with c_us1:
            st.markdown("**🚀 Top Creșteri (Gainers)**")
            if not us_gain.empty:
                st.dataframe(
                    us_gain[['Instrument', 'Preț', 'Variație %']].style
                    .map(color_change_val, subset=['Variație %'])
                    .format({'Preț': ro_spec('.2f'), 'Variație %': ro_spec('.2f', '%')}),
                    width='stretch', hide_index=True
                )
        
        with c_us2:
            st.markdown("**🔻 Top Scăderi (Losers)**")
            if not us_lose.empty:
                st.dataframe(
                    us_lose[['Instrument', 'Preț', 'Variație %']].style
                    .map(color_change_val, subset=['Variație %'])
                    .format({'Preț': ro_spec('.2f'), 'Variație %': ro_spec('.2f', '%')}),
                    width='stretch', hide_index=True
                )

        st.markdown("---")

        st.subheader("🇪🇺 Top Mișcări EUROPA")
        c_eu1, c_eu2 = st.columns(2)
        
        with c_eu1:
            st.markdown("**🚀 Top Creșteri (Gainers)**")
            if not eu_gain.empty:
                st.dataframe(
                    eu_gain[['Instrument', 'Preț', 'Variație %']].style
                    .map(color_change_val, subset=['Variație %'])
                    .format({'Preț': ro_spec('.2f'), 'Variație %': ro_spec('.2f', '%')}),
                    width='stretch', hide_index=True
                )
        
        with c_eu2:
            st.markdown("**🔻 Top Scăderi (Losers)**")
            if not eu_lose.empty:
                st.dataframe(
                    eu_lose[['Instrument', 'Preț', 'Variație %']].style
                    .map(color_change_val, subset=['Variație %'])
                    .format({'Preț': ro_spec('.2f'), 'Variație %': ro_spec('.2f', '%')}),
                    width='stretch', hide_index=True
                )

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

        tab_bvb, tab_global = st.tabs(["🇷🇴 BVB (Local)", "🌍 Internațional (Global)"])

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
                        return lambda v: "N/A" if v is None else v       # coloană text
                    unit = bvb_units[col]
                    if unit == "%":
                        return ro_spec(".2f", "%")
                    if unit == " lei" or "eps" in col.lower() or "câștig pe acțiune" in col.lower():
                        return ro_spec(".4f", unit)
                    return ro_spec(",.2f", unit)
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
                      help="Câte condiții sunt îndeplinite: preț > SMA 50, preț > SMA 200, SMA 50 > SMA 200. "
                           "Aceeași regulă ca „Moment” din Analiză Companie (sub 200 de ședințe: N/A).")
            r2.metric("Momentum 12-1", _pct(reg["momentum"], 1),
                      help="Randamentul de acum 12 luni până acum o lună (ultima lună exclusă).")
            r3.metric("Scor moment", "N/A" if reg["score"] is None else format_ro(reg["score"], 0),
                      help="Sub-scorul tehnic 0–100 = media trendului (0–1) și a momentumului scalat (−30% → 0, +30% → 1). "
                           "Pragurile au fost testate pe acțiuni BVB, nu pe indici: aici e descriptiv.")

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
            color = lambda v: "" if v is None or pd.isna(v) else ("color: #3FB950" if v > 0 else "color: #F85149" if v < 0 else "")
            st.dataframe(
                view.style.format({"Preț": ro_spec(",.2f", unit), "Variație": ro_spec("+.2f", "%"), "σ": ro_spec("+.1f", "σ")})
                .map(color, subset=["Variație"]),
                width='stretch', hide_index=True,
                column_config={"σ": st.column_config.Column(help="Variația zilei ÷ abaterea standard a variațiilor zilnice din "
                                                                  "ultimele 60 de ședințe (fără ziua curentă; minimum 20). "
                                                                  "Peste 2σ = mișcare neobișnuită pentru acest titlu."),
                               "Valoare": st.column_config.Column(help="Preț × volum în ședință (valoarea tranzacționată), "
                                                                        "în lei la cursul BNR."),
                               "Variație": st.column_config.Column(help="Față de închiderea ședinței precedente, preț neajustat.")})

        def _news_list(items, src_key):
            if not items:
                return False
            st.markdown(compact_html("".join(
                f"""<div style="margin-bottom:10px; border-bottom:1px solid #30363D; padding-bottom:5px;">
                <a href="{html.escape(str(it.get('link', '')), quote=True)}" style="color:#58A6FF; text-decoration:none; font-weight:600;" target="_blank">{html.escape(str(it.get('title', '')))}</a>
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

        tab_bvb, tab_eu, tab_us, tab_news = st.tabs(["🇷🇴 BVB", "🇪🇺 Europa", "🇺🇸 Wall Street", "📰 Știri"])

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
                m2.metric("Mișcarea BET în σ", _sig(bet["sigma"]), help=briefing_calc.sigma_label(bet["sigma"]).capitalize()
                          + f". Variația zilei ÷ abaterea standard a variațiilor din ultimele {bet['n']} ședințe (fără ziua curentă).")
                m3.metric("Lățime (cresc / scad)", f"{bvb_b['up']} / {bvb_b['down']}" if bvb_b["n"] else "N/A",
                          help="Companiile din universul urmărit (tools/univers_bvb.txt: BET Plus, BET-FI, BETAeRO) "
                               "care au tranzacționat în ultima ședință.")
                m4.metric("Valoare tranzacționată", briefing_calc.money_short(bvb_turn["today"], "RON"),
                          None if bvb_turn["ratio"] is None else f"×{format_ro(bvb_turn['ratio'], 2)} față de mediană",
                          delta_color="off",
                          help=f"Suma preț × volum pe universul urmărit în ultima ședință, față de mediana celor {bvb_turn['n']} "
                               "ședințe dinainte (minimum 10). Doar universul, nu toată bursa.")
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
                        f_color = lambda v: "" if v is None or pd.isna(v) else ("color: #3FB950" if v > 0 else "color: #F85149" if v < 0 else "")
                        st.dataframe(
                            fv.style.format({"Preț": ro_spec(",.2f", " lei"), "Variație": ro_spec("+.2f", "%"),
                                             "σ": ro_spec("+.1f", "σ")}, na_rep="N/A")
                            .map(f_color, subset=["Variație"]),
                            width='stretch', hide_index=True, height=min(38 + 35 * len(fv), 600),
                            column_config={"σ": st.column_config.Column(help="Variația zilei ÷ abaterea standard a variațiilor "
                                                                             "din ultimele 60 de ședințe (fără ziua curentă; minimum 20)."),
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
                               help=briefing_calc.sigma_label(mv["sigma"]).capitalize()
                                    + f". σ față de ultimele {mv['n']} ședințe. {kind.capitalize()}."
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
                          help="Companiile din universul urmărit (tools/univers_urmarit_ue.txt: DAX 40, CAC 40, Londra, "
                               "Milano) care au tranzacționat în ultima ședință.")
                m2.metric("Valoare tranzacționată (universul urmărit)", briefing_calc.money_short(eu_turn["today"], "RON"),
                          None if eu_turn["ratio"] is None else f"×{format_ro(eu_turn['ratio'], 2)} față de mediană",
                          delta_color="off",
                          help="Preț × volum pe universul urmărit în ultima ședință, fiecare titlu în lei la cursul BNR al "
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
                        s_color = lambda v: ("color: #3FB950" if str(v).startswith("+") and str(v) != "+0,00%"
                                             else "color: #F85149" if str(v).startswith("-") else "")
                        st.dataframe(fv.style.map(s_color, subset=["Variație"]), width='stretch', hide_index=True,
                                     height=min(38 + 35 * len(fv), 600),
                                     column_config={"σ": st.column_config.Column(help="Variația zilei ÷ abaterea standard a "
                                                                                      "variațiilor din ultimele 60 de ședințe "
                                                                                      "(fără ziua curentă; minimum 20)."),
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
                               help=briefing_calc.sigma_label(mv["sigma"]).capitalize()
                                    + f". σ față de ultimele {mv['n']} ședințe. Indice de preț (fără dividende).")

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
                          help="Volatilitatea anualizată pe 30 de zile implicită în opțiunile pe S&P 500 (puncte procentuale).")
                v2.metric("Percentila VIX pe 1 an", "N/A" if vix_s.empty else format_ro(stress_calc.percentile_rank(vix_s), 0, "%"),
                          help=f"Ce procent din cele {len(vix_s)} închideri din ultimul an au fost ≤ VIX-ul de azi.")
                v3.metric("VIX / VIX3M", "N/A" if vt is None else f"{format_ro(vt['ratio'], 2)} ({vt['state']})",
                          help="Frica pe 30 de zile față de cea pe 3 luni. ≥ 1,00 = stres (structură inversată), 0,90–1,00 = "
                               "tensiune, sub 0,90 = normal. Praguri uzuale de piață, netestate în aplicație.")

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
                    col.metric(f"{a_sym} − {b_sym} (o zi)",
                               "N/A" if rd["diff_pp"] is None else ("+" if rd["diff_pp"] >= 0 else "") + format_ro(rd["diff_pp"], 2, " pp"),
                               help=f"{label}: variația zilei a lui {a_sym} minus a lui {b_sym} (prețuri ajustate). Pozitiv = "
                                    "companiile cu pondere egală (sau cele mici) au făcut mai bine decât indicele ponderat cu "
                                    "capitalizarea, deci mișcarea zilei a fost largă, nu dusă de câțiva giganți.")
                _breadth_bar(briefing_calc.breadth_counts(us_t), f"Universul urmărit ({len(us_t)} simboluri, nu S&P 500)")
                us_turn = briefing_calc.turnover_vs_median(brief["us_close"][us_cols], brief["us_volume"].reindex(columns=us_cols))
                usd_rz = rz_fx.get("USD")
                st.metric("Valoare tranzacționată (universul urmărit)",
                          briefing_calc.money_short(us_turn["today"] * usd_rz if (usd_rz and us_turn["today"] is not None) else None, "RON"),
                          None if us_turn["ratio"] is None else f"×{format_ro(us_turn['ratio'], 2)} față de mediană",
                          delta_color="off",
                          help=f"Preț × volum pe universul urmărit în ultima ședință, în lei la cursul BNR"
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
                    s_color = lambda v: ("color: #3FB950" if str(v).startswith("+") and not str(v).startswith("+0,00")
                                         else "color: #F85149" if str(v).startswith("-") else "")
                    st.dataframe(sv.style.map(s_color, subset=["Variație", "Față de S&P 500"]), width='stretch', hide_index=True,
                                 column_config={
                                     "Variație": st.column_config.Column(help="Față de închiderea precedentă, preț neajustat: în ziua "
                                                                              "ex-dividend scăderea include dividendul."),
                                     "σ": st.column_config.Column(help="Variația zilei ÷ abaterea standard a variațiilor zilnice ale "
                                                                       "ETF-ului din ultimele 60 de ședințe (fără ziua curentă; minimum 20). "
                                                                       "Peste 2σ = mișcare neobișnuită pentru sector (prag netestat)."),
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
                       "cea mai nouă știre despre pozițiile tale (lista în tabul „📰 Știri”). Detaliile sunt mai jos.")
            st.markdown("---")

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
                   f"Rezumatul Zilei); la SUA și Europa, media volumului pe 20 de ședințe ≥ "
                   f"{format_ro(scanner_calc.MIN_AVG_SHARES, 0)} de acțiuni.")

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
                        # Nelichide: BVB sub 50.000 lei azi (ca în Rezumat), restul sub 5.000 de acțiuni în medie
                        if not scanner_calc.liquidity_ok(t, curr_p, curr_vol, avg_vol_20): continue
                        
                        rvol = rs["rvol"]
                        change_pct = rs["change_pct"]
                        
                        # --- MAGIA AI: Interogăm modelul Isolation Forest ---
                        is_anomaly = detect_volume_anomaly_ai(df_t)
                        
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
                            "Alertă AI": "🚨 ATENTIE" if is_anomaly else "-",  # <--- NOUA COLOANĂ AI
                            "Status": "🚀 BREAKOUT" if (rvol > 2.0 and change_pct > 1.5) 
                                     else ("⚠️ PANIC SELL" if (rvol > 2.0 and change_pct < -1.5) 
                                     else ("✅ ACUMULARE" if (rvol > 1.2 and change_pct > 0) else "Normal"))
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
                                
                            # Evidențiem DOAR coloana de AI dacă e anomalie (O colorăm diferit, gen Wall Street Alert)
                            if "ATENTIE" in str(row['Alertă AI']):
                                idx = row.index.get_loc('Alertă AI')
                                styles[idx] += '; color: #FFAB00; font-weight: bold; background-color: rgba(255, 171, 0, 0.2); border: 1px solid #FFAB00;'
                                
                            return styles

                        # --- AFIȘARE TABEL FĂRĂ INDEX ȘI CU STIL ---
                        st.success(f"Găsit: {len(df_filtered)} companii cu volum neobișnuit în {market_choice}.")
                        
                        st.dataframe(
                            df_filtered.style.apply(style_scanner_rows, axis=1).format({
                                "Preț": ro_spec('.2f'),
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
                        
                        st.caption("🟢 **Verde Aprins:** Breakout Confirmat | 🟢 **Verde Pal:** Acumulare Discretă | 🔴 **Roșu:** Panic Sell")
                    else:
                        st.info(f"Nicio acțiune din {market_choice} nu depășește pragul de {threshold}x azi.")
                else:
                    st.warning("Eroare la preluarea datelor. Yahoo Finance ar putea limita cererile.")
    # ==================================================
    # 8. WATCHLIST (FINAL FIX - TIMEZONE PROOF)
    # ==================================================
    elif sectiune == "8. Watchlist":
        st.title("Lista de Urmărire (Watchlist)")
        st.markdown("Monitorizează acțiunile pe care vrei să le cumperi când prețul scade.")

        # --- FORMULAR ADĂUGARE ---
        with st.expander("➕ Adaugă Alertă Nouă", expanded=False):
            with st.form("wl_form"):
                c1, c2, c3 = st.columns([1, 1, 2])
                s_wl = c1.text_input("Simbol (ex: TSLA)").upper()
                p_wl = c2.number_input("Preț Țintă (Target)", min_value=0.0, step=0.1)
                n_wl = c3.text_input("Notă (ex: Suport major, aștept earnings)")
                
                if st.form_submit_button("Adaugă în Listă"):
                    if s_wl and p_wl > 0:
                        if add_to_watchlist(s_wl, p_wl, n_wl):
                            st.success(f"Adăugat {s_wl} la ținta {p_wl}!")
                            st.rerun()
                    else:
                        st.warning("Introdu un simbol și un preț valid.")

        # --- AFIȘARE TABEL ---
        df_wl = load_watchlist()
        
        if not df_wl.empty:
            # 1. Luăm prețurile live pentru toate simbolurile din listă
            tickers_list = df_wl['Symbol'].unique().tolist()
            live_data = pd.Series()

            if tickers_list:
                with st.spinner("Actualizăm prețurile (Global)..."):
                    try:
                        # DESCĂRCARE DATE PE 5 ZILE (pentru siguranță)
                        data_bulk = yf.download(tickers_list, period="5d", progress=False, auto_adjust=True)['Close']
                        
                        # Tratare caz un singur ticker (Series -> DataFrame)
                        if isinstance(data_bulk, pd.Series):
                             data_bulk = data_bulk.to_frame(name=tickers_list[0])
                        
                        # --- LOGICĂ DE EXTRAGERE PREȚ VALID INDIFERENT DE ORĂ ---
                        current_prices = {}
                        
                        for col in data_bulk.columns:
                            # Luăm coloana și ștergem valorile goale (NaN)
                            valid_values = data_bulk[col].dropna()
                            
                            if not valid_values.empty:
                                # Luăm ultima valoare existentă (chiar dacă e de ieri)
                                current_prices[col] = valid_values.iloc[-1]
                            # fără date: simbolul lipsește din serie și apare „N/A”, nu preț 0
                        
                        # Convertim înapoi în Series pentru restul codului
                        live_data = pd.Series(current_prices)
                        # --------------------------------------------------------

                    except Exception as e:
                        logger.warning("Watchlist, prețuri live: %s", e)
                        live_data = pd.Series(dtype=float)

            # 2. Construim tabelul final
            display_rows = []
            for index, row in df_wl.iterrows():
                sym = row['Symbol']
                target = smart_to_float(row['TargetPrice'])  # acceptă și formatul românesc ("25,5")
                note = row['Notes']
                
                # Preț sau țintă lipsă rămân lipsă („N/A”), nu 0; distanța e față de țintă, ca în cardul din analiză.
                curr, target, dist_pct, is_buy = watchlist_status(live_data.get(sym), target)
                if curr is None:
                    status = "Fără preț"
                elif target is None:
                    status = "Fără țintă"
                else:
                    status = "✅ CUMPĂRĂ" if is_buy else "⏳ Așteaptă"
                
                display_rows.append({
                    "Simbol": sym,
                    "Preț Curent": curr,
                    "Preț Țintă 🎯": target,
                    "Distanță (%)": dist_pct,
                    "Recomandare": status,
                    "Notă": note,
                    "_is_buy": is_buy 
                })
            
            df_res = pd.DataFrame(display_rows)

            # 3. Stilizare și Afișare
            def highlight_buy(row):
                if row['_is_buy']:
                    return ['background-color: rgba(63, 185, 80, 0.2); font-weight: bold'] * len(row)
                else:
                    return [''] * len(row)

            st.dataframe(
                df_res.style.apply(highlight_buy, axis=1)
                .format({"Preț Curent": ro_spec('.2f'), "Preț Țintă 🎯": ro_spec('.2f'), "Distanță (%)": ro_spec('+.2f', '%')}, na_rep="N/A"),
                width='stretch',
                height=500,
                column_config={
                    "_is_buy": None, 
                },
                hide_index=True 
            )
            
            # Buton ștergere
            with st.expander("🗑️ Șterge din listă"):
                del_sym = st.selectbox("Alege simbol de șters:", tickers_list)
                if st.button("Șterge"):
                    if remove_from_watchlist(del_sym):
                        st.warning(f"Șters {del_sym}.")
                        st.rerun()

        else:
            st.info("Nu ai nicio acțiune în Watchlist. Folosește formularul de sus.")

if __name__ == "__main__":
    main()

