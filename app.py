import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import feedparser
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
from analytics.macro import yoy_pct, real_rate
from analytics.portfolio import value_positions, portfolio_curve as build_portfolio_curve
from analytics import fundamentals as fund
from analytics.risk import beta_benchmark, beta_weekly, jensen_alpha
from analytics.peers import PEERS, METRICS as PEER_METRICS, peer_region, peer_list, peer_medians, versus_median
from data.bvb_sheet import parse_bvb_sheet, bvb_symbol, reprice as reprice_bvb
from data.helpers import num, close_frame, slice_window, now_ro, struct_time_utc_to_ro, smart_to_float, parse_ecb_csv, positive_or_none, entry_target_view

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
        print(f"DEBUG: preț live {ticker}: HTTP {response.status_code}")
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as e:
        print(f"DEBUG: preț live {ticker} indisponibil: {e}")
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
        print(f"DEBUG: prețuri live indisponibile: {e}")
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
        print(f"DEBUG: dată de știre neinterpretabilă: {e}")
    return now_ro()

# --- FUNCȚIE NOUĂ DE PARSARE INTELIGENTĂ (SENIOR FIX) ---
# smart_to_float este acum în data/helpers.py (aceeași logică, cu teste în tests/test_helpers.py)
def format_large_currency(val):
    """Formatează numerele mari (Trilioane, Miliarde) pentru afișare string."""
    try:
        if isinstance(val, str):
            val = smart_to_float(val)
        
        if val is None or val == 0: return "-"
        if val >= 1e12: return f"$ {val/1e12:.2f} T"
        if val >= 1e9: return f"$ {val/1e9:.2f} B"
        if val >= 1e6: return f"$ {val/1e6:.2f} M"
        return f"$ {val:,.2f}"
    except:
        return str(val)
    
def format_num(val, is_pct=False):
    """Formatare afișare (folosește smart_to_float intern)"""
    if val is None: return "N/A"
    # Asigurăm conversia dacă vine string
    if isinstance(val, str):
        val = smart_to_float(val)
    if pd.isna(val): return "N/A"
        
    if is_pct: return f"{val * 100:.2f}%"
    if val >= 1e12: return f"{val/1e12:.2f} T"
    if val >= 1e9: return f"{val/1e9:.2f} B"
    if val >= 1e6: return f"{val/1e6:.2f} M"
    return f"{val:,.2f}"

def format_amount(val):
    """Sumă din situațiile financiare, scalată (mld / mil), cu semn. None sau NaN -> 'N/A'."""
    if val is None or pd.isna(val):
        return "N/A"
    sign = "-" if val < 0 else ""
    a = abs(val)
    if a >= 1e12: return f"{sign}{a/1e12:,.2f} T"
    if a >= 1e9: return f"{sign}{a/1e9:,.2f} mld"
    if a >= 1e6: return f"{sign}{a/1e6:,.2f} mil"
    return f"{sign}{a:,.2f}"

def calculate_portfolio_beta(portfolio_curve, benchmark_ticker="SPY"):
    """Calculează Beta și Corelația globală a întregului portofoliu."""
    if portfolio_curve is None or portfolio_curve.empty:
        return 0.0, 0.0
    try:
        start_date = portfolio_curve.index[0]
        bench_data = yf.download(benchmark_ticker, start=start_date, progress=False)['Close']
        if isinstance(bench_data, pd.DataFrame): bench_data = bench_data.iloc[:, 0]
        
        combined = pd.DataFrame({'Port': portfolio_curve, 'Bench': bench_data}).ffill().dropna()
        returns = combined.pct_change().dropna()
        
        # Corelația globală (0 la 1)
        correlation = returns['Port'].corr(returns['Bench'])
        
        # Beta (Sensibilitatea la piață)
        variance = returns['Bench'].var()
        beta = returns['Port'].cov(returns['Bench']) / variance if variance != 0 else 1.0
        
        return correlation, beta
    except:
        return 0.0, 1.0

def get_macro_interpretation(ticker_data):
    """
    Motor IA Macro Profesional: Analiză corelații, inflație și impact sectorial.
    """
    try:
        def get_chg(t): return ticker_data[t]['Close'].pct_change().iloc[-1] if t in ticker_data else 0

        usd = get_chg('DX-Y.NYB')
        gold = get_chg('GC=F')
        oil = get_chg('CL=F')
        copper = get_chg('HG=F')
        tnx_chg = get_chg('^TNX') # Yield 10Y
        
        v = []

        # --- 1. CORELAȚII ACTIVE (DINAMICE) ---
        if usd < -0.003 and gold > 0.003:
            v.append("🟡 **AUR:** Refugiu activ. Dolarul slăbește, confirmând rolul aurului de protecție a puterii de cumpărare.")
        
        if usd > 0.003 and oil < -0.005:
            v.append("🛢️ **PETROL:** Presiune valutară. Dolarul puternic scumpește barilul pentru importatori, reducând cererea.")

        if copper > 0.01:
            v.append("🏗️ **CUPRU:** Semnal expansiune. Creșterea metalelor industriale indică activitate industrială robustă.")

        # --- 2. IMPACT SECTORIAL & INFLAȚIE (PROFESIONAL) ---
        if tnx_chg > 0.01: # Dacă dobânzile cresc
            v.append("🚀 **SECTOR BANCAR:** Impact Pozitiv. Creșterea yield-urilor îmbunătățește marjele nete de dobândă (spread).")
            v.append("📉 **TECH & GROWTH:** Risc ridicat. Dobânzile mari scad valoarea prezentă a profiturilor viitoare (model DCF).")
            v.append("🚩 **SMALL CAPS:** Vulnerabilitate crescută la refinanțarea datoriilor cu dobândă variabilă.")
        elif tnx_chg < -0.01: # Dacă dobânzile scad
            v.append("🟢 **TECH & REAL ESTATE:** Mediu favorabil. Costul capitalului scade, stimulând evaluările activelor imobiliare și tehnologice.")
        
        # --- 3. ALERTA "BLACK SWAN" (CRERATĂ DE TINE) ---
        if gold > 0.01 and usd > 0.005 and get_chg('^VIX') > 0.10:
            v.append("🚨 **ALERTA BLACK SWAN:** Fuga masivă către siguranță detectată (Aur+Dolar+VIX în creștere). Risc sistemic ridicat!")

        return v if v else ["⚖️ **ECHILIBRU:** Corelațiile macro sunt stabile azi. Mișcările reflectă fundamentele individuale ale activelor."]
    except:
        return ["⚠️ Date insuficiente pentru procesarea corelațiilor macro."]

def calculate_sortino_ratio(portfolio_curve, risk_free_rate=0.04):
    """
    Calculează Sortino Ratio izoland exclusiv deviația standard negativă.
    Standardul de aur pentru hedge-fund-uri.
    """
    if portfolio_curve is None or len(portfolio_curve) < 5:
        return 0.0
        
    try:
        returns = portfolio_curve.pct_change().dropna()
        # Randament mediu anualizat
        mean_return = returns.mean() * 252
        
        # Păstrăm doar randamentele sub zero (Downside Risk)
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) < 2:
            return 0.0 # Prea puține date negative pentru calcul
            
        # Volatilitate negativă anualizată
        downside_std = np.sqrt((negative_returns**2).sum() / len(returns)) * np.sqrt(252)
        
        if downside_std == 0: return 0.0
        
        return (mean_return - risk_free_rate) / downside_std
    except:
        return 0.0

def calculate_investment_rating_pro(info, inst_pct, rvol, spread_val, mos_val):
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
        details.append(f"✅ **Evaluare:** Marjă de siguranță excelentă ({mos_val:.1f}%). Preț subevaluat.")
    elif mos_val < -10:
        score -= 15
        details.append(f"🚨 **Evaluare:** Supraevaluare semnificativă. Risc ridicat de corecție.")

    # 3. ANALIZA MACRO
    if spread_val < 0:
        score -= 20
        details.append("🚨 **Macro:** Curbă 10Y-3M inversată. Risc sistemic de recesiune detectat.")
    else:
        score += 5
        details.append("✅ **Macro:** Mediul economic este favorabil expansiunii.")

    # 4. SĂNĂTATE FINANCIARĂ
    roe = num(info, 'returnOnEquity')
    if roe is not None and roe > 0.20:
        score += 10
        details.append(f"🚀 **Eficiență:** ROE excepțional ({roe*100:.1f}%).")
    
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
        print(f"DEBUG: preț țintă din watchlist pentru {symbol}: {exc}")
    return None

@st.cache_data(ttl=21600, show_spinner=False)
def get_peer_rows(region, sector):
    """Indicatorii comparabililor dintr-o regiune și un sector (listele din analytics/peers.py).

    Întoarce (rânduri, simboluri fără date). În cache 6 ore, pe (regiune, sector): fiecare
    comparabil costă o cerere la endpoint-ul Yahoo cel mai limitat, iar înainte tabelul era
    recitit la fiecare interacțiune cu pagina. Valorile lipsă rămân None (afișate N/A).
    """
    rows, failed = [], []
    for p_sym in PEERS.get(region, {}).get(sector or "", []):
        _yf_limiter.wait_if_needed()
        try:
            inf = yf.Ticker(p_sym).info or {}
        except Exception as e:
            print(f"DEBUG: comparabil {p_sym} indisponibil: {e}")
            failed.append(p_sym)
            continue
        row = {"Simbol": p_sym, "Capitalizare": num(inf, 'marketCap'), "Monedă": inf.get('currency') or ""}
        for key, label, mult in PEER_METRICS:
            value = num(inf, key)
            row[label] = None if value is None else value * mult
        if all(row[label] is None for _, label, _ in PEER_METRICS):
            failed.append(p_sym)       # simbol delistat, redenumit sau refuzat de Yahoo
            continue
        rows.append(row)
    return rows, failed

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
        except: continue
    all_news.sort(key=lambda x: x['date_obj'], reverse=True)
    return all_news

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

def get_company_news_rss(symbol):
    rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
    news_list = []
    try:
        feed = feedparser.parse(rss_url)
        if not feed.entries: return []
        for entry in feed.entries[:7]:
            dt = parse_date(entry)
            news_list.append({
                "title": entry.title,
                "link": entry.link,
                "publisher": "Yahoo Finance",
                "date_str": dt.strftime("%Y-%m-%d %H:%M")
            })
    except: return []
    return news_list

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
    data = yf.download(list(tickers.values()), period="5y", group_by='ticker', progress=False)
    return tickers, data

@st.cache_data(ttl=3600)
def get_market_data():
    try:
        spy = yf.Ticker("SPY").history(period="1y")['Close']
        return spy
    except: return None

@st.cache_data(ttl=3600)
def get_risk_free_rate():
    """Descarcă randamentul titlurilor de stat SUA pe 10 ani (^TNX) ca proxy pentru Risk Free Rate."""
    try:
        tnx = yf.Ticker("^TNX").history(period="1d")
        if not tnx.empty:
            return tnx['Close'].iloc[-1] / 100
    except:
        pass
    return 0.04 # Fallback la 4%

def calculate_alpha(stock_hist, beta):
    try:
        spy = get_market_data()
        if spy is None or stock_hist is None: return None
        
        # Sincronizare lungime date
        min_len = min(len(spy), len(stock_hist))
        stock_close = stock_hist['Close'].iloc[-min_len:]
        spy_close = spy.iloc[-min_len:]
        
        # Calcul randament total
        ret_stock = (stock_close.iloc[-1] / stock_close.iloc[0]) - 1
        ret_market = (spy_close.iloc[-1] / spy_close.iloc[0]) - 1
        
        # Rata dinamică
        risk_free = get_risk_free_rate()
        
        if beta is None: beta = 1.0
        
        # Formula CAPM: Alpha = R_stock - (R_rf + Beta * (R_market - R_rf))
        alpha = ret_stock - (risk_free + beta * (ret_market - risk_free))
        return alpha
    except: return None

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
            print(f"DEBUG: {attr} indisponibil pentru {symbol}: {e}")
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
        print(f"DEBUG: benchmark {symbol} indisponibil: {e}")
    return pd.Series(dtype=float)


STATEMENT_RATIO_LABELS = {
    "trailingPE": "P/E", "priceToBook": "P/BV", "trailingEps": "EPS", "bookValue": "valoare contabilă/acțiune",
    "returnOnEquity": "ROE", "returnOnAssets": "ROA", "profitMargins": "marjă netă",
    "operatingMargins": "marjă operațională", "debtToEquity": "datorii/capital", "currentRatio": "current ratio",
    "quickRatio": "quick ratio", "totalRevenue": "venituri", "netIncomeToCommon": "profit net",
    "operatingCashflow": "flux din exploatare", "totalDebt": "datorie totală", "totalCash": "numerar",
}


@st.cache_data(ttl=900, show_spinner=False)
def load_bvb_fundamentals():
    """Indicatorii din foaia `BVB` a `portofoliu_db`, pe simbol (vezi data/bvb_sheet.py).
    Dict gol dacă foaia nu poate fi citită. Doar citire: foaia nu e modificată niciodată de aici."""
    try:
        ws = connect_to_gsheets("BVB")
        if not ws:
            return {}
        return parse_bvb_sheet(ws.get_all_values())
    except Exception as e:
        print(f"DEBUG: foaia BVB nu a putut fi citită: {e}")
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
    valutei; când beta e din Yahoo, rămâne calculul existent (față de SPY).
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
                                      f"rată fără risc {rf_val * 100:.2f}% [{rf_label}]")
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

    if out["beta"] is not None:
        out["alpha"] = calculate_alpha(hist, out["beta"])
        out["alpha_label"] = "față de S&P 500 (SPY), ultimul an; rată fără risc: titluri SUA 10 ani"
    else:
        out["alpha_label"] = "indisponibil: lipsește beta"
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
        print(f"DEBUG: randament AAA (FRED) indisponibil: {e}")
    return None, "Randamentul obligațiunilor AAA (FRED) nu a putut fi citit."


def get_graham_yield(currency):
    """Y din formula revizuită a lui Graham, în procente, pentru valuta acțiunii.

    USD: randamentul AAA din FRED. Alte valute: nu există o serie AAA locală, deci se
    aproximează ca rată fără risc a valutei + marja AAA față de titlurile SUA pe 10 ani;
    eticheta spune că e proxy. (None, motiv) dacă lipsește oricare componentă.
    """
    aaa, aaa_label = get_aaa_yield()
    if aaa is None:
        return None, aaa_label
    cur = (currency or "").upper()
    if cur == "USD":
        return aaa, aaa_label
    rf_us, _ = get_risk_free_for_currency("USD")
    rf_local, rf_label = get_risk_free_for_currency(cur)
    if rf_us is None or rf_local is None:
        return None, f"Nu pot aproxima randamentul AAA în {cur or 'valuta necunoscută'} (lipsește o rată fără risc)."
    spread = max(aaa - rf_us * 100, 0.0)
    return rf_local * 100 + spread, f"proxy: {rf_label} + marja AAA din SUA ({spread:.2f} pp)"


@st.cache_data(ttl=21600, show_spinner=False)
def get_risk_free_for_currency(currency):
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
            closes = yf.Ticker("^TNX").history(period="5d")['Close'].dropna()
            if len(closes) and _valid(float(closes.iloc[-1])):
                return float(closes.iloc[-1]) / 100, f"titluri SUA 10 ani (^TNX, {closes.index[-1]:%d.%m.%Y})"
        except Exception as e:
            print(f"DEBUG: ^TNX indisponibil: {e}")
        return None, "Randamentul titlurilor SUA pe 10 ani (^TNX) nu a putut fi citit."
    if cur == "EUR":
        try:
            end = datetime.today()
            serie = web.DataReader('IRLTLT01DEM156N', 'fred', end - timedelta(days=200), end).iloc[:, 0].dropna()
            if len(serie) and _valid(float(serie.iloc[-1])):
                return float(serie.iloc[-1]) / 100, f"Bund 10 ani (FRED IRLTLT01DEM156N, {serie.index[-1]:%m.%Y})"
        except Exception as e:
            print(f"DEBUG: Bund 10 ani (FRED) indisponibil: {e}")
        return None, "Randamentul Bund pe 10 ani (FRED) nu a putut fi citit."
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
                print(f"DEBUG: BCE (randament RO 10 ani): HTTP {resp.status_code}")
        except requests.RequestException as e:
            print(f"DEBUG: BCE (randament RO 10 ani) indisponibil: {e}")
        return None, "Randamentul titlurilor de stat românești pe 10 ani (BCE) nu a putut fi citit."
    return None, f"Nu există încă o sursă automată pentru rata fără risc în {cur or 'valuta necunoscută'}."

# --- Pune acest bloc sus, lângă celelalte funcții (calculate_alpha, etc.) ---

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
        print(f"DEBUG: scor sănătate incomplet: {e}")

    # Fără niciun indicator de bilanț, scorul ar rămâne 5/10 („mediu") fără nicio bază.
    if all(num(info, k) is None for k in ('debtToEquity', 'returnOnEquity', 'currentRatio')):
        return None, pros, cons
    
    return max(1, min(10, score)), pros, cons

def get_sector_benchmarks(sector):
    """Definește pragurile 'normale' în funcție de industrie."""
    # Benchmarks implicite (Standard)
    benchmarks = {"pe_threshold": 20, "roe_target": 0.12, "de_max": 150}
    
    # Ajustări pe sectoare specifice
    sector_maps = {
        "Technology": {"pe_threshold": 35, "roe_target": 0.20, "de_max": 100},
        "Energy": {"pe_threshold": 12, "roe_target": 0.10, "de_max": 200},
        "Financial Services": {"pe_threshold": 15, "roe_target": 0.10, "de_max": 400},
        "Utilities": {"pe_threshold": 18, "roe_target": 0.08, "de_max": 300}
    }
    
    return sector_maps.get(sector, benchmarks)
    
# Actualizăm funcția de audit să folosească aceste praguri
def generate_advanced_audit_v2(info, alpha, beta, h_score):
    """
    Audit Instituțional Complet (6 Piloni).
    Include protecție pentru date lipsă (NVS, BVB) și toate interpretările.
    """
    sector = info.get('sector') or 'sector necunoscut'
    limits = get_sector_benchmarks(sector)
    
    # Valorile lipsă rămân None și pilonul respectiv e sărit. Înainte deveneau 0 și apăreau
    # verdicte false („ROE 0%: eficiență sub-optimă", „0% datorii: structură sănătoasă").
    pe = num(info, 'trailingPE') or 0
    roe = num(info, 'returnOnEquity')
    de = num(info, 'debtToEquity')
    cr = num(info, 'currentRatio') or 0
    
    safe_alpha = alpha if alpha is not None else 0
    safe_beta = beta
    
    audit = []
    missing = [name for name, v in (("P/E", num(info, 'trailingPE')), ("ROE", roe), ("datorii", de),
                                     ("lichiditate", num(info, 'currentRatio')), ("beta", beta)) if v is None]
    if missing:
        audit.append(f"ℹ️ **DATE LIPSĂ:** {', '.join(missing)}. Pilonii respectivi nu sunt evaluați.")

    # --- 1. EVALUARE VS SECTOR ---
    if pe > 0:
        rel_price = pe / limits['pe_threshold']
        if rel_price < 0.8 and roe > limits['roe_target']:
            audit.append(f"💰 **EVALUARE:** Subevaluată în {sector}. P/E {pe:.1f} este atractiv față de media de {limits['pe_threshold']}.")
        elif rel_price > 1.3:
            audit.append(f"⚖️ **EVALUARE:** Scumpă raportat la sector ({pe:.1f} vs {limits['pe_threshold']}).")
        else:
            audit.append(f"📊 **EVALUARE:** Preț corect în contextul {sector} (P/E {pe:.1f}).")

    # --- 2. PROFITABILITATE & EFICIENȚĂ ---
    if roe is not None and roe > 0.25:
        audit.append(f"🚀 **PROFITABILITATE:** Eficiență de elită (ROE {roe*100:.1f}%). Management performant.")
    elif roe is not None and roe < 0.10:
        audit.append(f"📉 **PROFITABILITATE:** Eficiență sub-optimă ({roe*100:.1f}%). Capitalul nu produce suficient.")

    # --- 3. SOLVABILITATE (DATORII) ---
    if de is not None and de > limits['de_max']:
        audit.append(f"🚩 **SOLVABILITATE:** Îndatorare peste limita sectorului ({de:.1f}%). Risc structural ridicat.")
    elif de is not None:
        audit.append(f"✅ **SOLVABILITATE:** Structură de capital sănătoasă ({de:.1f}% debt/equity).")

    # --- 4. LICHIDITATE (CASH-FLOW) ---
    if cr > 0:
        if cr < 1.0:
            audit.append(f"❌ **LICHIDITATE:** Critică ({cr:.2f}). Firma depinde de finanțări externe pe termen scurt.")
        elif cr > 1.5:
            audit.append(f"💧 **LICHIDITATE:** Solidă ({cr:.2f}). Există suficient 'cash' pentru siguranță.")

    # --- 5. PERFORMANȚĂ PIAȚĂ (ALPHA) ---
    alpha_p = safe_alpha * 100
    if safe_alpha > 0.02:
        audit.append(f"📈 **PERFORMANȚĂ:** Alpha Pozitiv ({alpha_p:.1f}%). Randament peste indexul de referință.")
    elif safe_alpha < -0.02:
        audit.append(f"🥀 **PERFORMANȚĂ:** Subperformanță ({alpha_p:.1f}%). Activul pierde în fața pieței.")

    # --- 6. RISC DE PIAȚĂ (BETA) ---
    if safe_beta is None:
        pass
    elif safe_beta > 1.3:
        audit.append(f"🎢 **VOLATILITATE (Beta {safe_beta:.2f}):** Risc ridicat. Mișcări mult mai ample decât piața.")
    elif safe_beta < 0.8:
        audit.append(f"🛡️ **VOLATILITATE (Beta {safe_beta:.2f}):** Profil defensiv. Stabilă în perioade de criză.")
    else:
        audit.append(f"⚖️ **VOLATILITATE (Beta {safe_beta:.2f}):** Mișcare sincronizată cu piața generală.")

    return audit

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
        print(f"DEBUG: info indisponibil pentru {getattr(t, 'ticker', '?')}: {e}")
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
                    print(f"DEBUG: fast_info.{attr} indisponibil: {e}")
    except Exception as e:
        print(f"DEBUG: fast_info indisponibil: {e}")
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
        print(f"DEBUG: earnings indisponibil: {e}")
        return None


@st.cache_data(ttl=900)
def get_stock_data(symbol):
    try:
        # Lăsăm yfinance să gestioneze conexiunea automat pentru a evita eroarea curl_cffi
        t = yf.Ticker(symbol)
        
        # Încercăm să descărcăm datele istorice
        hist = t.history(period="5y")

        # Fallback pentru Bursa de Valori București (BVB)
        if hist.empty and not symbol.endswith(".RO"):
            sym_ro = symbol + ".RO"
            t_ro = yf.Ticker(sym_ro)
            hist_ro = t_ro.history(period="5y")
            if not hist_ro.empty:
                return hist_ro, _safe_info(t_ro), _safe_earnings(t_ro), sym_ro

        if hist.empty:
            return None, None, None, symbol

        return hist, _safe_info(t), _safe_earnings(t), symbol

    except Exception as e:
        # Dacă apare o eroare de conexiune, o afișăm în consolă pentru debug, nu blocăm UI-ul
        print(f"DEBUG: Eroare yfinance pentru {symbol}: {e}")
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

def plot_correlation_matrix(tickers):
    """Generează matricea de corelație și un raport vizual premium."""
    if len(tickers) < 2: return None
    try:
        # 1. Descărcare și calcul
        data = yf.download(tickers, period="1y", progress=False)['Close']
        returns = data.pct_change().dropna()
        corr_matrix = returns.corr()
        
        # 2. Text Heatmap
        text_matrix = []
        for i in range(len(corr_matrix)):
            row_text = []
            for j in range(len(corr_matrix)):
                val = corr_matrix.iloc[i, j]
                if i == j: label = "1.00"
                elif val > 0.8: label = f"{val:.2f}<br>⚠️ Risc"
                elif val > 0.5: label = f"{val:.2f}<br>Moderat"
                else: label = f"{val:.2f}<br>✅ OK"
                row_text.append(label)
            text_matrix.append(row_text)

        # 3. Grafic Plotly optimizat
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale=[[0.0, '#00FF00'], [0.5, '#ffffff'], [1.0, '#01464D']],
            zmin=-1, zmax=1,
            text=text_matrix,
            texttemplate="%{text}",
            hovertemplate="Corelație: %{z:.2f}<extra></extra>"
        ))
        
        fig.update_layout(
            height=400, template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(side="bottom")
        )

        st.plotly_chart(fig, width='stretch')
        
        # 4. RAPORT VIZUAL STILIZAT (Cards)
        st.markdown("### 📋 Analiză Strategica a Diversificării")
        
        cols = st.columns(len(corr_matrix.columns) - 1 if len(corr_matrix.columns) <= 3 else 2)
        col_idx = 0

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                score = corr_matrix.iloc[i, j]
                t1, t2 = corr_matrix.columns[i], corr_matrix.columns[j]
                
                # Logică Culori și Iconițe
                if score > 0.75:
                    color, icon, label = "#F85149", "🚫", "Diversificare Slabă"
                    bg_light = "rgba(248, 81, 73, 0.1)"
                elif score > 0.40:
                    color, icon, label = "#DBAB09", "⚠️", "Diversificare Moderată"
                    bg_light = "rgba(219, 171, 9, 0.1)"
                else:
                    color, icon, label = "#3FB950", "🛡️", "Diversificare Optimă"
                    bg_light = "rgba(63, 185, 80, 0.1)"

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
                                <span style="color: {color}; font-family: monospace; font-size: 18px; font-weight: bold;">{score:.2f}</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                col_idx += 1

        return True
    except Exception as e:
        st.error(f"Eroare Matrice: {e}")
        return None
    
def render_benchmark_comparison(portfolio_curve, bench_ticker="SPY", bench_name="S&P 500"):
    """Compară performanța portofoliului cu benchmark-ul folosind procente în tooltip."""
    if portfolio_curve is None or portfolio_curve.empty:
        return
    
    try:
        start_date = portfolio_curve.index[0]
        spy_data = yf.download(bench_ticker, start=start_date, progress=False)['Close']
        
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
        
        st.plotly_chart(fig, width='stretch')

        # Afișare Alpha Card (Rămâne neschimbat, dar folosim variabilele noi)
        alpha_color = "#3FB950" if alpha > 0 else "#F85149"
        st.markdown(f"""
            <div style="background-color: #161B22; padding: 20px; border-radius: 12px; border-top: 4px solid {alpha_color}; text-align: center;">
                <h4 style="color: #8B949E; margin-bottom: 5px;">Alpha (Diferență față de {bench_name})</h4>
                <h1 style="color: {alpha_color}; margin: 0;">{alpha:+.2f}%</h1>
                <p style="color: #8B949E; font-size: 14px;">Portofoliu: {port_ret:+.2f}% | {bench_name}: {bench_ret:+.2f}%</p>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Benchmark Eroare: {str(e)}")

# --- FUNCȚII NOI PENTRU REZUMAT ZILNIC (DAILY BRIEFING) ---

def generate_market_narrative(ticker_data, symbol, name):
    try:
        if isinstance(ticker_data.columns, pd.MultiIndex):
            if symbol in ticker_data.columns.levels[0]:
                close = ticker_data[symbol]['Close']
            else:
                return f"Date indisponibile pentru {name}.", 0, 0
        else:
            close = ticker_data['Close']

        close = close.dropna()
        if len(close) < 2: return "Date insuficiente.", 0, 0

        curr = close.iloc[-1]
        prev = close.iloc[-2]
        change_pct = ((curr - prev) / prev) * 100
        
        if change_pct > 1.0:
            trend = "o creștere puternică"
            sentiment = "pozitiv"
        elif change_pct > 0.2:
            trend = "o creștere moderată"
            sentiment = "ușor optimist"
        elif change_pct > -0.2:
            trend = "o evoluție stabilă"
            sentiment = "neutru"
        elif change_pct > -1.0:
            trend = "o scădere moderată"
            sentiment = "precaut"
        else:
            trend = "o scădere semnificativă"
            sentiment = "negativ"
            
        text = f"**{name}** a înregistrat {trend} de **{change_pct:.2f}%**, închizând la {curr:,.2f}. Sentimentul pieței este {sentiment}."
        return text, change_pct, curr
    except Exception as e:
        return f"Nu s-au putut genera date pentru {name}.", 0, 0

@st.cache_data(ttl=1800)
def get_daily_briefing_data():
    bvb_tickers = [
        'TVBETETF.RO', 'TLV.RO', 'SNP.RO', 'H2O.RO', 'TRP.RO', 'FP.RO', 'ATB.RO', 'BIO.RO', 'ALW.RO', 'AST.RO', 
        'EBS.RO', 'IMP.RO', 'SNG.RO', 'BRD.RO', 'ONE.RO', 'TGN.RO', 'SNN.RO', 'DIGI.RO', 'M.RO', 'EL.RO', 'MILK.RO', 
        'SMTL.RO', 'AROBS.RO', 'AQ.RO', 'ASC.RO', 'ARS.RO', 'BRK.RO', 'IARV.RO', 'TTS.RO', 'WINE.RO', 'TEL.RO', 'DN.RO', 'AG.RO', 
        'BENTO.RO', 'PE.RO', 'COTE.RO', 'PBK.RO', 'SAFE.RO', 'TBK.RO', 'CFH.RO', 'SFG.RO'
    ]
    bvb_data = yf.download(bvb_tickers, period="1mo", group_by='ticker', progress=False)
    
    us_tickers = [
        '^GSPC', '^DJI', '^IXIC', '^VIX', 
        'NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'CG', 'SNOW', 'CEG', 'ASML', 'ARM', 'CRWV', 'FN', 'SNDK', 'MU', 
        'AMD', 'INTC', 'NFLX', 'JPM', 'BAC', 'SOFI', 'MS', 'HON', 'V', 'T', 'INOD', 'MA', 'MDB', 'AIG', 'AXP', 'SCHW', 'NET', 'BIIB', 
        'WMT', 'KO', 'PEP', 'PG', 'DXCM', 'COP', 'OXY', 'DVN', 'LNG', 'GPOR', 'UUUU', 'FSLR', 'VG', 'TTE', 'RIO', 'BHP', 'D', 'VALE', 'METC', 'MP', 'LLY', 'AMGN', 'XOM', 'CVX', 
        'PLTR', 'PANW', 'ANET', 'QCOM', 'ORCL', 'TSM', 'GS', 'CRM', 'WFC', 'NVO', 'NVS', 'MCD', 'SPCX', 'SMR', 'CMG', 'OKLO', 'SNY', 'JNJ', 'BA', 'GD', 'RTX', 'LMT', 'KTOS', 'PM', 'COO', 'MRK', 'PFE', 'C'
    ]
    us_data = yf.download(us_tickers, period="1mo", group_by='ticker', progress=False)
    
    return bvb_data, us_data

def get_bvb_stats(data, tickers):
    stats = []
    
    for t in tickers:
        if t in ['TVBETETF.RO', '^GSPC', '^DJI', '^IXIC', '^VIX']: continue 
        
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if t not in data.columns.levels[0]: continue
                df_t = data[t]
            else:
                continue

            series_close = df_t['Close'].dropna()
            series_vol = df_t['Volume'].dropna()
            
            if len(series_close) >= 2:
                curr = series_close.iloc[-1]
                prev = series_close.iloc[-2]
                pct = ((curr - prev) / prev) * 100
                
                vol = series_vol.iloc[-1] if not series_vol.empty else 0
                
                stats.append({
                    'Simbol': t.replace('.RO', ''), 
                    'Preț': curr,
                    'Variație': pct,
                    'Volum': vol
                })
        except Exception as e:
            continue
    
    df = pd.DataFrame(stats)
    if df.empty: 
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    gainers = df.sort_values('Variație', ascending=False).head(10)
    losers = df.sort_values('Variație', ascending=True).head(10)
    volume_leaders = df.sort_values('Volum', ascending=False).head(10)
    
    return gainers, losers, volume_leaders

def calculate_fear_greed_proxy(data):
    try:
        if isinstance(data.columns, pd.MultiIndex):
             vix_series = data['^VIX']['Close'].dropna()
             sp500_close = data['^GSPC']['Close'].dropna()
        else:
             return 50, "Neutral 😐", 0

        if vix_series.empty or sp500_close.empty:
             return 50, "Neutral 😐", 0

        current_vix = vix_series.iloc[-1]
        
        vix_score = 100 - ((current_vix - 10) / (40 - 10) * 100)
        vix_score = max(0, min(100, vix_score))
        
        curr_sp = sp500_close.iloc[-1]
        mean_5d = sp500_close.mean()
        
        diff_pct = (curr_sp / mean_5d) - 1
        mom_score = 50 + (diff_pct * 100 * 25) 
        mom_score = max(0, min(100, mom_score))
        
        final_score = (vix_score * 0.6) + (mom_score * 0.4)
        
        if final_score >= 75: label = "Extreme Greed 🤑"
        elif final_score >= 55: label = "Greed 😋"
        elif final_score >= 45: label = "Neutral 😐"
        elif final_score >= 25: label = "Fear 😨"
        else: label = "Extreme Fear 😱"
        
        return final_score, label, current_vix
    except Exception as e:
        return 50, "Neutral 😐", 0
    
def calculate_bvb_sentiment(bvb_data):
    """
    Algoritm avansat de sentiment BVB (v2.0).
    Integrează: Preț vs Medie (5z/20z), Volume Relativ și Volatilitatea.
    """
    try:
        # 1. Extragere date proxy BET
        if isinstance(bvb_data.columns, pd.MultiIndex):
             bet_df = bvb_data['TVBETETF.RO'].dropna()
        else:
             return 50, "Neutral ⚖️"

        if bet_df.empty or len(bet_df) < 5: return 50, "Neutral ⚖️"

        # 2. Parametri Preț
        curr_bet = bet_df['Close'].iloc[-1]
        mean_5d = bet_df['Close'].rolling(window=5).mean().iloc[-1]
        mean_20d = bet_df['Close'].mean() # Media întregului set descărcat (5 zile în codul tău)
        
        # Deviația față de termen scurt (5 zile)
        dev_short = ((curr_bet / mean_5d) - 1) * 100
        
        # 3. Parametri Volum (Factorul de confirmare)
        curr_vol = bet_df['Volume'].iloc[-1]
        mean_vol = bet_df['Volume'].mean()
        # Relative Volume (RVOL) - dacă e > 1.5, mișcarea este instituțională, nu retail
        rvol = curr_vol / mean_vol if mean_vol > 0 else 1.0

        # 4. Calcul Scor de Bază (0-100)
        # Am calibrat formula: +/- 1.5% mișcare pe BVB este considerată acum prag critic
        score = 50 + (dev_short * 33.3) 
        
        # 5. Ajustare dinamică cu Volum
        # Dacă prețul scade pe volum mare, scorul scade și mai mult (Panic confirmation)
        # Dacă prețul crește pe volum mare, scorul crește și mai mult (Buying frenzy)
        if rvol > 1.3:
            if dev_short > 0: score += (rvol * 5) # Boost optimism
            else: score -= (rvol * 10) # Penalizare panică (vânzare pe volum mare e mai gravă)

        score = max(0, min(100, score))

        # 6. Verdict bazat pe praguri profesionale
        if score >= 80: label = "Optimism Excesiv (Frenzy) 🚀"
        elif score >= 60: label = "Sentiment Pozitiv 📈"
        elif score >= 40: label = "Echilibru / Stabilitate ⚖️"
        elif score >= 20: label = "Precauție (Vânzări sub presiune) ⚠️"
        else: label = "Panică / Capitulare 🚨"
        
        return score, label
    except:
        return 50, "Neutral ⚖️"
    
# --- FUNCȚII PORTOFOLIU (RESCRISE PENTRU GOOGLE SHEETS) ---
def load_portfolio():
    """Citește datele din Google Sheets folosind Secrets."""
    sheet = connect_to_gsheets()  # Sheet1 implicit = portofoliu
    if sheet:
        try:
            data = sheet.get_all_records()
            return pd.DataFrame(data)
        except:
            return pd.DataFrame()
    return pd.DataFrame()

# --- FUNCȚII WATCHLIST (CORECȚIE BUG) ---
def load_watchlist():
    """Citește datele din foaia 'watchlist'."""
    ws = connect_to_gsheets("watchlist")  # Direct, fără sheet.spreadsheet.worksheet()
    if ws:
        try:
            data = ws.get_all_records()
            return pd.DataFrame(data)
        except Exception as e:
            return pd.DataFrame()
    return pd.DataFrame()

def add_to_watchlist(symbol, target, note):
    """Adaugă o intrare nouă în watchlist."""
    ws = connect_to_gsheets("watchlist")
    if ws:
        try:
            ws.append_row([symbol, float(target), note])
            st.cache_data.clear()
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
                st.cache_data.clear()
                return True
        except:
            pass
    return False

def add_trade(s, q, p, d, c):
    """Adaugă tranzacția direct în Google Sheets."""
    sheet = connect_to_gsheets()
    if sheet:
        row = [s, str(d), float(q), float(p), c]
        sheet.append_row(row)
        st.cache_data.clear()

@st.cache_data(ttl=300)
def get_portfolio_history_data(tickers):
    if not tickers: return pd.DataFrame()
    data = yf.download(tickers, period="5y", group_by='ticker')
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
        hist_data = yf.download(tickers, period="5y", group_by='ticker', progress=False)

    # close_frame tratează toate formele întoarse de yf.download (un simbol sau mai multe)
    closes = close_frame(hist_data, tickers)

    df_result, total_daily_pl_abs, total_daily_pl_pct, notes = value_positions(positions, current_prices, closes)
    portfolio_curve, curve_notes = build_portfolio_curve(positions, closes)
    notes.update(curve_notes)

    # Fereastră calendaristică (1A = un an de date, ~252 de ședințe), nu 365 de rânduri
    portfolio_curve = slice_window(portfolio_curve, history_range)
    
    return df_result, portfolio_curve, total_daily_pl_abs, total_daily_pl_pct, notes

from scipy.stats import norm # Adaugă acest import la începutul fișierului main.py

def calculate_risk_metrics(portfolio_curve, confidence_level=0.95):
    """Calculează Max Drawdown, Sharpe Ratio, VaR și Volatilitate Anualizată."""
    if portfolio_curve is None or portfolio_curve.empty or len(portfolio_curve) < 5:
        return 0.0, 0.0, 0.0, 0.0
    
    try:
        # 1. Max Drawdown
        rolling_max = portfolio_curve.cummax()
        drawdown = (portfolio_curve - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        # 2. Sharpe Ratio & Volatilitate
        returns = portfolio_curve.pct_change().dropna()
        if returns.std() == 0: return max_dd, 0.0, 0.0, 0.0
        
        # Volatilitate Anualizată (Standard Deviation * sqrt(252 zile de tranzacționare))
        volatility_ann = returns.std() * np.sqrt(252)
        
        rf_daily = 0.04 / 252 
        sharpe = np.sqrt(252) * ((returns.mean() - rf_daily) / returns.std())
        
        # 3. Value at Risk (VaR)
        mu, sigma = returns.mean(), returns.std()
        var_pct = norm.ppf(1 - confidence_level, mu, sigma)
        var_abs = var_pct * portfolio_curve.iloc[-1]
        
        return max_dd, sharpe, var_abs, volatility_ann
    except:
        return 0.0, 0.0, 0.0, 0.0

@st.cache_data(ttl=3600)
def get_portfolio_sectors(df_current):
    """
    ARHITECTURĂ #5: Versiune batch optimizată.
    ÎNAINTE: N apeluri seriale yf.Ticker(sym).info = 30-60 secunde
    ACUM: 1 apel yf.Tickers() pentru toate = 3-5 secunde
    """
    if df_current.empty: return pd.DataFrame()

    tickers_list = df_current['Symbol'].unique().tolist()
    total_mkt_val = df_current['MarketValue'].sum()

    # UN SINGUR apel pentru TOATE simbolurile
    _yf_limiter.wait_if_needed()
    try:
        bulk_data = yf.Tickers(" ".join(tickers_list))
    except Exception:
        bulk_data = None

    sector_map = {}
    for sym in tickers_list:
        try:
            sec = "Nedefinit"
            if bulk_data and sym in bulk_data.tickers:
                sec = bulk_data.tickers[sym].info.get('sector', 'Nedefinit') or 'Nedefinit'
            val = df_current[df_current['Symbol'] == sym]['MarketValue'].sum()
            sector_map[sec] = sector_map.get(sec, 0) + val
        except:
            val = df_current[df_current['Symbol'] == sym]['MarketValue'].sum()
            sector_map['Nedefinit'] = sector_map.get('Nedefinit', 0) + val

    df_sec = pd.DataFrame(list(sector_map.items()), columns=['Sector', 'MarketValue'])
    df_sec['Pondere %'] = (df_sec['MarketValue'] / total_mkt_val) * 100
    return df_sec.sort_values(by='Pondere %', ascending=False)

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
    
    us_stocks = ['NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'CG', 'SNOW', 'CEG', 'ASML', 'ARM', 'CRWV', 'FN', 'SNDK', 'MU', 
                 'AMD', 'INTC', 'NFLX', 'JPM', 'BAC', 'SOFI', 'MS', 'HON', 'V', 'INOD', 'MA', 'MDB', 'AIG', 'AXP', 'SCHW', 'NET', 'SPCX', 'BIIB', 
                 'WMT', 'KO', 'PEP', 'PG', 'DXCM', 'GPOR', 'COP', 'OXY', 'VG', 'DVN', 'LNG', 'T', 'UUUU', 'FSLR', 'TTE', 'RIO', 'BHP', 'D', 'VALE', 'METC', 'MP', 'LLY', 'AMGN', 'XOM', 'CVX', 
                 'PLTR', 'PANW', 'ANET', 'QCOM', 'ORCL', 'TSM', 'CMG', 'GS', 'CRM', 'WFC', 'NVO', 'NVS', 'MCD', 'SMR', 'OKLO', 'SNY', 'JNJ', 'BA', 'GD', 'RTX', 'LMT', 'KTOS', 'PM', 'COO', 'MRK', 'PFE', 'C']
    eu_stocks = ['SAP.DE', 'MC.PA', 'ASML', 'SIE.DE', 'TTE.PA', 'AIR.PA', 'ALV.DE', 'DTE.DE', 'VOW3.DE', 'BAYN.DE', 'UCG.MI', 'ENR.DE', 'DBK.DE', 'ULVR.L', 'REL.L', 
                 'BMW.DE', 'BNP.PA', 'SAN.PA', 'OR.PA', 'GLNCY', 'MBG.DE', 'BSP.DE', 'RHM.DE', 'ZAL.DE', 'LDO.MI', 'RNO.PA', 'BA.L', 'DGE.L', 'SHEL.L', 'BATS.L', 'RACE.MI', 'AZN', 'HSBA.L']

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
                if prev:
                    change = price - prev
                    pct = (change / prev) * 100
                else: change = 0; pct = 0
                
                data.append({
                    'Instrument': name, 'Simbol': sym, 'Preț': price, 'Variație': change, 'Variație %': pct
                })
            except: continue
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
        data = yf.download(list(sectors_map.keys()), period="5d", group_by='ticker', progress=False)
        
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
            except:
                continue
                
        df = pd.DataFrame(results)
        if not df.empty:
            # Sortăm crescător pentru ca pe graficul orizontal (Plotly) cele mai mari să fie sus
            df = df.sort_values(by='Variație %', ascending=True)
        return df
    except Exception as e:
        print(f"Eroare sectoare: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_credit_risk_data(period="1y"):
    """
    Descarcă datele pentru piața de credit pe o perioadă specificată.
    """
    try:
        data = yf.download(['HYG', 'IEF'], period=period, progress=False)['Close']
        if 'HYG' in data.columns and 'IEF' in data.columns:
            ratio = data['HYG'] / data['IEF']
            return ratio.dropna()
        return pd.Series()
    except Exception as e:
        print(f"Eroare date bonduri: {e}")
        return pd.Series()

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
        data = yf.download(list(tickers.values()), period="3mo", progress=False)['Close']
        
        # Redenumim coloanele cu numele noastre intuitive
        rename_map = {v: k for k, v in tickers.items()}
        data = data.rename(columns=rename_map)
        
        # Calculăm randamentele și matricea de corelație (Pearson)
        returns = data.pct_change().dropna()
        corr_matrix = returns.corr()
        return corr_matrix
    except Exception as e:
        print(f"Eroare Cross-Asset: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_index_technical_levels(symbol):
    """Calculează nivelurile critice (SMA 50, SMA 200) pentru un indice."""
    try:
        # Descărcăm 1 an de date pentru a avea destule zile pentru SMA 200
        df = yf.download(symbol, period="1y", progress=False)['Close']
        if isinstance(df, pd.DataFrame):
            df = df.iloc[:, 0]
            
        df = df.dropna()
        if len(df) < 200: return None # Nu avem destule date
        
        curr_price = df.iloc[-1]
        sma_50 = df.rolling(window=50).mean().iloc[-1]
        sma_200 = df.rolling(window=200).mean().iloc[-1]
        
        # Calculăm distanța procentuală
        dist_50 = ((curr_price - sma_50) / sma_50) * 100
        dist_200 = ((curr_price - sma_200) / sma_200) * 100
        
        return {
            "price": curr_price,
            "sma50": sma_50,
            "sma200": sma_200,
            "dist50": dist_50,
            "dist200": dist_200
        }
    except:
        return None

def calculate_market_breadth(data, tickers):
    """Calculează Lățimea Pieței (Avans vs Declin)"""
    adv, dec, flat = 0, 0, 0
    for t in tickers:
        # Ignorăm indicii majori (ca să numărăm doar companiile)
        if t.startswith('^') or t == 'TVBETETF.RO': continue 
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if t not in data.columns.levels[0]: continue
                series = data[t]['Close'].dropna()
            else:
                series = data['Close'].dropna()
            
            if len(series) >= 2:
                change = series.iloc[-1] - series.iloc[-2]
                if change > 0.001: adv += 1
                elif change < -0.001: dec += 1
                else: flat += 1
        except: continue
    
    total = adv + dec + flat
    if total == 0: return 0, 0, 0, 0, 0, 0
    return adv, dec, flat, (adv/total*100), (dec/total*100), (flat/total*100)

@st.cache_data(ttl=3600)
def get_volume_analysis(symbol):
    """Calculează intensitatea volumului față de media de 20 de zile."""
    try:
        t = yf.Ticker(symbol)
        # Avem nevoie de 1 lună de date pentru o medie mobilă de 20 de zile
        df = t.history(period="1mo")
        if df.empty or len(df) < 20: return None
        
        curr_vol = df['Volume'].iloc[-1]
        avg_vol = df['Volume'].rolling(window=20).mean().iloc[-1]
        
        # Intensitatea (Raportul)
        intensity = curr_vol / avg_vol
        return {
            "current": curr_vol,
            "average": avg_vol,
            "intensity": intensity
        }
    except:
        return None

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
                    trend = f"{change:+.0f}K"
                elif code in ['UNRATE', 'FEDFUNDS', 'UMCSENT']:
                    change = val_curr - val_prev
                    trend = f"{change:+.2f} pct."
                else:
                    change = ((val_curr - val_prev) / val_prev) * 100
                    trend = f"{change:+.2f}%"
                
                yoy = yoy_pct(col_data) if code in yoy_codes else None
                    
                results.append({
                    "Indicator": name,
                    "Valoare Curentă": val_curr,
                    "Lună Precedentă": val_prev,
                    "Evoluție (MoM)": trend,
                    "Evoluție (An/An)": f"{yoy:+.2f}%" if yoy is not None else "—",
                    "_simbol": code,
                    "_change_raw": change,
                    "_yoy": yoy
                })
        except Exception:
            continue # Dacă un simbol are probleme, trecem la următorul fără să oprim aplicația
            
    return pd.DataFrame(results)

def interpret_macro_data_ai(df_macro):
    """
    Analiză Macro de nivel Senior Fund Manager.
    Sintetizează 12 indicatori FRED pentru a detecta schimbările de paradigmă economică.
    """
    if df_macro.empty: return []
    
    bullets = []
    m = {row['_simbol']: row for _, row in df_macro.iterrows()}
    
    # --- HELPER: Extragere Valori ---
    def gv(sym): return m.get(sym, {}).get('Valoare Curentă', 0)
    def gc(sym): return m.get(sym, {}).get('_change_raw', 0)

    # 1. ANALIZA INFLAȚIEI (Cea mai mare frică a pieței)
    pce = gc('PCEPILFE') # Core PCE MoM
    cpi_core = gc('CPILFESL') # Core CPI MoM
    fed_rate = gv('FEDFUNDS')
    
    # Calculăm Rata Reală (Dobânda Fed - Inflația an/an)
    # Dacă e pozitivă și mare, Fed-ul "strânge de gât" economia.
    # CPIAUCSL este un INDICE (nivel ~320), nu o rată: inflația e variația lui an/an.
    cpi_yoy = m.get('CPIAUCSL', {}).get('_yoy')
    cpi_yoy = float(cpi_yoy) if cpi_yoy is not None and pd.notna(cpi_yoy) else None
    real_rate_pct = real_rate(fed_rate if 'FEDFUNDS' in m else None, cpi_yoy)

    if pce > 0.2 or cpi_core > 0.3:
        bullets.append("🔥 **INFLAȚIE REZISTENTĂ:** Datele Core PCE și CPI indică prețuri 'lipicioase'. Fed-ul nu poate tăia dobânzile fără riscul unui al doilea val inflaționar. Rămâneți prudenți pe sectorul imobiliar și Tech cu evaluări mari.")
    elif pce <= 0.15 and real_rate_pct is not None and real_rate_pct > 2:
        bullets.append("❄️ **DEZINFLAȚIE CU DOBÂNZI REALE MARI:** Inflația scade, dar dobânzile rămân sus. Aceasta este o rețetă pentru 'Pivot' – momentul în care Fed va trebui să taie rapid pentru a nu strivi economia.")

    # 2. RADIOGRAFIA PIEȚEI MUNCII (Motorul consumului)
    nfp = gc('PAYEMS')
    adp = gc('ADPCHGS')
    unrate = gv('UNRATE')
    jolts_chg = gc('JTSJOL')

    # Corelația ADP vs NFP
    if nfp > 200 and adp < 100:
        bullets.append("🔄 **DIVERGENȚĂ PRIVAT-STAT:** Angajările oficiale (NFP) sunt susținute de sectorul public, în timp ce sectorul privat (ADP) frânează. Este un semnal 'Late Cycle' – economia reală încetinește sub suprafață.")
    
    # Regula Sahm & Tightness
    if gc('UNRATE') > 0.1 and unrate > 4.0:
        bullets.append("🚨 **RECESIUNEA 'BATE LA UȘĂ':** Șomajul crește constant. Istoric, o creștere de 0.5% a ratei șomajului de la minimul anului declanșează Regula Sahm (recesiune inevitabilă).")
    
    if jolts_chg < -400:
        bullets.append("📉 **CAPITULAREA ANGAJATORILOR:** Scăderea masivă a joburilor vacante (JOLTS) arată că firmele au tăiat bugetele de expansiune. Presiunea pe creșterea salariilor va scădea, ajutând inflația, dar lovind consumul.")

    # 3. CONSUMUL ȘI SENTIMENTUL (Inima PIB-ului)
    retail = gc('RSAFS')
    sentiment = gc('UMCSENT')

    if retail < 0 and sentiment < 0:
        bullets.append("⚠️ **STAGNARE CONSUM:** Americanii nu mai au încredere în viitor și au tăiat cheltuielile de retail. Deoarece consumul este 70% din PIB, acesta este un semnal de alarmă pentru profiturile companiilor din S&P 500.")
    elif retail > 0.5 and sentiment < -2.0:
        bullets.append("💸 **SPENDING DE NECESITATE:** Vânzările cresc, dar încrederea scade. Oamenii cheltuie mai mult pe aceleași bunuri (din cauza inflației), nu pentru că au surplus. Este un semnal negativ pentru marjele de profit.")

    # 4. INDICATORI AVANSAȚI (Leading Indicators)
    housing = gc('HOUST')
    ind_prod = gc('INDPRO')

    if housing < -3.0 and ind_prod < -0.3:
        bullets.append("🏗️ **FRÂNĂ INDUSTRIALĂ ȘI IMOBILIARĂ:** Cele mai sensibile sectoare la dobânzi sunt în contracție. Acest duo anticipează de obicei o scădere a PIB-ului în următoarele 2-3 trimestre.")
    elif housing > 2.0:
        bullets.append("🏠 **RELIENȚĂ IMOBILIARĂ:** Sectorul construcțiilor sfidează dobânzile mari. Acest lucru oferă un suport solid economiei și sugerează că o recesiune 'hard' este puțin probabilă acum.")

    # 5. VERDICTUL MASTER (SINTEZA)
    if unrate < 4.0 and pce < 0.2 and retail > 0:
        bullets.append("🌟 **REGIM GOLDILOCKS:** Nici prea cald, nici prea rece. Joburi sunt, inflația e stabilă, consumul merge. Este mediul ideal pentru investiții în acțiuni (Equity Bull Market).")
    elif unrate > 4.2 and pce > 0.25:
        bullets.append("🌪️ **SCENARIU DE COȘMAR (STAGFLAȚIE):** Șomaj în creștere + Inflație persistentă. Este singurul mediu în care portofoliile diversificate eșuează. Cash-ul și Aurul sunt singurele refugii.")

    return bullets

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
                        st.markdown(f"""
                        <div class="news-card">
                            <a href="{item['link']}" class="news-title" target="_blank">{item['title']}</a>
                            <div class="news-meta"><b>{item['source']}</b> • {item['date_str']}</div>
                            <div style="color:#B0B8C4; font-size:14px; line-height: 1.5;">{item['summary'][:250]}...</div>
                        </div>
                        """, unsafe_allow_html=True)
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
            # Definim variabilele globale de diagnostic la început pentru a fi disponibile peste tot
            try:
                # Înlocuim 2Y=F (delistat) cu ^IRX (3-Month Yield)
                t_10y = yf.Ticker("^TNX").fast_info.last_price
                t_3m = yf.Ticker("^IRX").fast_info.last_price
                # Spread-ul preferat de FED: 10Y - 3M
                spread = t_10y - t_3m
            except:
                spread = 0.5 # Fallback neutru
            
        if hist is None or hist.empty:
            st.error("Simbol invalid sau date indisponibile.")
        else:
            # 1. Informații Generaley
            st.markdown(f"## {info.get('longName') or real_sym}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Sector", info.get('sector') or 'N/A')
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
                sl_display = f"{sl_price:.2f}" if sl_price else "N/A"
                sl_risk_display = f"Risc: -{dist_sl:.1f}%{sl_hit_note}" if sl_price else "Istoric insuficient pentru ATR"
                
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
                                <h2 style="color:{t_color}; margin:5px 0; font-size:28px;">{curr_price:.2f}</h2>
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
                 m1.metric(f"Interval ({time_opt})", f"{curr_price:.2f} {info.get('currency', '')}", f"{diff_val:.2f} ({diff_pct:.2f}%)")
                 m2.metric("Evoluție Azi", f"{curr_price:.2f}", f"{day_val:.2f} ({day_pct:.2f}%)")

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
            st.plotly_chart(fig, width='stretch')

            # --- SEPARATORUL SOLICITAT (ADAUGĂ ACEASTĂ LINIE) ---
            st.markdown("---")

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
            if info.get('_bvb_indicators'):
                with st.expander(f"📄 Toți indicatorii din foaia BVB pentru {bvb_symbol(real_sym)}, față de piață"):
                    st.dataframe(pd.DataFrame(
                        [(row[0].strip(), row[1] if row[2] is not None else "N/A", row[3], row[4]) for row in info['_bvb_indicators']],
                        columns=["Indicator", bvb_symbol(real_sym), "Media BVB (din foaie)", "Mediana BVB (calculată)"]),
                        hide_index=True, width='stretch')
                    st.caption("Valorile companiei sunt cele din foaie, la prețul actualizării ei. Media e coloana A a foii; mediana e calculată "
                               "din aceleași companii și nu e trasă de extreme (un singur P/E foarte mare ridică media, nu și mediana).")
            risk_stats = resolve_beta_alpha(real_sym, info, hist)
            beta_val, alpha_val = risk_stats["beta"], risk_stats["alpha"]
            de_ratio = info.get('debtToEquity')
            de_display = f"{de_ratio:.2f}%" if de_ratio is not None else "N/A"

            with st.container():
                c_eval, c_prof, c_indat, c_risc = st.columns(4)
                with c_eval:
                    st.markdown("**Evaluare & Dividende**")
                    st.metric("P/E Ratio", format_num(info.get('trailingPE')))
                    st.metric("Forward P/E", format_num(info.get('forwardPE')))
                    div_rate = info.get('dividendRate')
                    div_display = f"{div_rate} ({ (div_rate/curr_price*100):.2f}%)" if (div_rate and curr_price) else "N/A"
                    st.metric("Dividend (Randament)", div_display)
                    st.metric("P/BV", format_num(info.get('priceToBook')))
                    gn_calc = (info.get('trailingPE', 0) or 0) * (info.get('priceToBook', 0) or 0)
                    st.metric("GN (Graham)", f"{gn_calc:.2f}" if gn_calc > 0 else "N/A")
                    st.metric("EPS", format_num(info.get('trailingEps')))
                    st.metric("Val. Contabilă/Acțiune", format_num(info.get('bookValue')))
                with c_prof:
                    st.markdown("**Profitabilitate**")
                    st.metric("ROA", format_num(info.get('returnOnAssets'), True))
                    st.metric("ROE", format_num(info.get('returnOnEquity'), True))
                    st.metric("Marjă Netă", format_num(info.get('profitMargins'), True))
                    st.metric("Marjă Operațională", format_num(info.get('operatingMargins'), True))
                with c_indat:
                    st.markdown("**Îndatorare**")
                    st.metric("Datorii/Capital", de_display)
                    st.metric("Current Ratio", format_num(info.get('currentRatio')))
                    st.metric("Quick Ratio", format_num(info.get('quickRatio')))
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
                                text=df_plot['Net Margin'].apply(lambda x: f"{x:.1f}%"),
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
                            
                            st.plotly_chart(fig, width='stretch')

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
            p_sector = info.get('sector')
            own_row = {"Simbol": real_sym, "Capitalizare": num(info, 'marketCap'), "Monedă": info.get('currency') or ""}
            for p_key, p_label, p_mult in PEER_METRICS:
                p_val = num(info, p_key)
                own_row[p_label] = None if p_val is None else p_val * p_mult

            peer_rows, peer_failed = [], []
            if peer_list(real_sym, p_sector):
                with st.spinner("Se citesc comparabilii (o singură dată la 6 ore pe sector)..."):
                    all_rows, peer_failed = get_peer_rows(p_region, p_sector)
                peer_rows = [r for r in all_rows if r["Simbol"].upper() != real_sym.upper()]
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
                    p_col.metric(p_title, p_fmt.format(p_own), "fără mediană de comparație", delta_color="off")
                else:
                    p_col.metric(p_title, p_fmt.format(p_own),
                                 f"{p_diff * 100:+.0f}% față de mediana {p_fmt.format(p_med)} (n={p_n})", delta_color="off")

            st.write("")

            # --- PASUL 2: tabelul ---
            if p_region == "BVB":
                st.info("Pentru BVB comparația se face cu media și mediana pieței din foaia BVB: vezi expanderul "
                        "„Toți indicatorii din foaia BVB” de sub Indicatori Fundamentali.")
            elif not peer_list(real_sym, p_sector):
                st.info("Nu există o listă de comparabili pentru acest simbol: "
                        + ("Yahoo nu a trimis sectorul." if not p_sector else
                           f"sectorul „{p_sector}” sau piața simbolului nu are listă definită."))
            elif not peer_rows:
                st.warning("Yahoo nu a trimis date pentru niciun comparabil (probabil o limitare temporară).")
                if st.button("🔄 Reîncearcă citirea comparabililor", key="retry_peers"):
                    get_peer_rows.clear()
                    st.rerun()
            else:
                st.markdown(f"**🔍 Comparabili: sectorul „{p_sector}”, {'SUA' if p_region == 'US' else 'Europa'}**")
                median_row = {"Simbol": "Mediana comparabililor", "Capitalizare": None, "Monedă": ""}
                median_row.update({label: medians[label][0] for _, label, _ in PEER_METRICS})
                df_peers = pd.DataFrame([own_row, median_row] + sorted(
                    peer_rows, key=lambda r: -(r["Capitalizare"] or 0)))
                df_peers["Capitalizare"] = [
                    "" if cap is None else f"{format_num(cap)} {cur}".strip()
                    for cap, cur in zip(df_peers["Capitalizare"], df_peers["Monedă"])]
                df_peers = df_peers.drop(columns=["Monedă"])
                st.dataframe(df_peers.style.format({
                    "P/E": "{:.1f}", "P/BV": "{:.2f}", "ROE (%)": "{:.1f}%", "ROA (%)": "{:.1f}%",
                    "Marjă netă (%)": "{:.1f}%", "Datorii/Capital (%)": "{:.0f}%"
                }, na_rep="N/A").apply(
                    lambda row: ["font-weight: bold; background-color: #21262D" if row.name < 2 else "" for _ in row], axis=1),
                    width='stretch', hide_index=True)
                p_counts = ", ".join(f"{label.replace(' (%)', '')} n={medians[label][1]}" for _, label, _ in PEER_METRICS)
                p_note = (f"Mediana e calculată din {len(peer_rows)} comparabili, fără {real_sym}; observații pe indicator: {p_counts}. "
                          "La P/E și P/BV valorile negative sunt excluse. Eșantionul e mic și ales manual (companii mari): "
                          "arată unde se situează compania, nu dacă e scumpă sau ieftină.")
                if p_region == "EU":
                    p_note += " Capitalizările sunt în moneda fiecărei burse; rapoartele nu depind de monedă."
                if peer_failed:
                    p_note += " Fără date de la Yahoo: " + ", ".join(peer_failed) + "."
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
                st.metric("Preț Țintă (Mediu)", f"{target} {info.get('currency','USD')}" if target else "N/A")

            with col_an_right:
                st.markdown("""<div class="fin-card"><h4>🆚 Raportări vs Așteptări</h4></div>""", unsafe_allow_html=True)
                if earn_df is not None and not earn_df.empty:
                    def style_surprise(val):
                        color = '#3FB950' if val > 0 else '#F85149' if val < 0 else '#8B949E'
                        return f'color: {color}; font-weight: bold'
                    e_disp = earn_df[['epsEstimate', 'epsActual', 'epsDifference', 'surprisePercent']].copy()
                    e_disp.columns = ['Estimare', 'Realizat', 'Diferență', 'Surpriză %']
                    st.dataframe(e_disp.style.applymap(style_surprise, subset=['Surpriză %']).format({'Estimare': '{:.2f}', 'Realizat': '{:.2f}', 'Diferență': '{:.2f}', 'Surpriză %': '{:.2%}'}), width='stretch')
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
                            <h1 style="color:{v_color}; margin:10px 0;">{rvol:.2f}x</h1>
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
                    st.write(f"Variație Preț: **{day_change:+.2f}%**")
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
                        <h2 style='margin:10px 0; color:{i_col};'>{inst_percent:.2f}%</h2>
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
                    st.plotly_chart(fig_mini, width='stretch')

                # Rândul 2: Tabelul Nominal (Balenele)
                st.markdown("#### 🏛️ Top 10 Deținători Instituționali")
                if not df_inst_list.empty:
                    st.dataframe(
                        df_inst_list.style.format({
                            'Deținere (%)': '{:.2f}%',
                            'Valoare ($)': '${:,.0f}'
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
                    f"Reper: FCF a crescut cu {dcf_in['fcf_cagr'] * 100:.1f}% pe an între "
                    f"{dcf_in['fcf_history'].index[0]:%Y} și {dcf_in['fcf_history'].index[-1]:%Y} "
                    f"({len(dcf_in['fcf_history'])} ani raportați)."
                )

            use_wacc = False
            if wacc_res is not None:
                use_wacc = ctrl2.toggle(
                    f"Folosește WACC calculat ({wacc_res['wacc'] * 100:.1f}%)",
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
            if dcf_in["fcf_average"] is not None:
                avg_label = f"Media ultimilor {dcf_in['fcf_average_years']} ani fiscali"
                fcf_choice = st.radio(
                    "FCF de pornire în DCF", ["FCF curent", avg_label], horizontal=True, key="v_final_fcf_base",
                    help="FCF curent = ultimele 4 trimestre (sau ultimul an fiscal). Media anilor fiscali netezește "
                         "un vârf de investiții sau un an atipic. Alegerea rămâne valabilă și când schimbi simbolul."
                )
                if fcf_choice == avg_label:
                    fcf_base, fcf_base_label = dcf_in["fcf_average"], avg_label.lower()
            if fcf_distortion and not dcf_na:
                st.warning(f"⚠️ DCF: {fcf_distortion} Compară cu varianta pe media anilor fiscali.")

            # --- LOGICĂ REACTIVĂ ---
            # 1. Graham, formula revizuită: V = EPS × (8,5 + 2g) × 4,4 / Y (analytics/fundamentals.py).
            #    None = nu se aplică (EPS ≤ 0 sau lipsește randamentul AAA); nu se înlocuiește cu 0.
            graham_y, graham_y_label = get_graham_yield(t_curr)
            graham_calc = fund.graham_revised(num(info, 'trailingEps'), growth_val, graham_y)
            graham_num = fund.graham_number(num(info, 'trailingEps'), num(info, 'bookValue'))
            if graham_calc is not None:
                graham_note = (f"Y = {graham_y:.2f}%" + (f" · creștere plafonată la {fund.GRAHAM_MAX_GROWTH:.0f}%"
                                                         if growth_val > fund.GRAHAM_MAX_GROWTH else ""))
            elif (num(info, 'trailingEps') or 0) <= 0:
                graham_note = "EPS lipsă sau negativ"
            else:
                graham_note = graham_y_label
            graham_num_txt = f"Nr. Graham: {graham_num:.2f}" if graham_num is not None else "Nr. Graham: N/A"
            
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

                with cv1:
                    st.markdown(_value_card("Preț curent", f"{price_f:.2f}", "#30363D", unit=t_curr), unsafe_allow_html=True)

                with cv2:
                    if graham_calc is not None:
                        diff_g = ((price_f - graham_calc) / graham_calc) * 100
                        g_col = "#3FB950" if price_f < graham_calc else "#F85149"
                        st.markdown(_value_card(
                            "Graham (formula revizuită)", f"{graham_calc:.2f}", g_col,
                            verdict=f'{"SUBEVALUAT" if price_f < graham_calc else "SUPRAEVALUAT"} ({abs(diff_g):.1f}%)',
                            note=f"{graham_note} · {graham_num_txt}"), unsafe_allow_html=True)
                    else:
                        st.markdown(_value_card("Graham (formula revizuită)", "N/A", "#30363D",
                                                verdict=graham_note, note=graham_num_txt), unsafe_allow_html=True)

                with cv3:
                    if dcf_calc is not None and dcf_calc > 0:
                        diff_d = ((price_f - dcf_calc) / dcf_calc) * 100
                        d_col = "#3FB950" if price_f < dcf_calc else "#F85149"
                        st.markdown(_value_card(
                            "Valoare justă (DCF pe FCF)", f"{dcf_calc:.2f}", d_col,
                            verdict=f'{"SUBEVALUAT" if price_f < dcf_calc else "SUPRAEVALUAT"} ({abs(diff_d):.1f}%)',
                            note=f"scont {discount_rate * 100:.1f}% · g terminal {gterm_val:.1f}%"), unsafe_allow_html=True)
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
                sens.index = [f"scont {r * 100:.1f}%" for r in r_grid]
                sens.columns = [f"g {g * 100:.1f}%" for g in g_grid]

                def _sens_color(v):
                    if pd.isna(v) or not price_f:
                        return "color: #8B949E"
                    return "color: #3FB950" if v > price_f else "color: #F85149"

                st.dataframe(sens.style.format("{:.2f}", na_rep="N/A").map(_sens_color), width='stretch')
                st.caption(
                    f"Verde = peste prețul curent ({price_f:.2f} {t_curr}), roșu = sub. Rândul din mijloc e rata de scont folosită. "
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
                    kd_txt = f"{wacc_res['kd'] * 100:.2f}%" if wacc_res['kd'] is not None else "N/A"
                    tax_txt = f"{wacc_res['tax'] * 100:.1f}%" if wacc_res['tax'] is not None else "N/A"
                    st.write(
                        f"WACC = **{wacc_res['wacc'] * 100:.2f}%** "
                        f"= {wacc_res['w_e'] * 100:.0f}% × cost capital propriu {wacc_res['ke'] * 100:.2f}% "
                        f"+ {wacc_res['w_d'] * 100:.0f}% × cost datorie {kd_txt} × (1 − impozit {tax_txt})"
                    )
                    st.write(
                        f"Cost capital propriu (CAPM) = rată fără risc {wacc_res['rf'] * 100:.2f}% [{rf_label}] "
                        f"+ beta {wacc_res['beta']:.2f} ({beta_label}) × primă de risc {wacc_res['erp'] * 100:.1f}% (ipoteză fixă)"
                    )
                    for wacc_note in wacc_res["notes"]:
                        st.caption(f"Aproximare: {wacc_note}")
                else:
                    st.write(f"WACC indisponibil. Rata fără risc: {rf_label if rf_val is None else f'{rf_val * 100:.2f}% [{rf_label}]'}")
                st.write(f"Rata de scont folosită în calcul: **{discount_rate * 100:.2f}%** ({'WACC calculat' if use_wacc else 'aleasă manual'}).")

                if dcf_res["flows"]:
                    st.markdown("**Proiecția pe 5 ani**")
                    df_flows = pd.DataFrame(dcf_res["flows"])
                    df_flows = pd.DataFrame({
                        "An": df_flows["year"],
                        "Creștere": df_flows["growth"].map(lambda v: f"{v * 100:.1f}%"),
                        "FCF proiectat": df_flows["fcf"].map(format_amount),
                        "Valoare actualizată": df_flows["pv"].map(format_amount),
                    })
                    st.dataframe(df_flows, hide_index=True, width='stretch')
                    st.write(
                        f"Valoarea întreprinderii {format_amount(dcf_res['enterprise_value'])} "
                        f"= fluxuri actualizate {format_amount(dcf_res['pv_fcf'])} "
                        f"+ valoare terminală actualizată {format_amount(dcf_res['pv_terminal'])} "
                        f"({dcf_res['terminal_share'] * 100:.0f}% din total). "
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
                az_value = f"{altman_res['value']:.2f}" if altman_res["value"] is not None else "N/A"
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
                            [(az_names[k], "N/A" if v is None else f"{v:.3f}") for k, v in az_detail["components"].items()],
                            columns=["Componentă", "Valoare"]), hide_index=True, width='stretch')
                        az_date = f"{az_detail['balance_date']:%d.%m.%Y}" if az_detail["balance_date"] is not None else "N/A"
                        z_txt = f"{az_detail['z']:.2f}" if az_detail["z"] is not None else "N/A"
                        z2_txt = f"{az_detail['z2']:.2f}" if az_detail["z2"] is not None else "N/A"
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
            pio = fund.piotroski(fin.get("income"), fin.get("balance"), fin.get("cashflow"))
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
                st.markdown(f"""
                <div style="background:#161B22; padding:25px; border-radius:15px; border:2px solid {pio_color}; text-align:center;">
                    <p style="color:#8B949E; margin:0; font-size:11px; text-transform:uppercase;">Criterii trecute</p>
                    <h1 style="color:{pio_color}; margin:10px 0; font-size:40px;">{pio_value}</h1>
                    <p style="color:#C9D1D9; font-size:12px; margin:0;">{pio_label}</p>
                </div>
                """, unsafe_allow_html=True)
                nd_ebitda = lev["net_debt_to_ebitda"]
                st.metric("Datorie netă / EBITDA", "N/A" if nd_ebitda is None else f"{nd_ebitda:.2f}x",
                          help="Câți ani de EBITDA ar acoperi datoria netă. Negativ = numerar net. N/A când EBITDA e negativ sau lipsește. "
                               f"Datorie netă {format_amount(lev['net_debt'])}, EBITDA {format_amount(lev['ebitda'])}.")
                int_cov = lev["interest_coverage"]
                st.metric("Acoperirea dobânzii", "N/A" if int_cov is None else f"{int_cov:.1f}x",
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
                               "La bănci, criteriile de lichiditate și marjă brută nu au date.")
                else:
                    st.caption("Piotroski are nevoie de doi ani fiscali de situații financiare.")

            # --- RAPORT FINAL PE CATEGORII ---
            st.markdown("---")
            st.subheader("🕵️‍♂️ Audit Instituțional (6 Piloni)")
            
            # 1. EXECUTĂM CALCULELE (Aceste rânduri îți lipsesc acum!)
            h_score, pros, cons = calculate_health_score_ext(info)
            audit_report = generate_advanced_audit_v2(info, alpha_val, beta_val, h_score)
            
            # 2. PREGĂTIM DATELE PENTRU AFIȘARE (Plasa de siguranță anti-crash)
            display_beta = f"{beta_val:.2f}" if beta_val is not None else "N/A"
            display_alpha = f"{alpha_val*100:.1f}%" if alpha_val is not None else "N/A"

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
                        <h1 style="color:{mos_color}; margin:10px 0; font-size:40px;">{mos_val:.1f}%</h1>
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
                    
                    st.write(f"💵 **Preț Actual:** {current_p:.2f} | 🎯 **Fair Value:** {target_val:.2f}")
                    st.progress(max(0.0, min(mos_val / 100.0, 1.0)))
            else:
                st.warning(f"⚠️ Marja de siguranță indisponibilă: {dcf_res['reason'] or 'lipsește prețul curent.'}")

            # --- MODUL: SUSTENABILITATE ȘI CALITATE (VERSIUNE EXTINSĂ) ---
            st.markdown("---")
            st.subheader("🧬 Sustenabilitate și Calitatea Profitului")
            
            div_verdicts, q_ratio, p_ratio = analyze_dividend_quality(info)
            q_display = f"{q_ratio:.2f}x" if q_ratio is not None else "N/A"
            p_display = f"{p_ratio:.1f}%" if p_ratio is not None else "N/A"
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
                    emp_display = f"{emp:,}" if isinstance(emp, (int, float)) else "N/A"
                    
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
                c_news_ai = get_company_news_rss(real_sym)
                s_score_val = analyze_sentiment_ai(c_news_ai) if c_news_ai else 0
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
                       
            c_news_ai = get_company_news_rss(real_sym)
            cai1, cai2 = st.columns([1, 2])
            
            # --- PARTEA ACTUALIZATĂ (Scorul FinBERT cu Legendă) ---
            with cai1:
                st.write("📊 **Analiză Sentiment AI (Context 24h)**")
                if c_news_ai:
                    with st.spinner("FinBERT procesează narațiunea pieței..."):
                        s_score = analyze_sentiment_ai(c_news_ai)
                        
                        # --- LOGICĂ EXTINSĂ PE 5 PRAGURI ---
                        if s_score >= 0.25:
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
                            <h1 style='color:{c_ai}; margin:0; font-size:48px;'>{s_score:.2f}</h1>
                            <div style='color:{c_ai}; font-weight:bold; font-size:18px; margin-top:5px;'>{status_text}</div>
                            <p style='color:#8B949E; font-size:11px; margin-top:10px;'>Încredere Model: {confidence} ({len(c_news_ai)} surse)</p>
                        </div>
                        
                        <div style='margin-top: 15px; padding: 15px; background: #21262D; border-radius: 10px; border-left: 3px solid #58A6FF;'>
                            <p style='color:#8B949E; font-size: 11px; margin:0 0 8px 0; text-transform: uppercase; font-weight: bold;'>Ghid Intervale (Scală -1 la +1)</p>
                            <div style='color:#C9D1D9; font-size: 13px; line-height: 1.8;'>
                                <div><span style='color:#3FB950;'>■</span> <b>+0.25 la +1.00:</b> Euforie media</div>
                                <div><span style='color:#3FB950; opacity:0.7;'>■</span> <b>+0.05 la +0.24:</b> Optimism moderat</div>
                                <div><span style='color:#8B949E;'>■</span> <b>-0.05 la +0.05:</b> Zgomot neutru</div>
                                <div><span style='color:#F85149; opacity:0.7;'>■</span> <b>-0.24 la -0.06:</b> Îngrijorare / Pesimism</div>
                                <div><span style='color:#F85149;'>■</span> <b>-1.00 la -0.25:</b> Panică extremă</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
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
            
            # ==================================================
            # CALCUL RATING FINAL (CONCLUZIA)
            # ==================================================
            st.markdown("---")
            st.subheader("🎯 Verdict Final: Rating de Investiție")

            # Pregătim variabilele (inițializare explicită — nu mai depindem de locals())
            try:
                t_10y = yf.Ticker("^TNX").fast_info.last_price
                t_3m = yf.Ticker("^IRX").fast_info.last_price
                curr_spread = t_10y - t_3m
            except:
                curr_spread = 0.5

            s_inst = inst_percent  # None = acționariat indisponibil
            s_mos = mos_val
            s_rvol = rvol if 'rvol' in dir() or rvol is not None else 1.0
            
            # Apelăm funcția PRO folosind noul spread 10Y-3M
            final_score, highlights = calculate_investment_rating_pro(info, s_inst, s_rvol, spread, s_mos)
            
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
                        st.plotly_chart(fig_season, width='stretch')
                    
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
                        st.success(f"**🌟 Cea mai bună lună: {best_m['Luna']}**\n\nIstoric, prețul a crescut în **{best_wins} din {best_n} ani** ({best_m['Win Rate (%)']:.0f}%), cu un randament mediu de **{best_m['Randament Mediu (%)']:+.2f}%**.")
                        
                        st.error(f"**🚨 Cea mai slabă lună: {worst_m['Luna']}**\n\nIstoric, prețul a crescut în doar **{worst_wins} din {worst_n} ani** ({worst_m['Win Rate (%)']:.0f}%), cu un randament mediu de **{worst_m['Randament Mediu (%)']:+.2f}%**.")
                        
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
            with st.spinner("Se decodează contractele Market Makerilor..."):
                # Folosim simbolul real identificat anterior
                opt_data, opt_msg = get_options_analysis_ai(real_sym)
            if opt_data:
                # Afișăm ora la care au fost preluate datele din Yahoo
                st.caption(f"⏱️ Date opțiuni actualizate la ora: {opt_data.get('timestamp', 'N/A')}")
                
                # ... restul codului de afișare GEX, IV, etc.    
            if opt_data:
                # FIX DEFINIȚIE PREȚ: Luăm prețul curent din variabila 'hist' deja existentă
                current_market_price = hist['Close'].iloc[-1] if not hist.empty else 0

                # --- 1. CARD GEX (NOUA FUNCȚIONALITATE) ---
                st.markdown(f"""
                    <div style="background:{opt_data['gex_color']}22; padding:25px; border-radius:15px; border-left: 10px solid {opt_data['gex_color']}; margin-bottom:25px; border: 1px solid {opt_data['gex_color']}44;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <p style="margin:0; color:#8B949E; text-transform:uppercase; font-size:11px; letter-spacing:1px; font-weight:bold;">Expunere Netă la Radiații Gamma (Proxy)</p>
                                <h1 style="margin:10px 0; color:{opt_data['gex_color']}; font-size:36px;">{opt_data['net_gex']:,.0f} unități</h1>
                            </div>
                            <div style="text-align:right;">
                                <span style="background:{opt_data['gex_color']}; color:white; padding:8px 16px; border-radius:25px; font-weight:bold; font-size:13px;">
                                    {opt_data['gex_verdict'].split(':')[0]}
                                </span>
                            </div>
                        </div>
                        <p style="margin:10px 0 0 0; font-size:15px; color:#C9D1D9;">{opt_data['gex_verdict'].split(': ')[1]}</p>
                    </div>
                """, unsafe_allow_html=True)

                # --- RÂNDUL 1: VITEZOMETRU ȘI METRICE ---
                col_gau, col_met = st.columns([1.5, 2])

                with col_gau:
                    # Preluăm culoarea din opt_data calculată mai sus
                    iv_display_color = opt_data.get('iv_color', '#FFFFFF')
                    
                    fig_iv = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = opt_data['iv'],
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Termometru IV", 'font': {'size': 18, 'color': '#8B949E'}},
                        # --- AICI COLORĂM NUMĂRUL (Procentul) ---
                        number = {
                            'suffix': "%", 
                            'font': {'color': iv_display_color, 'size': 50} 
                        },
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                            'bar': {'color': iv_display_color}, # Bara urmărește culoarea textului
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
                    st.plotly_chart(fig_iv, width='stretch')

                with col_met:
                    m1, m2 = st.columns(2)
                    m1.metric("Put/Call Ratio (OI)", f"{opt_data['oi_pc_ratio']:.2f}",
                              help="Open Interest (Banii blocați): Raportul total al pariurilor pe scădere (Puts) vs creștere (Calls). Sub 0.7 = Optimism (Bullish); Peste 1.0 = Pesimism/Frică (Bearish).")
                    m2.metric("Put/Call Ratio (Volum)", f"{opt_data['vol_pc_ratio']:.2f}",
                              help="Volum (Banii de azi): Indică panica sau entuziasmul intraday. Se mișcă mai rapid decât OI. Valori mari bruste indică vânzări de panică.")
                    
                    m3, m4 = st.columns(2)
                    mp = opt_data['max_pain']
                    mp_display = f"${mp:.1f}" if mp is not None else "N/A"
                    if mp is not None and current_market_price > 0:
                        diff_mp = ((mp / current_market_price) - 1) * 100
                        m3.metric("Preț Max Pain", mp_display, f"{diff_mp:.1f}% față de preț",
                                  help="Strike-ul la care opțiunile acestei expirări ar valora cel mai puțin, calculat din open interest. Este un reper, nu o țintă de preț.")
                    else:
                        m3.metric("Preț Max Pain", mp_display)
                    m4.metric("Data Expirării", opt_data['expiration'])
                # --- INTEGRARE IV RANK & PERCENTILE ---
                with col_met:
                    # Obținem datele din motorul AI
                    from ai_engine import calculate_iv_rank_percentile
                    iv_rank, iv_pct = calculate_iv_rank_percentile(real_sym, opt_data['iv'])
                    
                    st.write("") # Spațiu vizual
                    c_rank1, c_rank2 = st.columns(2)
                    
                    # Interpretare culori
                    rank_col = "#F85149" if iv_rank > 70 else ("#3FB950" if iv_rank < 30 else "#D29922")
                    
                    c_rank1.metric("IV Rank", f"{iv_rank:.1f}%", 
                                help="Unde se află IV-ul de azi între minimul și maximul din ultimul an.")
                    c_rank2.metric("IV Percentile", f"{iv_pct:.1f}%",
                                help="Câte zile din ultimul an au avut un IV mai mic decât cel de azi.")

                # --- VERDICT FINAL OPȚIUNI (ACTUALIZAT) ---
                if iv_rank > 80:
                    st.warning(f"💎 **STRATEGIE SHORT VOL:** Opțiunile sunt la extreme istorice. Este un moment excelent pentru a VINDE volatilitate (ex: Covered Calls), nu pentru a cumpăra.")
                elif iv_rank < 20:
                    st.success(f"🛒 **STRATEGIE LONG VOL:** Opțiunile sunt neobișnuit de ieftine față de istoric. Moment ideal pentru achiziție de protecție (Puts) sau speculă (Calls).")    

                # --- RÂNDUL 2: VERDICTUL INTELIGENT (RESTABILIT) ---
                oi_pc = opt_data['oi_pc_ratio']
                vol_pc = opt_data['vol_pc_ratio']
                iv_val = opt_data['iv']

                if oi_pc < 0.7 and vol_pc < 0.7:
                    v_text, v_col = "🚀 **BULLISH CONVINCED:** Instituțiile cumpără masiv. Trend ascendent solid.", "#3FB950"
                elif oi_pc < 0.7 and vol_pc > 1.1:
                    v_text, v_col = "🔄 **DIVERGENȚĂ:** Optimism pe termen lung, dar azi apare frică (Puts). Posibilă corecție!", "#D29922"
                elif oi_pc > 1.1:
                    v_text, v_col = "🐻 **BEARISH DOMINANT:** Piața pariază pe scădere. Opțiunile Put domină peisajul.", "#F85149"
                else:
                    v_text, v_col = "⚖️ **NEUTRU:** Echilibru între cumpărători și vânzători.", "#8B949E"

                st.markdown(f"<div style='background:{v_col}22; padding:20px; border-radius:12px; border-left: 5px solid {v_col}; margin-bottom:20px;'>{v_text}</div>", unsafe_allow_html=True)

                # --- RÂNDUL 3: STRATEGIA IV (RESTABILIT) ---
                if iv_val > 45:
                    st.warning(f"⚠️ **ALERTA IV:** Deși direcția pare {v_text.split(':')[0][2:]}, opțiunile sunt **prea scumpe** ({iv_val:.1f}%). Riscul de scădere a valorii prin volatilitate este uriaș.")
                elif iv_val < 20:
                    st.success(f"💎 **OPORTUNITATE IV:** Opțiunile sunt foarte ieftine ({iv_val:.1f}%). Moment ideal pentru a paria pe direcția identificată.")

                # --- RÂNDUL 4: DETALIEREA INDICATORILOR (RESTABILIT) ---
                st.markdown("#### 🔍 Detalierea Indicatorilor")
                interpretare_data = [
                    {"Indicator": "📉 Put/Call (OI)", "Valoare": f"{oi_pc:.2f}", "Interpretare": "Bullish" if oi_pc < 0.7 else "Bearish" if oi_pc > 1.1 else "Neutru"},
                    {"Indicator": "⚡ Put/Call (Volum)", "Valoare": f"{vol_pc:.2f}", "Interpretare": "Sentiment Bullish" if vol_pc < 0.7 else "Panică" if vol_pc > 1.1 else "Normal"},
                    {"Indicator": "🧲 Max Pain Price", "Valoare": mp_display, "Interpretare": f"Reper la expirarea din {opt_data['expiration']}, nu țintă de preț"},
                    {"Indicator": "🌡️ Volatilitate (IV)", "Valoare": f"{iv_val:.1f}%", "Interpretare": "EVITĂ derivate (Scump)" if iv_val > 40 else "OK de cumpărat (Ieftin)"}
                ]
                st.table(interpretare_data)
            else:
                st.info(opt_msg)
                        
            # =================================================================
            # 👑 VERDICT FINAL MASTER AI: DECIZIA DE INVESTIȚIE (RADIOGRAFIE)
            # =================================================================
            st.markdown("---")
            st.subheader("👑 Decizie Master AI")
            
            # --- PROTECȚIE VARIABILE (inițializare explicită — fără locals()) ---
            s_inst = inst_percent  # None = acționariat indisponibil
            s_mos = mos_val
            s_rvol = rvol if 'rvol' in dir() else 1.0
            s_score_final = s_score_val if 's_score_val' in dir() else 0
            opt_final = opt_data if 'opt_data' in dir() else None
            
            # Extragem datele specifice modulelor adiacente
            z_score_val = z_val_swot if 'z_val_swot' in dir() else None
            cash_ratio = q_ratio if 'q_ratio' in dir() else 1.0
            ai_regime = regime_msg if 'regime_msg' in dir() else "Neutru"

            # Extragem Spread-ul Macro la cald
            try:
                t_10y = yf.Ticker("^TNX").fast_info.last_price
                t_3m = yf.Ticker("^IRX").fast_info.last_price
                curr_spread = t_10y - t_3m
            except: curr_spread = 0.5

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
                pe_ratio = info.get('trailingPE', 0) or 0
                pb_ratio = info.get('priceToBook', 0) or 0
                margins = (info.get('profitMargins', 0) or 0) * 100
                beta = beta_val or 1
                
                # --- BULLET 1: Profitabilitate & Venituri ---
                if margins > 0:
                    if margins > 15:
                        text_fin = f"Compania raportează rezultate fundamentale puternice, cu o marjă netă de profit de {margins:.2f}%. Această eficiență operațională ridicată consolidează încrederea investitorilor pe termen lung."
                        bullets.append({"icon": "↗️", "color": "#3FB950", "text": text_fin})
                    else:
                        text_fin = f"Compania menține o profitabilitate moderată, operând cu o marjă netă de {margins:.2f}%, o valoare standard pentru dinamica actuală a sectorului său."
                        bullets.append({"icon": "➡️", "color": "#8B949E", "text": text_fin})
                elif margins < 0:
                    text_fin = f"Rezultatele recente indică presiuni operaționale, compania raportând pierderi cu o marjă netă negativă de {margins:.2f}%, un factor de risc major pentru acționari."
                    bullets.append({"icon": "↘️", "color": "#F85149", "text": text_fin})

                # --- BULLET 2: Sentimentul Știrilor (Context) ---
                c_news_ai = get_company_news_rss(real_sym)
                if c_news_ai:
                    top_news = c_news_ai[0]['title']
                    from ai_engine import analyze_sentiment_ai
                    s_score = analyze_sentiment_ai(c_news_ai)
                    
                    if s_score > 0.15:
                        text_news = f"Sentimentul media general este optimist. Atenția presei este captată de evoluții pozitive, un catalizator recent fiind: «{top_news}»."
                        bullets.append({"icon": "↗️", "color": "#3FB950", "text": text_news})
                    elif s_score < -0.15:
                        text_news = f"Narațiunea mediatică este dominată de îngrijorări sau reglementări stricte. Titlurile recente reflectă pesimism, precum: «{top_news}»."
                        bullets.append({"icon": "↘️", "color": "#F85149", "text": text_news})
                    else:
                        text_news = f"Acoperirea în presă este neutră, fără șocuri reputaționale majore în ultimele 24 de ore. Subiectul momentului: «{top_news}»."
                        bullets.append({"icon": "➡️", "color": "#8B949E", "text": text_news})

                # --- BULLET 3: Volatilitate / Risc (Analiză Beta) ---
                if beta > 1.3:
                    text_vol = f"Prețul acțiunilor a fost volatil comparativ cu piața de referință (Beta de {beta:.2f}). Această fluctuație amplă atrage speculatorii, dar poate îngrijora investitorii conservatori."
                    bullets.append({"icon": "↘️", "color": "#F85149", "text": text_vol})
                elif beta < 0.8:
                    text_vol = f"Acțiunea prezintă o volatilitate redusă față de piața generală (Beta de {beta:.2f}), comportându-se ca un activ defensiv în perioadele de incertitudine."
                    bullets.append({"icon": "↗️", "color": "#3FB950", "text": text_vol})

                # --- BULLET 4: Evaluare Multipli ---
                if pe_ratio > 30:
                    text_val = f"Multipli de evaluare ridicați, inclusiv un raport P/E de {pe_ratio:.2f} și P/B de {pb_ratio:.2f}, sugerează că acțiunea ar putea fi supraevaluată în raport cu fundamentele contabile curente."
                    bullets.append({"icon": "↘️", "color": "#F85149", "text": text_val})
                elif 0 < pe_ratio < 15:
                    text_val = f"Acțiunea se tranzacționează la multipli de evaluare atractivi (P/E de {pe_ratio:.2f}), indicând o potențială subevaluare din partea pieței raportat la capacitatea sa de generare a profitului."
                    bullets.append({"icon": "↗️", "color": "#3FB950", "text": text_val})

                # --- RANDARE UI PREMIUM ---
                if bullets:
                    st.markdown("""
                    <style>
                    .xtb-bullet-container {
                        background-color: #161B22; 
                        padding: 25px; 
                        border-radius: 12px; 
                        border: 1px solid #30363D;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    }
                    .xtb-row {
                        display: flex; 
                        align-items: flex-start; 
                        margin-bottom: 20px;
                    }
                    .xtb-icon {
                        margin-right: 15px; 
                        font-size: 20px; 
                        margin-top: -2px;
                    }
                    .xtb-text {
                        font-size: 15px; 
                        color: #C9D1D9; 
                        line-height: 1.5;
                        font-weight: 400;
                    }
                    .xtb-footer {
                        margin-top: 10px; 
                        font-size: 12px; 
                        color: #8B949E; 
                        border-top: 1px solid #30363D; 
                        padding-top: 15px; 
                        display: flex; 
                        justify-content: space-between;
                        align-items: center;
                    }
                    </style>
                    <div class="xtb-bullet-container">
                    """, unsafe_allow_html=True)
                    
                    for b in bullets:
                        st.markdown(f"""
                        <div class="xtb-row">
                            <div class="xtb-icon" style="color: {b['color']};">{b['icon']}</div>
                            <div class="xtb-text">{b['text']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    st.markdown("""
                        <div class="xtb-footer">
                            <span>Informațiile de mai sus au fost generate algoritmic. Nu le trata ca recomandare sau sfat de investiții.</span>
                            <span style="color:#A371F7; font-weight:bold;">⚡ Quant-AI Engine</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
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
        
        with st.expander("➕ Adaugă Tranzacție Nouă"):
            with st.form("add_pf"):
                c1, c2, c3, c4 = st.columns(4)
                s = c1.text_input("Simbol (ex: AAPL, EUNL.DE, TLV.RO)").upper()
                q = c2.number_input("Cantitate", min_value=0.01, value=1.0, format="%.4f")
                p = c3.number_input("Preț Achiziție", min_value=0.01, value=100.0, format="%.2f")
                curr = c4.selectbox("Moneda", ["USD", "EUR", "RON"]) 
                
                d_acq = st.date_input("Data", now_ro().date())
                
                if st.form_submit_button("Salvează") and s:
                    add_trade(s, q, p, d_acq, curr)
                    st.success(f"Adăugat {s} în Google Sheets!")
                    st.rerun()

        # Încărcăm datele din Google Sheets
        df_pf = load_portfolio()

        if df_pf.empty:
            st.info("Portofoliul este gol sau nu s-a putut conecta la Google Sheets.")
        else:
            st.markdown("### Perioadă Analiză")
            hist_range = st.select_slider("", options=["1Z", "1S", "1L", "3L", "6L", "1A", "3A", "5A"], value="1A", key="range_slider")
            
            tab_usd, tab_eur, tab_ron = st.tabs(["🇺🇸 Portofoliu USD", "🇪🇺 Portofoliu EUR", "🇷🇴 Portofoliu BVB (RON)"])
            
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
                c_kpi1.metric(f"Total Investit ({currency_symbol})", f"{total_invested:,.2f} {currency_symbol}")
                c_kpi2.metric(f"Valoare Curentă ({currency_symbol})", f"{total_current:,.2f} {currency_symbol}")
                c_kpi3.metric(f"Profit/Pierdere ({currency_symbol})", f"{total_profit_val:,.2f} {currency_symbol}", f"{total_profit_pct:.2f}%")

                if pf_notes.get('curve_start') is not None:
                    st.caption(
                        f"ℹ️ Graficele și indicatorii de risc de mai jos aplică deținerile de azi pe trecut, începând cu "
                        f"{pf_notes['curve_start'].strftime('%d.%m.%Y')} (prima zi în care toate pozițiile au preț; "
                        f"istoricul cel mai scurt: {pf_notes.get('limiting_symbol')}). Nu reprezintă performanța realizată efectiv."
                    )
                if pf_notes.get('no_history'):
                    st.caption(f"ℹ️ Fără istoric de preț, deci neincluse în grafice: {', '.join(pf_notes['no_history'])}.")

                # --- INTEGRARE MONTE CARLO ---
                st.markdown("---")
                st.subheader("🔮 Proiecție Probabilistă Monte Carlo (1 An)")

                with st.spinner("Se simulează 1000 de scenarii de piață..."):
                    paths, last_val = run_monte_carlo_sim(hist_curve)

                if paths is not None:
                    # Calculăm percentilele de risc
                    # Ce se întâmplă în cel mai rău caz (5%) și cel mai bun (95%)
                    p5 = np.percentile(paths[-1, :], 5)
                    p50 = np.percentile(paths[-1, :], 50)
                    p95 = np.percentile(paths[-1, :], 95)
                    
                    fig_mc = go.Figure()
                    
                    # Afișăm primele 100 de traiectorii (pentru a nu supraîncărca browser-ul)
                    x_range = list(range(252))
                    for i in range(100):
                        fig_mc.add_trace(go.Scatter(x=x_range, y=paths[:, i], mode='lines', 
                                                line=dict(width=0.5, color='rgba(3, 247, 6, 0.45)'),
                                                showlegend=False))
                    
                    # Adăugăm liniile critice
                    fig_mc.add_trace(go.Scatter(x=x_range, y=[p50]*252, line=dict(color='white', dash='dash'), name='Așteptare Medie'))
                    fig_mc.add_trace(go.Scatter(x=x_range, y=[p5]*252, line=dict(color='#F85149', width=2), name='Worst Case (5%)'))
                    
                    fig_mc.update_layout(height=400, template="plotly_dark", 
                                        hovermode="x", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                        yaxis_title=f"Valoare Portofoliu ({currency_symbol})")
                    
                    st.plotly_chart(fig_mc, width='stretch')
                    
                    # Dashboard de Risc sub grafic
                    c_mc1, c_mc2, c_mc3 = st.columns(3)
                    
                    # Calculăm Value at Risk Proiectat
                    expected_loss = ((last_val - p5) / last_val) * 100
                    
                    c_mc1.metric("Valoare în cel mai nefavorabil caz", f"{p5:,.2f} {currency_symbol}",
                                help="Value at Risk (VaR): Reprezintă pragul sub care există doar 5% șanse ca portofoliul tău să scadă în următorul an, conform celor 1000 de scenarii simulate.")

                    c_mc2.metric("Pierderea maximă așteptată", f"-{expected_loss:.2f}%", delta_color="inverse",
                                help="Procentul maxim de scădere din valoarea curentă în scenariul pesimist (cel de la linia roșie din grafic).")

                    c_mc3.metric("Probabilitatea de profit", f"{(np.sum(paths[-1, :] > last_val) / 1000 * 100):.1f}%",
                                help="Câte scenarii din cele 1000 simulate au terminat pe profit (peste valoarea actuală). O valoare peste 50% indică un avantaj statistic pe termen lung.")
                    
                    st.info(f"💡 **Analiza Quant:** Există o probabilitate de 95% ca peste un an, portofoliul tău să nu scadă sub valoarea de **{p5:,.2f} {currency_symbol}**. Acest model ia în calcul volatilitatea ta istorică.")
                
                if not hist_curve.empty:
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Scatter(
                        x=hist_curve.index, y=hist_curve.values, 
                        fill='tozeroy', line=dict(color='#238636'), name=f'Valoare {currency_symbol}'
                    ))
                    fig_hist.update_layout(height=350, template="plotly_dark", margin=dict(t=10, b=10), paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_hist, width='stretch')
                    # --- NOU: COMPARAȚIE BENCHMARK ---
                st.markdown("---")
                st.subheader("🏁 Performanță Relativă (Benchmark)")
                
                # Definirea benchmark-ului înainte de apel (Fix Alpha EUR)
                if currency_symbol == "$":
                    current_bench_ticker = "SPY"
                    current_bench_name = "S&P 500 (SPY)"
                elif currency_symbol == "RON":
                    # Schimbăm STOXX 600 cu indicele local BET
                    current_bench_ticker = "TVBETETF.RO"
                    current_bench_name = "BET (ETF TVBETETF)"
                else:
                    # Rămâne STOXX 600 doar pentru portofoliul în EURO
                    current_bench_ticker = "EXSA.DE"
                    current_bench_name = "STOXX Europe 600 (EXSA.DE)"

                with st.spinner(f"Se compară cu {current_bench_name}..."):
                    # Trimitem ticker-ul și numele corect către funcție
                    render_benchmark_comparison(hist_curve, current_bench_ticker, current_bench_name)
                
                # --- CALCUL METRICI (Rândul 1 & 2) ---
                # 1. Calculăm metricile de risc intern
                max_dd, sharpe, var_abs, vol_ann = calculate_risk_metrics(hist_curve)
                
                # 2. Alegem benchmark-ul automat bazat pe monedă
                if currency_symbol == "$":
                    bench_ticker = "SPY"
                    bench_name = "S&P 500 (SPY)"
                elif currency_symbol == "RON":
                    # Schimbarea crucială pentru piața din România
                    bench_ticker = "TVBETETF.RO"
                    bench_name = "BET (ETF TVBETETF)"
                else:
                    # Rămâne STOXX 600 doar pentru portofoliul în EUR
                    bench_ticker = "EXSA.DE"
                    bench_name = "STOXX Europe 600 (EXSA.DE)"
                
                # 3. Calculăm Corelația Globală și Beta
                global_corr, portfolio_beta = calculate_portfolio_beta(hist_curve, bench_ticker)

                st.markdown(f"#### 🛡️ Diagnostic Portofoliu ({currency_symbol})")
                
                # Rândul 1: Riscul Intern al activelor tale
                c_r1, c_r2, c_r3, c_r4 = st.columns(4)
                c_r1.metric("Max Drawdown", f"{max_dd*100:.2f}%", help="Cea mai mare scădere istorică.")
                c_r2.metric("Sharpe Ratio", f"{sharpe:.2f}", help="Eficiența profitului vs risc (Ideal > 1).")
                c_r3.metric("VaR (95%)", f"{currency_symbol} {abs(var_abs):,.2f}", help="Pierderea maximă probabilă într-o singură zi.")
                c_r4.metric("Volatilitate Anualizată", f"{vol_ann*100:.2f}%", help="Agitația generală a prețurilor portofoliului.")

                # Rândul 2: Relația Strategică cu Piața (Benchmark-ul)
                st.write("") # Mic spațiu vizual între rânduri
                c_b1, c_b2, c_b3, c_b4 = st.columns(4)
                
                c_b1.metric("Benchmark", bench_name)
                
                c_b2.metric("Corelație cu Piața", f"{global_corr:.2f}", 
                           help=f"Scorul de {global_corr:.2f} arată cât de mult imiți indexul {bench_name}. 1.00 = Copie fidelă.")
                
                c_b3.metric("Beta Portofoliu", f"{portfolio_beta:.2f}", 
                           help=f"Sensibilitatea la piață. Un Beta de {portfolio_beta:.2f} înseamnă că ești {'mai agresiv' if portfolio_beta > 1 else 'mai stabil'} decât media.")

                # Scorul de Diversificare Strategică
                div_score = (1 - abs(global_corr)) * 100
                c_b4.metric("Scor Diversificare", f"{div_score:.1f}%", 
                           help="Indică cât de independentă este strategia ta față de restul pieței.")
                
                sortino_val = calculate_sortino_ratio(hist_curve)
                # Creăm un rând nou de metrici sau adăugăm la cel existent
                st.write("") 
                c_s1, c_s2 = st.columns(2)

                with c_s1:
                    s_color = "#3FB950" if sortino_val > 2 else ("#D29922" if sortino_val > 1 else "#F85149")
                    st.metric("Sortino Ratio (Calitate Profit)", f"{sortino_val:.2f}", 
                            help="Ideal > 2. Arată cât de mult câștigi pentru fiecare unitate de risc la scădere.")

                with c_s2:
                    if sortino_val > 2:
                        st.success("🌟 **Excelent:** Portofoliul tău produce profituri cu un risc minim de scădere majoră.")
                    elif sortino_val > 1:
                        st.info("⚖️ **Acceptabil:** Profitul justifică riscul asumat.")
                    else:
                        st.warning("⚠️ **Risc Ridicat:** Obții profit, dar ești expus la scăderi violente. Verifică diversificarea.")

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
                        st.plotly_chart(fig_sym, width='stretch', key=f"pie_sym_{currency_symbol}")

                # 4. Detaliu Poziții
                st.subheader("Detaliu Poziții")
                if not df_calc.empty:
                    display_cols = ['Symbol', 'Quantity', 'AvgPrice', 'CurrentPrice', 'MarketValue', 'Profit', 'Profit %']
                    
                    def color_profit(val):
                        if pd.isna(val): return ''
                        color = '#3FB950' if val >= 0 else '#F85149'
                        return f'color: {color}'

                    st.dataframe(
                        df_calc[display_cols].style.map(color_profit, subset=['Profit', 'Profit %'])
                        .format({
                            # până la 4 zecimale, fără zerouri inutile (0.31 nu mai apare ca 0.3)
                            'Quantity': lambda q: f"{q:,.4f}".rstrip('0').rstrip('.'),
                            'AvgPrice': '{:.4f}', 'CurrentPrice': '{:.4f}',
                            'MarketValue': '{:,.2f}', 'Profit': '{:,.2f}', 'Profit %': '{:.2f}%'
                        }, na_rep="N/A"),
                        width='stretch'
                    )        

                with col_pie2:
                    st.caption("**După Sector Economic (%)**")
                    with st.spinner("Analizăm expunerea..."):
                        df_sectors = get_portfolio_sectors(df_calc)
                    
                    if not df_sectors.empty:
                        # Grafic Plăcintă cu Procente
                        fig_sec = go.Figure(data=[go.Pie(
                            labels=df_sectors['Sector'], 
                            values=df_sectors['Pondere %'], 
                            hole=.4,
                            textinfo='label+percent',
                            marker=dict(colors=['#1f6feb', '#238636', '#da3633', '#d29922'])
                        )])
                        fig_sec.update_layout(height=350, margin=dict(t=0, b=0, l=0, r=0), 
                                              template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_sec, width='stretch', key=f"pie_sec_{currency_symbol}")

                        # VERDICT DIVERSIFICARE
                        # 'Nedefinit' = Yahoo nu a trimis sectorul (sau ETF): nu e un sector real și nu se evaluează.
                        known = df_sectors[df_sectors['Sector'] != 'Nedefinit']
                        undefined_pct = df_sectors.loc[df_sectors['Sector'] == 'Nedefinit', 'Pondere %'].sum()
                        if known.empty:
                            st.info("ℹ️ Sectoarele nu sunt disponibile de la Yahoo pentru aceste poziții, deci concentrarea pe sectoare nu se poate evalua.")
                        else:
                            max_sector = known.iloc[0]
                            if max_sector['Pondere %'] > 40:
                                st.warning(f"⚠️ **Concentrare mare:** Sectorul '{max_sector['Sector']}' ocupă {max_sector['Pondere %']:.1f}% din portofoliu. Riști mult dacă acest sector scade.")
                            else:
                                st.success(f"✅ **Diversificare bună:** Niciun sector cunoscut nu depășește 40%.")
                            if undefined_pct > 0:
                                st.caption(f"ℹ️ {undefined_pct:.1f}% din portofoliu are sector necunoscut și nu intră în această evaluare.")
                    else:
                        st.info("Nu există date sectoriale.")

                # 3. Matrice Corelare
                st.markdown("---")
                st.subheader("🧩 Analiză Diversificare (Corelare)")
                current_tickers = df_subset['Symbol'].unique().tolist()
                if len(current_tickers) > 1:
                    with st.spinner("Analizăm suprapunerea riscului..."):
                        # Acum doar apelăm funcția, ea face totul
                        plot_correlation_matrix(current_tickers)
                else:
                    st.info("Adaugă cel puțin 2 active pentru analiză.")
                
                st.markdown("<br>", unsafe_allow_html=True)
                # =======================================================
                # MODUL NOU: OPTIMIZARE PORTOFOLIU AI (MARKOWITZ)
                # =======================================================
                st.markdown("---")
                st.subheader("🧠 Optimizator Portofoliu AI (Markowitz)")
                st.markdown("Inteligența artificială analizează corelațiile și volatilitatea istorică pentru a găsi alocarea matematic perfectă (risc minim, profit maxim).")

                if len(current_tickers) >= 2:
                    with st.spinner("Motorul Quant calculează Frontiera Eficientă..."):
                        # 1. Descărcăm prețurile de închidere curate pentru acțiunile tale
                        # close_frame elimină simbolurile fără nicio cotație (altfel optimizarea eșua cu un mesaj criptic)
                        hist_opt = close_frame(yf.download(current_tickers, period="1y", progress=False), current_tickers)
                        skipped_opt = [t for t in current_tickers if t not in hist_opt.columns]
                        if skipped_opt:
                            st.caption(f"ℹ️ Fără istoric de preț, excluse din optimizare: {', '.join(skipped_opt)}.")
                        
                        # 2. Trimitem datele la creierul AI
                        from ai_engine import optimize_portfolio_ai
                        if hist_opt.shape[1] >= 2:
                            opt_res, opt_msg = optimize_portfolio_ai(hist_opt)
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
                                    "Pondere Optimă AI (%)": opt_res['allocation'].get(sym, 0)
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
                                name='Sugestia AI', 
                                marker_color='#3FB950',
                                texttemplate='%{y:.1f}%',      # Scrie procentul pe bară
                                textposition='auto',
                                hovertemplate="<b>%{x}</b> (Optim AI): %{y:.2f}%<extra></extra>" # Formatare hover
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
                            st.plotly_chart(fig_opt, width='stretch', key=f"opt_bar_{currency_symbol}")

                            # 4. Afișăm Metricele Portofoliului Ideal cu explicații
                            st.markdown("##### 🏆 Cum ar arăta Portofoliul Ideal (Conform AI):")
                            c_op1, c_opt2, c_opt3 = st.columns(3)
                            
                            c_op1.metric("Randament Anual Așteptat", f"{opt_res['expected_return']:.2f}%",
                                        help="Randamentul anual pe care modelul matematic îl estimează pentru această alocare, bazat pe performanța istorică a activelor tale.")
                            
                            c_opt2.metric("Volatilitate (Risc)", f"{opt_res['expected_volatility']:.2f}%",
                                        help="Măsură a fluctuațiilor prețului. AI-ul încearcă să găsească ponderile care oferă cel mai mic 'stres' (swing-uri de preț) pentru randamentul țintit.")
                            
                            c_opt3.metric("Sharpe Ratio Optim", f"{opt_res['sharpe_ratio']:.2f}",
                                        help="Indicatorul de aur al eficienței: Profitul raportat la Risc. O valoare peste 1.0 este considerată bună, iar peste 2.0 este excelentă, indicând un profit mare cu riscuri bine controlate.")

                            st.info("💡 **Strategie Quant:** Barele verzi îți arată unde ar trebui să muți banii. Algoritmul îți sugerează să crești expunerea pe activele cu randament stabil și să o reduci pe cele care aduc doar 'zgomot' (volatilitate inutilă).")
                        else:
                            st.warning(opt_msg)
                else:
                    st.info("Adaugă cel puțin 2 acțiuni în portofoliu pentru ca AI-ul să poată calcula diversificarea optimă.") 

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
            
            try: vix_val = yf.Ticker("^VIX").fast_info.last_price
            except: vix_val = 20.0
            
            try:
                t_10y = yf.Ticker("^TNX").fast_info.last_price
                t_3m = yf.Ticker("^IRX").fast_info.last_price
                curr_yield_spread = t_10y - t_3m
            except: curr_yield_spread = 0.5

            news_samples = get_company_news_rss("^GSPC") + get_company_news_rss("^IXIC")
            from ai_engine import analyze_sentiment_ai
            macro_sentiment = analyze_sentiment_ai(news_samples) if news_samples else 0

            # --- 2. CALCULĂM SCORUL GLOBAL ---
            from ai_engine import calculate_master_macro_verdict
            m_score, m_label, m_col, m_desc, m_reasons = calculate_master_macro_verdict(
                macro_sectors, macro_risk_ratio, macro_corr, macro_sentiment, curr_yield_spread, vix_val
            )

        # --- 3. AFIȘARE BANNER SUPREM ---
        st.markdown(f"""
            <div style="background:linear-gradient(90deg, #161B22 0%, #21262D 100%); padding:30px; border-radius:15px; border-left: 10px solid {m_col}; margin-bottom:20px; box-shadow: 0 4px 15px rgba(0,0,0,0.5);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4 style="color:#8B949E; margin:0; text-transform:uppercase; letter-spacing:1px;">Verdict Sănătate Piață Globală</h4>
                        <h1 style="color:{m_col}; margin:10px 0; font-size:38px;">{m_label}</h1>
                        <p style="color:#C9D1D9; font-size:16px;">{m_desc}</p>
                    </div>
                    <div style="text-align:center; min-width: 120px;">
                        <div style="font-size:12px; color:#8B949E;">SCOR MACRO AI</div>
                        <div style="font-size:56px; font-weight:bold; color:{m_col};">{int(m_score)}</div>
                        <div style="font-size:14px; color:#8B949E;">/ 100</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # --- 4. RESTAURAREA ARGUMENTELOR VECHI ---
        st.markdown("#### 🔍 Argumentele Modelului:")
        c_re1, c_re2 = st.columns(2)
        for i, reason in enumerate(m_reasons):
            if i % 2 == 0: c_re1.markdown(f"{reason}")
            else: c_re2.markdown(f"{reason}")

        # --- 5. NOUL CARD DEEP-DIVE AI (CLONA XTB) ---
        st.markdown("---")
        st.subheader("✨ Sinteza Inteligenței Artificiale")
        
        with st.spinner("Sintetizăm intersecțiile de metale, valute, dobânzi și opțiuni..."):
            from ai_engine import generate_macro_ai_summary
            macro_bullets = generate_macro_ai_summary(
                vix_val, curr_yield_spread, macro_risk_ratio, macro_sectors, macro_corr, macro_data, df_fred
            )
            
            if macro_bullets:
                st.markdown("""
                <style>
                .xtb-bullet-container {
                    background-color: #161B22; 
                    padding: 25px; 
                    border-radius: 12px; 
                    border: 1px solid #30363D;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }
                .xtb-row {
                    display: flex; 
                    align-items: flex-start; 
                    margin-bottom: 20px;
                }
                .xtb-icon {
                    margin-right: 15px; 
                    font-size: 20px; 
                    margin-top: -2px;
                }
                .xtb-text {
                    font-size: 15px; 
                    color: #C9D1D9; 
                    line-height: 1.5;
                    font-weight: 400;
                }
                .xtb-footer {
                    margin-top: 10px; 
                    font-size: 12px; 
                    color: #8B949E; 
                    border-top: 1px solid #30363D; 
                    padding-top: 15px; 
                    display: flex; 
                    justify-content: space-between;
                    align-items: center;
                }
                </style>
                <div class="xtb-bullet-container">
                """, unsafe_allow_html=True)
                
                for b in macro_bullets:
                    st.markdown(f"""
                    <div class="xtb-row">
                        <div class="xtb-icon" style="color: {b['color']};">{b['icon']}</div>
                        <div class="xtb-text">{b['text']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown("""
                    <div class="xtb-footer">
                        <span>Informații derivate algoritmic din confluența dobânzilor, prețurilor de mărfuri (Aur, Petrol, Cupru), sentimentului și pieței de credit.</span>
                        <span style="color:#A371F7; font-weight:bold;">⚡ Quant-AI Engine</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Așteptăm date suficiente pentru generarea sintezei.")

        st.markdown("---")
        
        if st.button("🔄 Reîmprospătează Piața"):
            get_global_market_data.clear()
            get_macro_data_visuals.clear()
            st.rerun()

        # Urmează restul codului tău cu "🚨 Early Warning System..."

        st.subheader("🚨 Early Warning System: Risc Recesiune (10Y-3M)")
        
        try:
            # Preluăm datele pentru 10 ani și 3 luni (standardul FED)
            t_10y = yf.Ticker("^TNX").fast_info.last_price
            t_3m = yf.Ticker("^IRX").fast_info.last_price 
            
            if t_10y > 0 and t_3m > 0:
                spread = t_10y - t_3m
                label_spread = "SPREAD 10Y - 3M (SĂNĂTATE MACRO)"
            else:
                spread = 0.5
                label_spread = "Spread indisponibil (Data Delay)"

            y_col1, y_col2 = st.columns([1, 2])
            with y_col1:
                spread_color = "#F85149" if spread < 0 else "#3FB950"
                st.markdown(f"""
                    <div style="background:#161B22; padding:20px; border-radius:15px; border:2px solid {spread_color}; text-align:center;">
                        <p style="color:#8B949E; margin:0; font-size:11px; text-transform:uppercase;">{label_spread}</p>
                        <h1 style="color:{spread_color}; margin:10px 0;">{spread:.3f}</h1>
                    </div>
                """, unsafe_allow_html=True)
            
            with y_col2:
                if spread < 0:
                    st.error(f"⚠️ **INVERSIUNE CONFIRMATĂ:** Spread-ul de {spread:.3f} indică o curbă a randamentelor inversată. Istoric, acest semnal a precedat fiecare recesiune majoră..")
                    st.write("👉 **Strategie:** Protejează capitalul. Redu expunerea pe acțiuni ciclice și crește ponderea în Cash/Aur.")
                else:
                    st.success(f"✅ **SĂNĂTATE MACRO:** Spread-ul de {spread:.3f} indică o curbă a randamentelor normală. Indică expansiune economică")
                    st.write("👉 **Strategie:** Poți menține o strategie de creștere (Growth). Mediul economic susține expansiunea bursieră.")

        except Exception as e:
            st.info("Sistemul de monitorizare a curbei randamentelor se recalibrează...")

        # --- AICI CONTINUĂ RESTUL CODULUI TĂU (HARTA, RADAR, MATRICE, TABELE) ---
        st.markdown("---")    

        # --- PASUL 1: DESCĂRCARE DATE (Trebuie să fie PRIMUL rând!) ---
        # Acum variabila macro_data este creată și poate fi folosită mai jos
        macro_tickers, macro_data = get_macro_data_visuals()

        # --- PASUL 2: MODUL INTERPRETARE DINAMICĂ DOBÂNZI ---
        st.markdown("### 🧭 Indicatori Macroeconomici")
        
        # Verificăm dacă avem date pentru randamentele pe 10 ani
        has_tnx = '^TNX' in macro_data.columns.levels[0] if isinstance(macro_data.columns, pd.MultiIndex) else '^TNX' in macro_data
        
        if has_tnx:
            try:
                tnx_series = macro_data['^TNX']['Close'].dropna() if isinstance(macro_data.columns, pd.MultiIndex) else macro_data['^TNX'].dropna()
                tnx_chg = tnx_series.pct_change().iloc[-1]
                
                if tnx_chg > 0.015:
                    macro_msg = "🚨 **RANDAMENTE ÎN CREȘTERE:** Yield-ul 10Y crește brusc. Acest lucru pune presiune pe acțiunile de Tehnologie (Growth) și crește costul creditării."
                elif tnx_chg < -0.015:
                    macro_msg = "🟢 **REDUCERE COST CAPITAL:** Yield-ul 10Y scade. Un mediu favorabil pentru acțiuni și pentru refinanțarea datoriilor companiilor."
                else:
                    macro_msg = "⚖️ **STABILITATE DOBÂNZI:** Yield-ul 10Y este stabil. Piața nu anticipează schimbări majore de politică monetară în acest moment."
                st.info(macro_msg)
            except:
                st.info("💡 Interpretare: Dacă US 10Y Yield crește brusc, acțiunile de tehnologie tind să scadă. Dacă Aurul crește, indică frică în piață.")
                
        # --- 1. CONFIGURARE UI (Selectori) ---
        c_sel1, c_sel2 = st.columns([1, 3])
        
        with c_sel1:
            st.markdown("##### 1. Alege Indicator:")
            selected_macro_name = st.radio("Indicator", list(macro_tickers.keys()), label_visibility="collapsed")
            selected_macro_sym = macro_tickers[selected_macro_name]
            
            st.markdown("##### 2. Perioadă:")
            # Slider pentru timp
            time_frame = st.select_slider("", options=["1L", "3L", "6L", "1A", "3A", "5A"], value="1A")

        # --- 2. PROCESARE DATE ---
        with c_sel2:
            # Extragere Serie de Date
            series = pd.Series()
            if isinstance(macro_data.columns, pd.MultiIndex):
                try:
                    if selected_macro_sym in macro_data.columns.levels[0]:
                        series = macro_data[selected_macro_sym]['Close'].dropna()
                except: pass
            else:
                series = macro_data['Close'] # Fallback

            if not series.empty:
                # 1. Date pentru Interval (Subset)
                subset = slice_window(series, time_frame)  # fereastră calendaristică, nu număr de rânduri
                
                # --- CALCULE METRICI SEPARATE (MODIFICAREA CERUTĂ) ---
                
                curr_val = series.iloc[-1] # Valoarea curentă (Azi)

                # A. Calcul Interval (Start Slider vs Azi)
                start_val = subset.iloc[0]
                diff_interval = curr_val - start_val
                pct_interval = (diff_interval / start_val) * 100 if start_val != 0 else 0

                # B. Calcul Zi (Ieri vs Azi) - Folosim seria completă, nu subsetul
                prev_day_val = series.iloc[-2] if len(series) >= 2 else curr_val
                diff_day = curr_val - prev_day_val
                pct_day = (diff_day / prev_day_val) * 100 if prev_day_val != 0 else 0
                
                # --- FORMATARE TEXT ---
                suffix = "%" if "Yield" in selected_macro_name else ""
                val_fmt = f"{curr_val:.4f}{suffix}"
                
                # --- AFIȘARE DUALĂ (2 Metric Carduri) ---
                m1, m2 = st.columns(2)
                
                m1.metric(
                    f"Interval ({time_frame})", 
                    val_fmt, 
                    f"{diff_interval:.4f} ({pct_interval:.2f}%)"
                )
                
                m2.metric(
                    "Evoluție Azi", 
                    val_fmt, 
                    f"{diff_day:.4f} ({pct_day:.2f}%)"
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
                is_stable = (y_max - y_min) / y_min < 0.1 
                
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
                
                st.plotly_chart(fig_macro, width='stretch')
                # --- NOU: MODUL INTERPRETARE DINAMICĂ MACRO ---
                st.markdown("#### 🧠 Analiza Corelațiilor (Ghid Macro)")
                
                # Trimitem tot bulk-ul de date macro descărcat anterior
                macro_verdicts = get_macro_interpretation(macro_data)
                
                for v in macro_verdicts:
                    st.info(v)
                
                # --- TABEL IMPACT SECTORIAL DINAMIC (AMBELE SCENARII) ---
                st.markdown("---")
                st.subheader("📊 Matricea de Sensibilitate Economică (Scenarii)")
                
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
                
                try:
                    # 1. Date și Calcule (Rămân la fel)
                    rot_data = yf.download(['^IXIC', '^DJI'], period="1y", progress=False)['Close']
                    ratio = rot_data['^IXIC'] / rot_data['^DJI']
                    ratio_sma = ratio.rolling(window=50).mean()
                    
                    current_ratio = ratio.iloc[-1]
                    prev_ratio = ratio.iloc[-22]
                    rot_change = ((current_ratio - prev_ratio) / prev_ratio) * 100

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
                    st.plotly_chart(fig_rot, width='stretch')

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
                                    Trend Rotație (30z)
                                </div>
                                <div style="font-size: 22px; font-weight: bold; color: #FFFFFF; margin-bottom: 18px;">
                                    {rot_status}
                                </div>
                                <div style="display: flex; align-items: center; justify-content: space-between;">
                                    <div style="background: {delta_color}22; color: {delta_color}; padding: 4px 12px; border-radius: 6px; font-weight: bold; font-size: 15px; border: 1px solid {delta_color}44;">
                                        {rot_change:+.2f}%
                                    </div>
                                    <div style="color: #8B949E; font-size: 13px;">
                                        Scor real: <span style="color: #FFFFFF; font-weight: bold; font-family: 'Courier New', monospace; font-size: 16px;">{current_ratio:.4f}</span>
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
                            
                except Exception as e:
                    st.info("Indicatorul de rotație se recalibrează...")
                    
                    # --- MODUL: BAROMETRU DE SENTIMENT GLOBAL (ȘTIRI) ---
                st.markdown("---")
                st.subheader("🎭 Barometru Sentiment Global al Media")
                
                try:
                    # Colectăm știrile de la indicii majori pentru un eșantion relevant
                    news_samples = get_company_news_rss("^GSPC") + get_company_news_rss("^IXIC")
                    
                    if news_samples:
                        # Calculăm scorul mediu folosind motorul tău de IA
                        from ai_engine import analyze_sentiment_ai
                        global_sentiment_score = analyze_sentiment_ai(news_samples)
                        
                        # Definire culori și mesaje profesionale
                        if global_sentiment_score > 0.15:
                            s_col, s_msg = "#3FB950", "🚀 BULLISH: Narațiunea globală este optimistă."
                        elif global_sentiment_score < -0.15:
                            s_col, s_msg = "#F85149", "📉 BEARISH: Predomină frica și incertitudinea."
                        else:
                            s_col, s_msg = "#8B949E", "⚖️ NEUTRU: Media reflectă o perioadă de consolidare."
                            
                        # Afișare vizuală
                        sb1, sb2 = st.columns([1, 2])
                        with sb1:
                            st.markdown(f"""
                            <div style="background:#161B22; padding:20px; border-radius:15px; border:2px solid {s_col}; text-align:center;">
                                <p style="color:#8B949E; margin:0; font-size:11px; text-transform:uppercase;">Scor Sentiment Media</p>
                                <h1 style="color:{s_col}; margin:10px 0; font-size:36px;">{global_sentiment_score:.2f}</h1>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with sb2:
                            st.info(s_msg)
                            st.write("**Corelație cu Rotația:**")
                            # Logică de corelare automată
                            if global_sentiment_score < 0 and rot_change < 0:
                                st.write("⚠️ **CONFIRMARE DEFENSIVĂ:** Atât știrile, cât și mișcările de capital (rotația) indică o fugă către siguranță.")
                            elif global_sentiment_score > 0 and rot_change > 0:
                                st.write("🌟 **CONFIRMARE GROWTH:** Sentimentul pozitiv din media susține migrarea banilor către Tehnologie.")
                            else:
                                st.write("🔄 **DIVERGENȚĂ:** Piața se mișcă într-o direcție, dar media raportează altceva. Atenție la potențiale capcane!")
                    else:
                        st.info("Sincronizare fluxuri știri pentru barometru...")
                except:
                    st.info("Barometrul de sentiment se actualizează...")

            else:
                st.warning("Date indisponibile sau eroare conexiune Yahoo.")
        
        # =================================================================
        # MODUL NOU: HARTA TERMICĂ A SECTOARELOR (MONEY FLOW)
        # =================================================================
        st.markdown("---")
        st.subheader("🗺️ Harta Termică a Sectoarelor (Money Flow)")
        st.markdown("Radiografia indicelui S&P 500: Urmărește în timp real în ce sectoare își mută instituțiile capitalul astăzi.")

        with st.spinner("Se calculează fluxul de capital pe sectoare..."):
            df_sectors = get_sector_performance()
            
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
                    text=[f"{val:+.2f}%" for val in df_sectors['Variație %']],
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
                    st.plotly_chart(fig_sec_heat, width='stretch')
                
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
                    top_sector = top_3.iloc[0] # Liderul absolut al zilei
                    
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
                    st.write("**🏆 Liderii Zilei (Top 3):**")
                    for _, row in top_3.iterrows():
                        st.write(f"<span style='color:#3FB950;'>▲</span> {row['Sector']} <span style='color:#8B949E; font-size:12px;'>({row['Variație %']:+.2f}%)</span>", unsafe_allow_html=True)
                        
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Afișare Codași cu procente și culori inteligente
                    st.write("**🐢 Codașii Zilei (Bottom 3):**")
                    for _, row in bottom_3.iterrows():
                        if row['Variație %'] < 0:
                            # E pe minus -> Săgeată roșie
                            st.write(f"<span style='color:#F85149;'>▼</span> {row['Sector']} <span style='color:#8B949E; font-size:12px;'>({row['Variație %']:+.2f}%)</span>", unsafe_allow_html=True)
                        else:
                            # E "codaș", dar e pe plus -> Săgeată galbenă (neutră)
                            st.write(f"<span style='color:#D29922;'>▲</span> {row['Sector']} <span style='color:#8B949E; font-size:12px;'>({row['Variație %']:+.2f}%)</span>", unsafe_allow_html=True)
        st.markdown("---")
        # =================================================================
        # MODUL NOU: RADAR DE RISC SISTEMIC (CU SELECTOR ȘI ZOOM)
        # =================================================================
        st.subheader("💣 Radar de Risc Sistemic (Piața de Credit)")
        
        # --- NOU: SELECTOR DE PERIOADĂ ---
        col_title, col_time = st.columns([2, 1])
        with col_time:
            time_map_bonds = {
                "1 Lună": "1mo", "3 Luni": "3mo", "6 Luni": "6mo", 
                "1 An": "1y", "3 Ani": "3y", "5 Ani": "5y"
            }
            selected_period_label = st.selectbox("Interval Radar:", list(time_map_bonds.keys()), index=3)
            selected_period_code = time_map_bonds[selected_period_label]

        with st.spinner(f"Se analizează stresul financiar pe {selected_period_label}..."):
            ratio_series = get_credit_risk_data(selected_period_code)
            
            if not ratio_series.empty:
                current_ratio = ratio_series.iloc[-1]
                # Calculăm media pe 20 de zile pentru context
                sma_20 = ratio_series.rolling(20).mean().iloc[-1] if len(ratio_series) >= 20 else ratio_series.mean()
                
                # Calculăm variația procentuală pe intervalul selectat
                start_ratio = ratio_series.iloc[0]
                total_change = ((current_ratio - start_ratio) / start_ratio) * 100
                
                c_bond1, c_bond2 = st.columns([2, 1.2])
                
                with c_bond1:
                    # --- GRAFIC CU ZOOM DINAMIC ---
                    fig_bonds = go.Figure()
                    
                    # Linia principală
                    fig_bonds.add_trace(go.Scatter(
                        x=ratio_series.index, y=ratio_series.values, 
                        mode='lines', name='Raport HYG/IEF', 
                        line=dict(color='#BF91FF', width=2.5),
                        fill='tozeroy', fillcolor='rgba(191, 145, 255, 0.05)'
                    ))
                    
                    # Adăugăm media mobilă pentru a vedea trendul
                    fig_bonds.add_trace(go.Scatter(
                        x=ratio_series.index, y=ratio_series.rolling(20).mean(), 
                        mode='lines', name='Trend (SMA 20)', 
                        line=dict(color='rgba(255, 255, 255, 0.3)', dash='dot')
                    ))

                    # TRUCUL PENTRU VIZIBILITATE: Zoom pe axa Y
                    y_min = ratio_series.min() * 0.99  # Luăm valoarea minimă și mai scădem 1%
                    y_max = ratio_series.max() * 1.01  # Luăm valoarea maximă și mai adăugăm 1%

                    fig_bonds.update_layout(
                        height=350, margin=dict(l=0, r=0, t=10, b=0), 
                        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                        yaxis=dict(
                            showgrid=True, gridcolor='#30363D',
                            range=[y_min, y_max], # ACEASTA ESTE LINIA CARE FACE ZOOM PE VARIAȚII
                            fixedrange=False
                        ),
                        xaxis=dict(showgrid=False),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_bonds, width='stretch')
                    
                with c_bond2:
                    st.markdown("#### 🌡️ Termometrul Creditării")
                    st.metric(f"Raport ({selected_period_label})", f"{current_ratio:.3f}", f"{total_change:+.2f}%")
                    
                    # Logica de interpretare adaptată
                    if current_ratio < sma_20:
                        st.error("🚨 **SCĂDERE DETECTATĂ:** Pofta de risc scade. Banii ies din obligațiunile firmelor și intră în titluri de stat.")
                    else:
                        st.success("✅ **STABILITATE:** Piața de credit susține încă evaluările acțiunilor.")
                        
                    st.info(f"💡 **Interpretare:** În intervalul de **{selected_period_label}**, linia s-a mișcat cu **{total_change:+.2f}%**. Orice pantă bruscă în jos pe grafic indică faptul că băncile devin nervoase.")
            else:
                st.info("Sincronizare date obligațiuni...")
        
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
                            row_text.append(f"{val:.2f}")
                        text_matrix.append(row_text)

                    # Desenăm Heatmap-ul (Verde = merg împreună, Roșu = opuse)
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale=[[0.0, '#F85149'], [0.5, '#161B22'], [1.0, '#3FB950']], 
                        zmin=-1, zmax=1,
                        text=text_matrix,
                        texttemplate="%{text}",
                        hoverinfo="text"
                    ))
                    
                    fig_corr.update_layout(
                        height=400,
                        template="plotly_dark",
                        margin=dict(l=0, r=0, t=10, b=0),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_corr, width='stretch')
                    
                with c_cross2:
                    st.markdown("#### 🧠 Radiografia Fluxului Global")
                    
                    # AI-ul trage concluzii automate pe baza intersecțiilor critice
                    try:
                        spy_tlt = corr_matrix.loc['Acțiuni (SPY)', 'Bonduri (TLT)']
                        spy_uup = corr_matrix.loc['Acțiuni (SPY)', 'Dolar (UUP)']
                        gld_uup = corr_matrix.loc['Aur (GLD)', 'Dolar (UUP)']
                        
                        # Regula 1: Risc vs Siguranță (SPY vs TLT)
                        if spy_tlt > 0.4:
                            st.error(f"🚨 **Șoc Inflaționist:** Acțiunile și Obligațiunile se mișcă împreună (Corelație +{spy_tlt:.2f}). Diversificarea clasică (60/40) nu te protejează. Cash-ul este rege.")
                        elif spy_tlt < -0.4:
                            st.success(f"✅ **Piață Normală:** Corelația Acțiuni/Bonduri este negativă ({spy_tlt:.2f}). Banii se mută ordonat între Risc și Siguranță.")
                        else:
                            st.warning(f"⚖️ **Tranziție:** Fără direcție clară între Acțiuni și Bonduri ({spy_tlt:.2f}).")
                        
                        # Regula 2: Dolarul (Wrecking Ball)
                        if spy_uup < -0.5:
                            st.info(f"💵 **Dolarul dictează:** Dolarul puternic lovește acțiunile (Corelație {spy_uup:.2f}). Urmărește moneda americană; dacă ea scade, bursa explodează.")
                            
                        # Regula 3: Aurul ca panică pură
                        if gld_uup > 0.3:
                            st.warning(f"🥇 **Frică Extremă:** Aurul crește ÎMPREUNĂ cu Dolarul (Corelație {gld_uup:.2f}). Marile fonduri cumpără masiv orice activ de refugiu, ignorând matematica standard.")
                    except:
                        st.write("Date insuficiente pentru diagnoza automată.")
                        
                    with st.expander("📖 Cum citești matricea?"):
                        st.write("""
                        * **Pătrate Verzi (+0.5 la +1.0):** Activele sunt "prietene". Dacă unul crește, crește și celălalt.
                        * **Pătrate Roșii (-0.5 la -1.0):** Activele sunt "inamice". Când unul urcă, celălalt scade (Corelație Inversă). Aceasta este cheia hedging-ului perfect.
                        * **Pătrate Negre (în jur de 0.0):** Activele se ignoră complet reciproc.
                        """)
            else:
                st.info("Sincronizare date Cross-Asset...")
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
                            'Valoare Curentă': '{:.2f}',
                            'Lună Precedentă': '{:.2f}'
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
            color = '#3FB950' if val >= 0 else '#F85149'
            return f'color: {color}'

        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.subheader("📊 Indici Principali")
            st.dataframe(
                df_ind.style.map(color_change_val, subset=['Variație', 'Variație %'])
                .format({'Preț': '{:.2f}', 'Variație': '{:.2f}', 'Variație %': '{:.2f}%'}),
                width='stretch', hide_index=True
            )
            
        with col_m2:
            st.subheader("🛢️ Mărfuri (Commodities)")
            st.dataframe(
                df_comm.style.map(color_change_val, subset=['Variație', 'Variație %'])
                .format({'Preț': '{:.2f}', 'Variație': '{:.2f}', 'Variație %': '{:.2f}%'}),
                width='stretch', hide_index=True
            )

        st.markdown("---")
        
        st.subheader("🇺🇸 Top Mișcări SUA (Blue Chips)")
        c_us1, c_us2 = st.columns(2)
        
        with c_us1:
            st.markdown("**🚀 Top Creșteri (Gainers)**")
            if not us_gain.empty:
                st.dataframe(
                    us_gain[['Instrument', 'Preț', 'Variație %']].style
                    .map(color_change_val, subset=['Variație %'])
                    .format({'Preț': '{:.2f}', 'Variație %': '{:.2f}%'}),
                    width='stretch', hide_index=True
                )
        
        with c_us2:
            st.markdown("**🔻 Top Scăderi (Losers)**")
            if not us_lose.empty:
                st.dataframe(
                    us_lose[['Instrument', 'Preț', 'Variație %']].style
                    .map(color_change_val, subset=['Variație %'])
                    .format({'Preț': '{:.2f}', 'Variație %': '{:.2f}%'}),
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
                    .format({'Preț': '{:.2f}', 'Variație %': '{:.2f}%'}),
                    width='stretch', hide_index=True
                )
        
        with c_eu2:
            st.markdown("**🔻 Top Scăderi (Losers)**")
            if not eu_lose.empty:
                st.dataframe(
                    eu_lose[['Instrument', 'Preț', 'Variație %']].style
                    .map(color_change_val, subset=['Variație %'])
                    .format({'Preț': '{:.2f}', 'Variație %': '{:.2f}%'}),
                    width='stretch', hide_index=True
                )

    # ==================================================
    # 5. IMPORT DATE (GOOGLE SHEETS) - BVB EXTINS & GLOBAL FIX
    # ==================================================
    elif sectiune == "5. Import Date":
        st.title("📂 Analiză Date (Cloud Sheets)")
        st.caption("Datele sunt curățate și standardizate automat (Format RO & US).")
        
        if st.button("🔄 Reîncarcă Datele"):
            st.cache_data.clear()
            st.rerun()

        tab_bvb, tab_global = st.tabs(["🇷🇴 BVB (Local)", "🌍 Internațional (Global)"])

        # Funcție locală de încărcare
        def load_gsheet_data(sheet_name):
            ws = connect_to_gsheets(sheet_name)  # Folosim singleton-ul
            if not ws: return pd.DataFrame()
            try:
                data = ws.get_all_values()
                if len(data) < 2: return pd.DataFrame()
                df = pd.DataFrame(data[1:], columns=data[0])
                return df
            except Exception as e:
                st.error(f"Eroare citire {sheet_name}: {e}")
                return pd.DataFrame()

        # --- TAB BVB (DATE EXTINSE) ---
        with tab_bvb:
            st.subheader("Date BVB")
            df_bvb = load_gsheet_data("BVB")

            if not df_bvb.empty:
                try:
                    col_indicators = df_bvb.columns[1] 
                    df_bvb = df_bvb[df_bvb[col_indicators] != ""]
                    
                    # Eliminăm duplicatele de pe coloana indicatorilor
                    df_bvb = df_bvb.drop_duplicates(subset=[col_indicators], keep='first')
                    
                    # Transpunere
                    final_df = df_bvb.set_index(col_indicators).T
                    final_df = final_df.loc[:, ~final_df.columns.str.contains('^Unnamed')] 

                    # === LISTA EXTINSĂ DE COLOANE NUMERICE ===
                    cols_numeric = [
                        "P/E 2024", "P/E TTM", "EV/EBITDA", "P/BV TTM", "GN", "P/S TTM",
                        "Rentabilitate active (ROA)", "Rentabilitate capital (ROE)",
                        "Marjă netă TTM", "Marjă operațională", "Câștig pe acțiune (EPS)", "EPS TTM",
                        "Lichiditate curentă", "Lichiditatea imediată", "Levier financiar",
                        "Div Yield", "Dividend Yield", "Net Debt/EBITDA", "Debt/EBITDA",
                        "Rata de îndatorare globală", "Rata de cash din capitalizare", "Rata de cash din activ net"
                    ]

                    for col in final_df.columns:
                        col_clean = col.strip()
                        # Verificăm dacă e în listă sau conține indicii de număr
                        if col_clean in cols_numeric or "%" in col_clean or "Ron" in col_clean or "lei" in col_clean:
                            final_df[col] = final_df[col].apply(smart_to_float)

                    st.dataframe(
                        final_df, height=600, width='stretch',
                        column_config={
                            # Rentabilitate & Marje
                            "Rentabilitate active (ROA)": st.column_config.NumberColumn(format="%.2f%%"),
                            "Rentabilitate capital (ROE)": st.column_config.NumberColumn(format="%.2f%%"),
                            "Marjă netă TTM": st.column_config.NumberColumn(format="%.2f%%"),
                            "Marjă operațională": st.column_config.NumberColumn(format="%.2f%%"),
                            
                            # Dividende
                            "Div Yield": st.column_config.NumberColumn(format="%.2f%%"),
                            "Dividend Yield": st.column_config.NumberColumn(format="%.2f%%"),
                            
                            # EPS
                            "Câștig pe acțiune (EPS)": st.column_config.NumberColumn(format="%.4f"),
                            "EPS TTM": st.column_config.NumberColumn(format="%.4f"),
                            
                            # Lichiditate & Datorii
                            "Lichiditate curentă": st.column_config.NumberColumn(format="%.2f"),
                            "Lichiditatea imediată": st.column_config.NumberColumn(format="%.2f"),
                            "Levier financiar": st.column_config.NumberColumn(format="%.2f"),
                            "Net Debt/EBITDA": st.column_config.NumberColumn(format="%.2f"),
                            "Debt/EBITDA": st.column_config.NumberColumn(format="%.2f"),
                            "Rata de îndatorare globală": st.column_config.NumberColumn(format="%.2f%%"),
                            
                            # Cash Rates
                            "Rata de cash din capitalizare": st.column_config.NumberColumn(format="%.2f%%"),
                            "Rata de cash din activ net": st.column_config.NumberColumn(format="%.2f%%"),
                        }
                    )
                except Exception as e:
                    st.error(f"Eroare structură BVB: {e}")
                    st.dataframe(df_bvb.head())
            else:
                st.info("Foaia BVB este goală.")

        # --- TAB GLOBAL (FORMATĂRI CORRECTE) ---
        with tab_global:
            st.subheader("Date Internaționale")
            df_g = load_gsheet_data("GLOBAL")

            if not df_g.empty:
                try:
                    df_g = df_g.loc[:, ~df_g.columns.str.contains('^Unnamed')]
                    if "Companii" in df_g.columns:
                        df_g = df_g.set_index("Companii")

                    clean_df_g = df_g.copy()

                    for col in clean_df_g.columns:
                        if col in ["Industrie", "Recomandare", "Sector"]: continue
                        clean_df_g[col] = clean_df_g[col].apply(smart_to_float)

                    # Formatare string pentru afișare (Trilioane/Miliarde)
                    display_df = clean_df_g.copy()
                    if "Capitalizare" in display_df.columns:
                        display_df["Capitalizare"] = display_df["Capitalizare"].apply(format_large_currency)
                    if "Val. intrinsecă" in display_df.columns:
                        display_df["Val. intrinsecă"] = display_df["Val. intrinsecă"].apply(format_large_currency)

                    st.dataframe(
                        display_df, height=600, width='stretch',
                        column_config={
                            "Capitalizare": st.column_config.TextColumn("Capitalizare", help="Valoare formatată"),
                            
                            # AICI E FIX-UL PENTRU PREȚ ($)
                            "Preț acțiune": st.column_config.NumberColumn("Preț acțiune", format="$ %.2f"),
                            "Preț țintă": st.column_config.NumberColumn("Preț țintă", format="$ %.2f"),
                            "Dividend": st.column_config.NumberColumn("Dividend", format="$ %.2f"),
                            "Val. intrinsecă": st.column_config.TextColumn("Val. intrinsecă"),
                            
                            # AICI E FIX-UL PENTRU DATORII (%)
                            "Datorii/Ac. Net": st.column_config.NumberColumn("Datorii/Ac. Net", format="%.2f%%"),
                            "Abatere": st.column_config.NumberColumn(format="%.2f%%"),
                            "Marjă P. Net": st.column_config.NumberColumn(format="%.2f%%"),
                            "ROA": st.column_config.NumberColumn(format="%.2f%%"),
                            "ROE": st.column_config.NumberColumn(format="%.2f%%"),
                            "Recomandare": st.column_config.TextColumn("Recomandare"),
                        }
                    )
                except Exception as e:
                    st.error(f"Eroare procesare Global: {e}")
                    st.dataframe(df_g)
            else:
                st.info("Foaia GLOBAL este goală.")

    # ==================================================
    # 6. REZUMATUL ZILEI (NOU & OPTIMIZAT)
    # ==================================================
    elif sectiune == "6. Rezumatul Zilei":
        st.title("🗞️ Rezumatul Zilei")
        st.markdown("Raport automat generat la închiderea piețelor.")
        
        now = now_ro()  # ora României, nu ora serverului (UTC)
        # Bursa din SUA deschide la 16:30, ora României
        us_market_not_open_yet = (now.hour, now.minute) < (16, 30)
        
        # Obținem datele
        with st.spinner("Generăm rezumatul pieței..."):
            bvb_data, us_data = get_daily_briefing_data()
        
        # --- TABURI PENTRU PIEȚE ---
        tab_bvb, tab_us = st.tabs(["🇷🇴 BVB (Ora 19:00)", "🇺🇸 Wall Street (Ora 23:00)"])
        
        # === REZUMAT BVB ===
        with tab_bvb:
            st.markdown(f"### 📅 Raport Bursa de Valori București - {now.strftime('%d-%m-%Y')}")
            
            # 1. Narrativa Principală (Indicele BET) - ACUM MODIFICAT SĂ ARATE CA WALL STREET
            bet_text, bet_change, bet_price = generate_market_narrative(bvb_data, 'TVBETETF.RO', 'Indicele BET')
            
            # Determinăm culoarea în funcție de schimbare
            c_bet = "#3FB950" if bet_change >= 0 else "#F85149"
            
            # Afișare stil "Card" (similar cu Wall Street)
            st.markdown(f"""
            <div style="background-color: #161B22; padding: 15px; border-radius: 10px; border-left: 5px solid {c_bet}; margin-bottom: 20px;">
                <h4 style="margin-top:0;">🇷🇴 Evoluția Pieței Locale</h4>
                <p style="margin:5px 0; font-size:18px;">
                    📉 <b>BET (TVBETETF):</b> {bet_price:,.2f} RON <span style="color:{c_bet}; font-weight:bold;">({bet_change:+.2f}%)</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # --- NOU: SENTIMENTUL PIEȚEI BVB (RADAR) ---
            # Calculăm scorul folosind funcția de proxy pentru BVB
            bvb_score, bvb_sentiment_label = calculate_bvb_sentiment(bvb_data)
            
            # Stabilim culoarea vizuală în funcție de scor
            b_color = "#3FB950" if bvb_score > 55 else ("#F85149" if bvb_score < 45 else "#8B949E")
            
            st.markdown(f"""
            <div style="background-color: #161B22; padding: 15px; border-radius: 10px; border: 1px solid {b_color}44; margin-bottom: 20px;">
                <div style="font-size: 12px; color: #8B949E; text-transform: uppercase; letter-spacing: 1px;">Pulsul Pieței Locale (BVB)</div>
                <div style="font-size: 20px; font-weight: bold; color: {b_color};">{bvb_sentiment_label}</div>
            </div>
            """, unsafe_allow_html=True)

            # Analiza Contextuală BVB (Explicație pentru investitor)
            if bvb_score < 35:
                bvb_conclusion = "🚨 **Vânzări emoționale pe BVB:** Piața locală este sub presiune. Investitorii tind să iasă din poziții, ceea ce poate crea oportunități pe companiile cu dividende mari."
            elif bvb_score > 65:
                bvb_conclusion = "🚀 **Apetit crescut pentru risc:** Există un val de optimism pe companiile din indexul BET. Atenție la raliurile care nu sunt susținute de volume mari."
            else:
                bvb_conclusion = "⚖️ **Stabilitate locală:** Bursa de la București tranzacționează calm, fără mișcări speculative majore."
            
            st.info(f"🇷🇴 {bvb_conclusion}")

            # Calculăm statisticile extinse
            if isinstance(bvb_data.columns, pd.MultiIndex):
                bvb_analysis_tickers = bvb_data.columns.levels[0].tolist()
            else:
                bvb_analysis_tickers = []

            gainers, losers, vol_leaders = get_bvb_stats(bvb_data, bvb_analysis_tickers)
            st.markdown("---")
            # =========================================================
            # NOU: MARKET BREADTH BVB
            # =========================================================
            st.markdown("#### ⚖️ Sănătatea Pieței (Market Breadth)")
            adv_b, dec_b, flat_b, adv_pb, dec_pb, flat_pb = calculate_market_breadth(bvb_data, bvb_analysis_tickers)
            
            if (adv_b + dec_b + flat_b) > 0:
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; margin-bottom: 8px; font-size: 14px;">
                    <span style="color:#3FB950; font-weight:bold;">🟢 Cresc: {adv_b} ({adv_pb:.1f}%)</span>
                    <span style="color:#8B949E; font-weight:bold;">⚪ Neutru: {flat_b}</span>
                    <span style="color:#F85149; font-weight:bold;">🔴 Scad: {dec_b} ({dec_pb:.1f}%)</span>
                </div>
                <div style="display: flex; height: 14px; border-radius: 7px; overflow: hidden; background-color: #30363D; margin-bottom: 5px;">
                    <div style="width: {adv_pb}%; background-color: #3FB950;"></div>
                    <div style="width: {flat_pb}%; background-color: #8B949E;"></div>
                    <div style="width: {dec_pb}%; background-color: #F85149;"></div>
                </div>
                """, unsafe_allow_html=True)
                
                # Verdict Pro Automat
                if adv_b > dec_b * 1.5:
                    st.caption("✅ **Raliu Sănătos:** Creșterea pieței este susținută de majoritatea companiilor (Participare largă).")
                elif dec_b > adv_b * 1.5:
                    st.caption("🚨 **Vânzare Generalizată:** Scăderile nu sunt izolate, presiunea acoperă tot spectrul.")
                else:
                    st.caption("🔄 **Piață Mixtă:** Echilibru. Banii se mută selectiv dintr-o acțiune în alta.")
            st.write("") # Spațiu
            st.markdown("---")
            # =========================================================
            # NOU: NIVELE TEHNICE CRITICE (BVB)
            # =========================================================
            st.markdown("#### 🎯 Nivele Tehnice Critice (Trend Macro)")
            with st.spinner("Calculăm mediile mobile..."):
                tech_bvb = get_index_technical_levels('TVBETETF.RO')
                
            if tech_bvb:
                c50 = "#3FB950" if tech_bvb['dist50'] > 0 else "#F85149"
                c200 = "#3FB950" if tech_bvb['dist200'] > 0 else "#F85149"
                
                # Diagnoza Trendului
                if tech_bvb['dist50'] > 0 and tech_bvb['dist200'] > 0:
                    regime_msg = "🟢 **BULL MARKET:** Prețul este deasupra mediilor majore. Trend ascendent confirmat."
                elif tech_bvb['dist50'] < 0 and tech_bvb['dist200'] < 0:
                    regime_msg = "🔴 **BEAR MARKET:** Prețul a spart suporturile majore. Presiune de vânzare activă."
                else:
                    regime_msg = "🟡 **ZONĂ DE TRANZIȚIE:** Prețul este prins între SMA 50 și SMA 200. Indecizie majoră."
                
                # Golden / Death Cross
                cross_msg = "✨ *Golden Cross activ* (SMA 50 este deasupra SMA 200)" if tech_bvb['sma50'] > tech_bvb['sma200'] else "☠️ *Death Cross activ* (SMA 50 a picat sub SMA 200)"

                st.markdown(f"""
                <div style="display:flex; gap:15px; margin-bottom: 10px;">
                    <div style="flex:1; background:#161B22; padding:15px; border-radius:10px; border:1px solid #30363D;">
                        <div style="color:#8B949E; font-size:11px; text-transform:uppercase;">Față de SMA 50 (Mediu)</div>
                        <div style="font-size:24px; font-weight:bold; color:{c50};">{tech_bvb['dist50']:+.2f}%</div>
                        <div style="color:#C9D1D9; font-size:12px; margin-top:5px;">Prag suport/rezistență: <b>{tech_bvb['sma50']:.2f}</b></div>
                    </div>
                    <div style="flex:1; background:#161B22; padding:15px; border-radius:10px; border:1px solid #30363D;">
                        <div style="color:#8B949E; font-size:11px; text-transform:uppercase;">Față de SMA 200 (Lung)</div>
                        <div style="font-size:24px; font-weight:bold; color:{c200};">{tech_bvb['dist200']:+.2f}%</div>
                        <div style="color:#C9D1D9; font-size:12px; margin-top:5px;">Prag suport/rezistență: <b>{tech_bvb['sma200']:.2f}</b></div>
                    </div>
                </div>
                <div style="background:#21262D; padding:12px; border-radius:8px; font-size:14px; border-left: 4px solid {'#3FB950' if tech_bvb['dist200'] > 0 else '#F85149'};">
                    {regime_msg} <br> <span style="color:#8B949E; font-size:12px;">{cross_msg}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Date tehnice indisponibile pentru acest indice.")
            st.markdown("---")
            
            # =========================================================
            # NOU: INTENSITATE VOLUM (BVB)
            # =========================================================
            st.markdown("#### 🔊 Intensitatea Tranzacționării (Volume)")
            vol_bvb = get_volume_analysis('TVBETETF.RO')
            
            if vol_bvb:
                v_int = vol_bvb['intensity']
                # Culori în funcție de intensitate
                v_color = "#3FB950" if v_int > 1.2 else ("#D29922" if v_int > 0.8 else "#8B949E")
                v_label = "ACTIVITATE RIDICATĂ" if v_int > 1.2 else ("NORMAL" if v_int > 0.8 else "LICHIDITATE SCĂZUTĂ")
                
                # Formula intensității: $I = \frac{V_{current}}{V_{average}}$
                st.markdown(f"""
                <div style="background:#161B22; padding:20px; border-radius:12px; border-top: 4px solid {v_color};">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <div style="color:#8B949E; font-size:12px; text-transform:uppercase;">Volum vs Media 20z</div>
                            <div style="font-size:28px; font-weight:bold; color:white;">{v_int:.2f}x</div>
                        </div>
                        <div style="text-align:right;">
                            <span style="background:{v_color}22; color:{v_color}; padding:5px 12px; border-radius:20px; font-weight:bold; font-size:12px;">
                                {v_label}
                            </span>
                        </div>
                    </div>
                    <div style="margin-top:15px; background:#30363D; height:8px; border-radius:4px; overflow:hidden;">
                        <div style="width:{min(v_int*50, 100)}%; background:{v_color}; height:100%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if v_int > 1.5:
                    st.caption("🔥 **Alertă:** Volum neobișnuit de mare. Instituțiile fac mutări importante în piața locală.")
                elif v_int < 0.6:
                    st.caption("💤 **Atenție:** Piață „adormită”. Mișcările de preț pot fi cauzate de ordine mici de retail.")
            # =========================================================
            st.markdown("---")
            
            # 2. Top Movers (5 companii)
            col_mov1, col_mov2 = st.columns(2)
            
            with col_mov1:
                st.markdown("**🚀 Top 5 Creșteri**")
                if not gainers.empty:
                    st.dataframe(
                        gainers[['Simbol', 'Preț', 'Variație']].style
                        .format({'Preț': '{:.2f}', 'Variație': '{:+.2f}%'})
                        .map(lambda x: 'color: #3FB950', subset=['Variație']),
                        width='stretch', hide_index=True
                    )
                else: st.info("Date indisponibile.")
                
            with col_mov2:
                st.markdown("**🔻 Top 5 Scăderi**")
                if not losers.empty:
                    st.dataframe(
                        losers[['Simbol', 'Preț', 'Variație']].style
                        .format({'Preț': '{:.2f}', 'Variație': '{:+.2f}%'})
                        .map(lambda x: 'color: #F85149', subset=['Variație']),
                        width='stretch', hide_index=True
                    )
                else: st.info("Date indisponibile.")
            
            st.markdown("---")
            
            # 3. Clasament Volum (Top 10)
            st.subheader("📊 Top Lichiditate (Volume Tranzacționate)")
            if not vol_leaders.empty:
                def format_vol(x):
                    if x > 1e6: return f"{x/1e6:.2f} M"
                    if x > 1e3: return f"{x/1e3:.2f} K"
                    return f"{x:.0f}"
                
                vol_display = vol_leaders.copy()
                vol_display['Volum'] = vol_display['Volum'].apply(format_vol)
                
                st.dataframe(
                    vol_display[['Simbol', 'Preț', 'Volum', 'Variație']].style
                    .format({'Preț': '{:.2f}', 'Variație': '{:+.2f}%'})
                    .applymap(lambda x: 'color: #3FB950' if x > 0 else 'color: #F85149', subset=['Variație']),
                    width='stretch', hide_index=True
                )
            else:
                st.info("Nu există date despre volume.")

            # 4. Top 5 Știri România
            st.markdown("---")
            st.subheader("🇷🇴 Top 5 Știri Financiare (România)")
            
            if 'raw_news' not in st.session_state:
                raw_news = fetch_news_data()
            else:
                raw_news = st.session_state.get('raw_news', fetch_news_data())
            
            ro_sources = ["Ziarul Financiar", "Biziday", "Economica", "Bursa", "Profit.ro", "StartupCafe", "Financial Intelligence", "Wall-Street"]
            ro_news = [n for n in raw_news if any(src.lower() in n['source'].lower() for src in ro_sources)]
            if not ro_news:
                 ro_news = filter_news(raw_news, "Financiar") + filter_news(raw_news, "Energie")
            
            seen = set()
            unique_ro_news = []
            for n in ro_news:
                if n['title'] not in seen:
                    unique_ro_news.append(n)
                    seen.add(n['title'])
            
            if unique_ro_news:
                news_html = ""
                for item in unique_ro_news[:5]:
                      news_html += f"""
                      <div style="margin-bottom: 10px; border-bottom: 1px solid #30363D; padding-bottom: 5px;">
                        <a href="{item['link']}" style="color: #58A6FF; text-decoration: none; font-weight: 600;" target="_blank">
                           {item['title']}
                        </a>
                        <div style="font-size: 12px; color: #8B949E;">{item['source']} • {item['date_str']}</div>
                      </div>
                      """
                st.markdown(news_html, unsafe_allow_html=True)
            else:
                st.info("Nu s-au găsit știri locale recente.")

        # === REZUMAT SUA ===
        with tab_us:
            msg_us = ""
            if us_market_not_open_yet:
                msg_us = "(Datele afișate sunt de la închiderea precedentă)"
            
            st.markdown(f"### 🌎 Raport Wall Street {msg_us}")
            
            # --- 1. Indici Principali & Fear Index ---
            c_idx, c_fg = st.columns([2, 1])
            
            with c_idx:
                # ACUM PRIMIM SI PRETUL (PRICE)
                sp500_txt, sp500_chg, sp500_price = generate_market_narrative(us_data, '^GSPC', 'S&P 500')
                nasdaq_txt, nasdaq_chg, nasdaq_price = generate_market_narrative(us_data, '^IXIC', 'Nasdaq')
                dow_txt, dow_chg, dow_price = generate_market_narrative(us_data, '^DJI', 'Dow Jones')
                
                # Culori border
                us_border = "#3FB950" if sp500_chg >= 0 else "#F85149"
                
                # Culori text (verde/rosu) pentru fiecare indice
                c_sp = "#3FB950" if sp500_chg >= 0 else "#F85149"
                c_nq = "#3FB950" if nasdaq_chg >= 0 else "#F85149"
                c_dj = "#3FB950" if dow_chg >= 0 else "#F85149"
                
                st.markdown(f"""
                <div style="background-color: #161B22; padding: 15px; border-radius: 10px; border-left: 5px solid {us_border};">
                    <p style="margin:5px 0; font-size:16px;">
                        🇺🇸 <b>S&P 500:</b> {sp500_price:,.2f} <span style="color:{c_sp}; font-weight:bold;">({sp500_chg:+.2f}%)</span>
                    </p>
                    <p style="margin:5px 0; font-size:16px;">
                        💻 <b>Nasdaq:</b> {nasdaq_price:,.2f} <span style="color:{c_nq}; font-weight:bold;">({nasdaq_chg:+.2f}%)</span>
                    </p>
                    <p style="margin:5px 0; font-size:16px;">
                        🏭 <b>Dow Jones:</b> {dow_price:,.2f} <span style="color:{c_dj}; font-weight:bold;">({dow_chg:+.2f}%)</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)

            with c_fg:
                # Preluăm datele calculate de funcția ta existentă
                fg_score, fg_label, vix_val = calculate_fear_greed_proxy(us_data)
                
                # Stabilim culoarea în funcție de sentiment
                fg_color = "#F85149" if fg_score < 40 else ("#3FB950" if fg_score > 60 else "#8B949E")
                
                st.markdown(f"""
                <div style="text-align: center; background-color: #21262D; padding: 15px; border-radius: 12px; border: 1px solid {fg_color}44;">
                    <small style="color: #8B949E; text-transform: uppercase; letter-spacing: 1px;">Sentimentul Wall Street</small>
                    <h1 style="color: {fg_color}; margin: 10px 0; font-size: 42px;">{int(fg_score)}</h1>
                    <div style="font-weight:bold; color: #FFFFFF; font-size: 18px; margin-bottom: 5px;">{fg_label}</div>
                    <div style="font-size: 12px; color: #8B949E;">VIX (Volatilitate): {vix_val:.2f}</div>
                </div>
                """, unsafe_allow_html=True)

            # --- NOU: CONCLUZIA ZILEI (CONTEXTUALĂ) ---
            st.markdown("#### 🧠 Analiza Contextuală a Zilei")
            
            if fg_score < 30:
                conclusion = "🚨 **PANICĂ ÎN PIAȚĂ:** Sentimentul este de teamă extremă. Din punct de vedere contrarian, acestea sunt momentele în care se caută oportunități de cumpărare 'la reducere'."
            elif fg_score > 70:
                conclusion = "⚠️ **EUFORIE EXCESIVĂ:** Piața este lăcomă. Istoric, acest nivel precede adesea o corecție minoră. Atenție la noi intrări acum."
            else:
                conclusion = "⚖️ **ECHILIBRU:** Piața tranzacționează fără o direcție emoțională clară. Prețurile sunt dictate de datele economice, nu de impulsuri."
            
            st.info(conclusion)

            # --- 2. Top Movers & Volume (Top 10 Companii) ---
            if isinstance(us_data.columns, pd.MultiIndex):
                all_us_tickers = us_data.columns.levels[0].tolist()
            else:
                all_us_tickers = []
                
            us_analysis_tickers = [t for t in all_us_tickers if not t.startswith('^')]
            
            us_gainers, us_losers, us_vol = get_bvb_stats(us_data, us_analysis_tickers)
            st.markdown("---")
            
            # =========================================================
            # NOU: MARKET BREADTH SUA
            # =========================================================
            st.markdown("#### ⚖️ Sănătatea Pieței (Market Breadth)")
            adv_u, dec_u, flat_u, adv_pu, dec_pu, flat_pu = calculate_market_breadth(us_data, us_analysis_tickers)
            
            if (adv_u + dec_u + flat_u) > 0:
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; margin-bottom: 8px; font-size: 14px;">
                    <span style="color:#3FB950; font-weight:bold;">🟢 Cresc: {adv_u} ({adv_pu:.1f}%)</span>
                    <span style="color:#8B949E; font-weight:bold;">⚪ Neutru: {flat_u}</span>
                    <span style="color:#F85149; font-weight:bold;">🔴 Scad: {dec_u} ({dec_pu:.1f}%)</span>
                </div>
                <div style="display: flex; height: 14px; border-radius: 7px; overflow: hidden; background-color: #30363D; margin-bottom: 5px;">
                    <div style="width: {adv_pu}%; background-color: #3FB950;"></div>
                    <div style="width: {flat_pu}%; background-color: #8B949E;"></div>
                    <div style="width: {dec_pu}%; background-color: #F85149;"></div>
                </div>
                """, unsafe_allow_html=True)
                
                # Verdict Pro Automat (Corelare cu indicii)
                if adv_u > dec_u * 1.5 and sp500_chg > 0:
                    st.caption("✅ **Breakout Confirmat:** Indicele S&P urcă și o face tragând majoritatea acțiunilor cu el.")
                elif dec_u > adv_u * 1.5 and sp500_chg > 0:
                    st.caption("⚠️ **Divergență Periculoasă (Mega-Cap Illusion):** S&P crește doar datorită a 2-3 giganți (ex: Nvidia, Apple). Restul pieței de fapt scade!")
                elif dec_u > adv_u * 1.5:
                    st.caption("🚨 **Bearish Breadth:** Scădere severă, instituțiile vând la nivel generalizat.")
                else:
                    st.caption("🔄 **Rotație Sectorială:** Volumele sunt împărțite în mod egal între câștigători și perdanți.")
            st.write("") # Spațiu
            st.markdown("---")
            
            # =========================================================
            # NOU: NIVELE TEHNICE CRITICE (SUA - S&P 500)
            # =========================================================
            st.markdown("#### 🎯 Nivele Tehnice Critice (Trend Macro S&P 500)")
            with st.spinner("Calculăm mediile mobile S&P 500..."):
                tech_us = get_index_technical_levels('^GSPC')
                
            if tech_us:
                c50_u = "#3FB950" if tech_us['dist50'] > 0 else "#F85149"
                c200_u = "#3FB950" if tech_us['dist200'] > 0 else "#F85149"
                
                if tech_us['dist50'] > 0 and tech_us['dist200'] > 0:
                    regime_us = "🟢 **BULL MARKET:** Trend lung intact. Scăderile spre SMA 50 sunt văzute ca oportunități de cumpărare ('Buy the dip')."
                elif tech_us['dist50'] < 0 and tech_us['dist200'] < 0:
                    regime_us = "🔴 **BEAR MARKET:** Teritoriul urșilor. Orice creștere este considerată o capcană ('Dead cat bounce')."
                else:
                    regime_us = "🟡 **ZONĂ DE CONSOLIDARE:** Luptă critică la nivelul mediilor mobile. Așteaptă confirmarea spargerii."
                
                cross_us = "✨ *Golden Cross activ* (Piață în expansiune lungă)" if tech_us['sma50'] > tech_us['sma200'] else "☠️ *Death Cross activ* (Risc de recesiune a pieței)"

                st.markdown(f"""
                <div style="display:flex; gap:15px; margin-bottom: 10px;">
                    <div style="flex:1; background:#161B22; padding:15px; border-radius:10px; border:1px solid #30363D;">
                        <div style="color:#8B949E; font-size:11px; text-transform:uppercase;">Față de SMA 50 (Mediu)</div>
                        <div style="font-size:24px; font-weight:bold; color:{c50_u};">{tech_us['dist50']:+.2f}%</div>
                        <div style="color:#C9D1D9; font-size:12px; margin-top:5px;">Prag suport/rezistență: <b>{tech_us['sma50']:,.2f}</b></div>
                    </div>
                    <div style="flex:1; background:#161B22; padding:15px; border-radius:10px; border:1px solid #30363D;">
                        <div style="color:#8B949E; font-size:11px; text-transform:uppercase;">Față de SMA 200 (Lung)</div>
                        <div style="font-size:24px; font-weight:bold; color:{c200_u};">{tech_us['dist200']:+.2f}%</div>
                        <div style="color:#C9D1D9; font-size:12px; margin-top:5px;">Prag suport/rezistență: <b>{tech_us['sma200']:,.2f}</b></div>
                    </div>
                </div>
                <div style="background:#21262D; padding:12px; border-radius:8px; font-size:14px; border-left: 4px solid {'#3FB950' if tech_us['dist200'] > 0 else '#F85149'};">
                    {regime_us} <br> <span style="color:#8B949E; font-size:12px;">{cross_us}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Date tehnice indisponibile pentru S&P 500.")
            st.markdown("---")
            
            # =========================================================
            # NOU: INTENSITATE VOLUM (SUA)
            # =========================================================
            st.markdown("#### 🔊 Intensitatea Tranzacționării (S&P 500)")
            vol_us = get_volume_analysis('^GSPC')
            
            if vol_us:
                v_int_u = vol_us['intensity']
                v_color_u = "#3FB950" if v_int_u > 1.2 else ("#D29922" if v_int_u > 0.8 else "#8B949E")
                v_label_u = "CONVINGERE TARE" if v_int_u > 1.2 else ("STABIL" if v_int_u > 0.8 else "FĂRĂ CONVINGERE")
                
                st.markdown(f"""
                <div style="background:#161B22; padding:20px; border-radius:12px; border-top: 4px solid {v_color_u};">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <div style="color:#8B949E; font-size:12px; text-transform:uppercase;">Volume Intensity Index</div>
                            <div style="font-size:28px; font-weight:bold; color:white;">{v_int_u:.2f}x</div>
                        </div>
                        <div style="text-align:right;">
                            <span style="background:{v_color_u}22; color:{v_color_u}; padding:5px 12px; border-radius:20px; font-weight:bold; font-size:12px;">
                                {v_label_u}
                            </span>
                        </div>
                    </div>
                    <div style="margin-top:15px; background:#30363D; height:8px; border-radius:4px; overflow:hidden;">
                        <div style="width:{min(v_int_u*50, 100)}%; background:{v_color_u}; height:100%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if v_int_u > 1.2 and sp500_chg > 0:
                    st.caption("🚀 **Bullish Confirmation:** Creșterea este susținută de volume mari. Trendul are „combustibil” instituțional.")
                elif v_int_u > 1.2 and sp500_chg < 0:
                    st.caption("🚨 **Heavy Distribution:** Se vinde masiv pe volume mari. Pericol de continuare a scăderii.")
                elif v_int_u < 0.7:
                    st.caption("⚠️ **Low Conviction:** Volum sub medie. Mișcarea zilei nu este confirmată de jucătorii mari.")
            st.markdown("---")
            # =========================================================

            c_us1, c_us2 = st.columns(2)
            with c_us1:
                st.markdown("**🚀 Top Creșteri (Big Caps)**")
                if not us_gainers.empty:
                    st.dataframe(
                        us_gainers[['Simbol', 'Preț', 'Variație']].style
                        .format({'Preț': '${:.2f}', 'Variație': '{:+.2f}%'})
                        .map(lambda x: 'color: #3FB950', subset=['Variație']),
                        width='stretch', hide_index=True
                    )
            
            with c_us2:
                st.markdown("**🔻 Top Scăderi (Big Caps)**")
                if not us_losers.empty:
                    st.dataframe(
                        us_losers[['Simbol', 'Preț', 'Variație']].style
                        .format({'Preț': '${:.2f}', 'Variație': '{:+.2f}%'})
                        .map(lambda x: 'color: #F85149', subset=['Variație']),
                        width='stretch', hide_index=True
                    )

            # --- 3. Top Știri Wall Street ---
            st.markdown("---")
            st.subheader("🇺🇸 Top 10 Știri Wall Street")
            
            if 'news_cache_us' not in st.session_state:
                 news_us_gspc = get_company_news_rss("^GSPC")
                 news_us_ixic = get_company_news_rss("^IXIC")
                 combined_us = news_us_gspc + news_us_ixic
                 combined_us.sort(key=lambda x: x['date_str'], reverse=True)
                 st.session_state['news_cache_us'] = combined_us
            
            final_us_news = st.session_state['news_cache_us']
            
            seen_us = set()
            unique_us_news = []
            for n in final_us_news:
                if n['title'] not in seen_us:
                    unique_us_news.append(n)
                    seen_us.add(n['title'])

            if unique_us_news:
                us_news_html = ""
                for item in unique_us_news[:10]:
                      us_news_html += f"""
                      <div style="margin-bottom: 10px; border-bottom: 1px solid #30363D; padding-bottom: 5px;">
                        <a href="{item['link']}" style="color: #58A6FF; text-decoration: none; font-weight: 600;" target="_blank">
                           {item['title']}
                        </a>
                        <div style="font-size: 12px; color: #8B949E;">{item['publisher']} • {item['date_str']}</div>
                      </div>
                      """
                st.markdown(us_news_html, unsafe_allow_html=True)
            else:
                st.info("Nu s-au putut încărca știrile din SUA.")

    # ==================================================
    # 7. SCANNER VOLUM (RVOL) - NOU
    # ==================================================
    elif sectiune == "7. Scanner Volum":
        st.title("📡 Scanner Volum Relativ (RVOL)")
        st.markdown("""
        Acest modul identifică **anomaliile de volum**. 
        Un RVOL (Relative Volume) mai mare de **1.5** indică un interes instituțional sau o știre importantă.
        """)
        
        # Slider pentru sensibilitate (Default 1.5)
        threshold = st.slider("Arată doar acțiunile cu Volum de 'X' ori mai mare decât media:", 
                            min_value=1.2, max_value=5.0, value=1.5, step=0.1)

        # Definim listele de scanare (Extinse)
        tickers_map = {
            "🇷🇴 BVB (România - BET)": [
                'TVBETETF.RO', 'TLV.RO', 'SNP.RO', 'H2O.RO', 'TRP.RO', 'FP.RO', 'ATB.RO', 'BIO.RO', 'ALW.RO', 'AST.RO', 
                'EBS.RO', 'IMP.RO', 'SNG.RO', 'BRD.RO', 'ONE.RO', 'TGN.RO', 'SNN.RO', 'DIGI.RO', 'M.RO', 'EL.RO', 'MILK.RO', 
                'SMTL.RO', 'AROBS.RO', 'AQ.RO', 'ARS.RO', 'ASC.RO', 'BRK.RO', 'IARV.RO', 'TTS.RO', 'WINE.RO', 'TEL.RO', 'DN.RO', 'AG.RO', 
                'BENTO.RO', 'PE.RO', 'COTE.RO', 'PBK.RO', 'SAFE.RO', 'TBK.RO', 'CFH.RO', 'SFG.RO'
            ],
            
            "🇺🇸 SUA - Tech & Growth (Nasdaq 100)": [
                'NVDA', 'MSFT', 'AAPL', 'AMZN', 'META', 'GOOGL', 'TSLA', 'AVGO', 'COST', 'PEP', 'CSCO', 'TMUS',
                'CMCSA', 'INTC', 'AMD', 'CLS', 'NFLX', 'TXN', 'ANET', 'NET', 'SBUX', 'ISRG', 'MDLZ', 'GILD',
                'ARM', 'BKNG', 'PANW', 'MU', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'CRWV', 'CSX', 'PYPL', 'ASML',
                'PLTR', 'CRWD', 'ZS', 'MSTR', 'QCOM', 'SNDK', 'HOOD', 'ROKU', 'INOD', 'U', 'ORCL', 'TSM', 'AFRM'
            ],
            
            "🇺🇸 SUA - Industrial & Finance (Dow/S&P)": [
                'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'AXP', 'V', 'MA', 'BRK-B',
                'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'GPOR', 'VG', 'HAL', 'MPC', 'DVN', 'UUUU', 'OKLO', 'VLO', 'T',
                'CAT', 'DE', 'BA', 'SPCX', 'LMT', 'RTX', 'GD', 'NOC', 'GE', 'MMM', 'HON', 'UNP', 'NVO', 'AMGN', 'BIIB', 'SNY', 'NVS',
                'JNJ', 'LLY', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO', 'MP', 'CMG', 'METC', 'RIO', 'BHP', 'AEM', 'DHR', 'BMY', 'CVS'
            ],
            
            "🇪🇺 Europa - Germania (DAX 40)": [
                'SAP.DE', 'SIE.DE', 'ALV.DE', 'DTE.DE', 'AIR.DE', 'BMW.DE', 'VOW3.DE', 'MBG.DE', 'BAS.DE', 'BAYN.DE',
                'ADS.DE', 'DHL.DE', 'DB1.DE', 'MUV2.DE', 'IFX.DE', 'EOAN.DE', 'RWE.DE', 'ENR.DE', 'DTG.DE', 'BSP.DE', 'RHM.DE', 'HEN3.DE', 'VNA.DE',
                'DBK.DE', 'CBK.DE', 'CON.DE', 'HEI.DE', 'SY1.DE', 'MTX.DE', 'BEI.DE', 'PUM.DE', 'ZAL.DE'
            ],
            
            "🇪🇺 Europa - Franța (CAC 40)": [
                'MC.PA', 'OR.PA', 'TTE.PA', 'SAN.PA', 'AIR.PA', 'SU.PA', 'AI.PA', 'BNP.PA', 'EL.PA', 'KER.PA',
                'RMS.PA', 'SAF.PA', 'CS.PA', 'DG.PA', 'RNO.PA', 'STLAP.PA', 'GLNCY', 'ACA.PA', 'ORA.PA', 'CAP.PA', 'EN.PA',
                'VIV.PA', 'ENG.PA', 'LR.PA', 'ML.PA', 'DGE.L', 'SU.PA', 'HO.PA', 'RI.PA', 'BN.PA', 'DSY.PA'
            ],
            
            "🇬🇧 UK & Others (FTSE/Global)": [
                'SHEL.L', 'AZN', 'HSBA.L', 'ULVR.L', 'BP', 'RIO', 'GSK.L', 'DGE.L', 'REL.L', 'BATS.L',
                'GLNCY', 'LSEG.L', 'AAL.L', 'BARC.L', 'LLOY.L', 'BA.L', 'LDO.MI', 'NWG.L', 'VOD.L', 'RR.L', 'TSCO.L',
                'ASML', 'NVO', 'SONY', 'TSM', 'BABA', 'JD', 'BIDU', 'TCEHY'
            ]
        }
        
        # Funcție internă de calcul RVOL + AI Isolation Forest
        def get_rvol_data(ticker_list):
            from ai_engine import detect_volume_anomaly_ai # Importăm funcția nouă AI
            try:
                # Descărcăm date pe 3 luni pentru a avea destul istoric de "învățare" pt ML
                data = yf.download(ticker_list, period="3mo", group_by='ticker', progress=False)
                results = []
                
                for t in ticker_list:
                    try:
                        # Gestionare MultiIndex vs Single Index
                        if isinstance(data.columns, pd.MultiIndex):
                            if t not in data.columns.levels[0]: continue
                            df_t = data[t]
                        else:
                            df_t = data
                        
                        vol = df_t['Volume'].dropna()
                        close = df_t['Close'].dropna()
                        
                        if len(vol) < 25: continue 
                        
                        # Calcule clasice matematice
                        curr_vol = vol.iloc[-1]
                        avg_vol_20 = vol.iloc[-21:-1].mean()
                        if avg_vol_20 < 5000: continue # Ignorăm acțiunile nelichide
                        
                        rvol = curr_vol / avg_vol_20
                        
                        curr_p = close.iloc[-1]
                        prev_p = close.iloc[-2]
                        change_pct = ((curr_p - prev_p) / prev_p) * 100
                        
                        # --- MAGIA AI: Interogăm modelul Isolation Forest ---
                        is_anomaly = detect_volume_anomaly_ai(df_t)
                        
                        results.append({
                            "Simbol": t.replace('.RO', ''),
                            "Preț": curr_p,
                            "Variație %": change_pct,
                            "Volum Azi": curr_vol,
                            "Volum Mediu (20z)": avg_vol_20,
                            "RVOL": rvol,
                            "Alertă AI": "🚨 ATENTIE" if is_anomaly else "-",  # <--- NOUA COLOANĂ AI
                            "Status": "🚀 BREAKOUT" if (rvol > 2.0 and change_pct > 1.5) 
                                     else ("⚠️ PANIC SELL" if (rvol > 2.0 and change_pct < -1.5) 
                                     else ("✅ ACUMULARE" if (rvol > 1.2 and change_pct > 0) else "Normal"))
                        })
                    except: continue
                    
                return pd.DataFrame(results)
            except: return pd.DataFrame()
                        

        # --- SELECTOR DE PIAȚĂ (DROPDOWN în loc de TABURI pentru eficiență) ---
        market_choice = st.selectbox("Alege Piața/Sectorul de scanat:", list(tickers_map.keys()))
        
        # Extragem tickerii pentru selecția făcută
        selected_tickers = tickers_map[market_choice]
        
        col_scan_btn, col_info = st.columns([1, 3])
        
        with col_scan_btn:
            run_scan = st.button(f"🔎 Scanează {len(selected_tickers)} companii", type="primary")
            
        with col_info:
            st.caption(f"Se vor analiza volumele pentru: {', '.join(selected_tickers[:5])} ... și altele.")

        if run_scan:
            with st.spinner(f"Analizăm {market_choice}... (Poate dura 10-20 secunde)"):
                df_res = get_rvol_data(selected_tickers)
                
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
                                "Preț": "{:.2f}",
                                "Variație %": "{:+.2f}%",
                                "Volum Azi": "{:,.0f}",
                                "Volum Mediu (20z)": "{:,.0f}",
                                "RVOL": "{:.2f}x"
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
                        data_bulk = yf.download(tickers_list, period="5d", progress=False)['Close']
                        
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
                            else:
                                current_prices[col] = 0.0
                        
                        # Convertim înapoi în Series pentru restul codului
                        live_data = pd.Series(current_prices)
                        # --------------------------------------------------------

                    except Exception as e:
                        # st.error(f"Eroare date: {e}") 
                        live_data = pd.Series()

            # 2. Construim tabelul final
            display_rows = []
            for index, row in df_wl.iterrows():
                sym = row['Symbol']
                target = smart_to_float(row['TargetPrice'])  # acceptă și formatul românesc ("25,5")
                note = row['Notes']
                
                # Extragem prețul curent din seria curățată
                try:
                    curr = float(live_data.get(sym, 0))
                except:
                    curr = 0

                # Calculăm distanța până la țintă
                if curr > 0:
                    dist_pct = ((curr - target) / curr) * 100
                    is_buy = curr <= target # E sub prețul țintă?
                else:
                    dist_pct = 0
                    is_buy = False
                
                display_rows.append({
                    "Simbol": sym,
                    "Preț Curent": curr,
                    "Preț Țintă 🎯": target,
                    "Distanță (%)": dist_pct,
                    "Recomandare": "✅ CUMPĂRĂ" if is_buy else "⏳ Așteaptă",
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
                .format({"Preț Curent": "{:.2f}", "Preț Țintă 🎯": "{:.2f}", "Distanță (%)": "{:.2f}%"}),
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

