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
# --- IMPORTURI NOI PENTRU GOOGLE SHEETS ---
import gspread
from google.oauth2.service_account import Credentials
import concurrent.futures # Pune acest import la începutul fișierului, sus de tot
from scipy.stats import norm

def fetch_portfolio_prices_parallel(tickers):
    """Descarcă prețurile pentru tot portofoliul simultan."""
    def get_price(ticker):
        try:
            t = yf.Ticker(ticker)
            # Folosim fast_info pentru că este mult mai rapidă decât .info
            return ticker, t.fast_info.last_price
        except:
            return ticker, 0.0

    prices = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(get_price, t): t for t in tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker, price = future.result()
            prices[ticker] = price
    return prices

# --- 0. CONFIGURARE GLOBALĂ ---
st.set_page_config(page_title="Terminal Investiții PRO", page_icon="📈", layout="wide")
socket.setdefaulttimeout(15) # Mărit timeout-ul pentru conexiuni lente

# --- CONFIGURARE CONEXIUNE GOOGLE SHEETS ---
def connect_to_gsheets():
    """Conectare securizată la Google Sheets folosind Secrets."""
    scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    try:
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
            creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
            client = gspread.authorize(creds)
            # Deschidem fișierul 'portofoliu_db' din Drive-ul tău
            sheet = client.open("portofoliu_db").sheet1
            return sheet
        else:
            st.error("⚠️ Nu s-au găsit credențialele în Secrets! Verifică setările din Streamlit Cloud.")
            return None
    except Exception as e:
        st.error(f"Eroare conectare Google Sheets: {e}")
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
        "https://asia.nikkei.com/rss/feed/nar", # Nikkei Asia (Excelent pt Japonia) 
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC,EURUSD=X,GC=F,CL=F&region=US&lang=en-US", 
        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
        "http://feeds.marketwatch.com/marketwatch/topstories",
        "https://www.investing.com/rss/news.rss"    
    ],
    "Categorii": {
        "General": [], 
        "Tehnologie": ["tehnologie", "tech", "it", "ai", "software", "hardware", "digital", "cyber", "apple", "microsoft", "google", "nvidia", "oracle", "amazon", "adobe", "asml", "tsm", "palantir", "qualcomm", "micron", "amd", "meta", "broadcom", "intel", "innodata", "crypto", "blockchain", "semiconductori", "startup"],
        "Energie": ["energie", "petrol", "gaze", "oil", "wti", "energy", "curent", "hidroelectrica", "omv", "romgaz", "nuclearelectrica", "electrica", "simtel", "transelectrica", "transgaz", "regenerabil", "eolian", "solar", "fotovoltaic", "exxon", "chevron", "devon", "lng", "oklo", "shell", "vistra", "nuscale"],
        "Financiar": ["banca", "bank", "credit", "bursa", "finante", "fonduri", "asigurari", "bvb", "fiscal", "profit", "taxe", "buget", "wall street", "jpm", "unicredit", "ubs", "goldman", "dobanda", "monetar", "Banca Transilvania", "BRD", "BAC", "WFC", "AXP", "JP", "Visa", "BNP", "GS", "Mastercard", "investitii"],
        "Farma": ["farma", "pharma", "sanatate", "medicament", "spital", "medical", "pfizer", "nvo", "sanofi", "eli lilly", "novartis", "biogen", "medicover", "medlife", "regina maria", "BIO", "Antibiotice", "biotech"],
        "Militar": ["militar", "aparare", "defense", "armata", "razboi", "nato", "arme", "securitate", "geopolitic", "taiwan", "ucraina", "rusia", "lmt", "raytheon", "bae", "Leonardo", "Boeing", "rheinmetall", "Thales", "Vinci", "Red Cat", "drone"],
        "Imobiliare": ["imobiliare", "real estate", "apartament", "garsoniera", "casa", "vila", "locuinta", "teren", "birou", "birouri", "santier", "dezvoltator", "rezidential", "chirie", "chirii", "ipotecar", "reit", "mall", "spatii comerciale", "impact", "one united"],
        "Auto": ["auto", "masini", "ev", "electric", "dacia", "ford", "tesla", "volkswagen", "bmw", "mercedes", "byd", "xpeng", "nio", "toyota", "audi", "ferrari", "inmatriculari", "autostrada"],
        "Asia": ["asia", "china", "japonia", "tokyo", "beijing", "shanghai", "hong kong", "taiwan", "india", "seul", "coreea", "nikkei", "yen", "yuan", "rupee", "boj", "evergrande", "alibaba", "tencent", "tsmc", "nifty", "hang seng"],
        "Aur/Metale": ["aur", "gold", "argint", "silver", "metal", "cupru", "precious", "aluminiu", "otel", "minereu", "rio tinto", "bhp", "METC", "glencore", "mp materials"],
        "Macro/Joburi": ["inflatie", "cpi", "pce", "fed", "bce", "robor", "ircc", "bnr", "somaj", "jobs", "angajari", "salarii", "pib", "gdp", "pmi", "recesiune", "dobanzi", "economie"]
    }
}

# --- FUNCȚII UTILITARE ---
def parse_date(entry):
    try:
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            return datetime.fromtimestamp(time.mktime(entry.published_parsed))
        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
            return datetime.fromtimestamp(time.mktime(entry.updated_parsed))
        elif hasattr(entry, 'published'):
            return datetime.now()
    except: pass
    return datetime.now()

# --- FUNCȚIE NOUĂ DE PARSARE INTELIGENTĂ (SENIOR FIX) ---
def smart_to_float(val):
    """Transformă orice număr (format US sau EU) în float curat."""
    if pd.isna(val) or val == '': return 0.0
    s = str(val).strip()
    # Păstrăm doar cifre, punct, virgulă și minus
    s = re.sub(r'[^\d.,-]', '', s)
    if not s: return 0.0

    # Logică de detecție a formatului
    if ',' in s and '.' in s:
        if s.rfind(',') > s.rfind('.'): # Format EU: 1.000,50
            s = s.replace('.', '').replace(',', '.')
        else: # Format US: 1,000.50
            s = s.replace(',', '')
    elif ',' in s:
        if s.count(',') > 1: # US Thousands: 1,000,000
            s = s.replace(',', '')
        else: # RO Decimal: 50,5
            s = s.replace(',', '.')
    elif '.' in s:
        if s.count('.') > 1: # RO Thousands: 1.000.000
            s = s.replace('.', '')
        # Altfel e US Decimal: 50.5
            
    try:
        return float(s)
    except ValueError:
        return 0.0
    
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
        
    if is_pct: return f"{val * 100:.2f}%"
    if val >= 1e12: return f"{val/1e12:.2f} T"
    if val >= 1e9: return f"{val/1e9:.2f} B"
    if val >= 1e6: return f"{val/1e6:.2f} M"
    return f"{val:,.2f}"

def get_sentiment(text):
    blob = TextBlob(text)
    pol = blob.sentiment.polarity
    if pol > 0.05: return "Pozitiv", "impact-poz", "↗"
    elif pol < -0.05: return "Negativ", "impact-neg", "↘"
    else: return "Neutru", "impact-neu", "→"

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

def calculate_investment_rating_pro(info, inst_pct, rvol, spread_val, mos_val):
    score = 50
    details = []
    
    # 1. ANALIZA SMART MONEY
    if inst_pct > 70:
        score += 15
        details.append("✅ **Smart Money:** Deținere de elită (>70%). Suport instituțional masiv.")
    elif inst_pct > 50:
        score += 10
        details.append("✅ **Smart Money:** Majoritate instituțională. Stabilitate ridicată.")
    elif inst_pct < 20:
        score -= 15
        details.append("⚠️ **Smart Money:** Deținere instituțională slabă. Risc de volatilitate retail.")

    # 2. ANALIZA EVALUARE
    if mos_val > 25:
        score += 15
        details.append(f"✅ **Evaluare:** Marjă de siguranță excelentă ({mos_val:.1f}%). Preț subevaluat.")
    elif mos_val < -10:
        score -= 15
        details.append(f"🚨 **Evaluare:** Supraevaluare semnificativă. Risc ridicat de corecție.")

    # 3. ANALIZA MACRO
    if spread_val < 0:
        score -= 20
        details.append("🚨 **Macro:** Curbă 10Y-2Y inversată. Risc sistemic de recesiune detectat.")
    else:
        score += 5
        details.append("✅ **Macro:** Mediul economic este favorabil expansiunii.")

    # 4. SĂNĂTATE FINANCIARĂ
    roe = info.get('returnOnEquity', 0)
    if roe > 0.20:
        score += 10
        details.append(f"🚀 **Eficiență:** ROE excepțional ({roe*100:.1f}%).")
    
    debt = info.get('debtToEquity', 0)
    if debt > 150:
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
                return smart_to_float(match.iloc[0]['TargetPrice'])
    except:
        pass
    return None

def get_peers_analysis(sector, industry, current_ticker):
    """Extrage competitori și include datoria pentru o analiză de risc."""
    peers_map = {
        "Technology": ["MSFT", "GOOGL", "NVDA", "AAPL", "AMD", "AVGO", "MU", "META", "TSM", "QCOM"],
        "Financial Services": ["JPM", "BAC", "GS", "WFC", "C", "V", "MS", "MA", "AXP", "SCHW"],
        "Energy": ["XOM", "CVX", "LNG", "OXY", "COP", "OXY", "DVN", "FSLR", "VST", "UUUU", "LEU", "CEG"],
        "Healthcare": ["LLY", "JNJ", "NVO", "NVS", "PFE", "SNY", "MRK"],
        "Industrials": ["LMT", "RTX", "NOC", "BA", "GD", "MMM", "CAT", "DAL", "UAL"],
        "Basic Materials": ["RIO", "VALE", "BHP", "FCX", "NEM", "AEM", "GLNCY", "USAR", "AREC", "MP", "METC", "LAC"],
        "Consumer Defensive": ["WMT", "KO", "CL", "KHC", "PG", "SFD", "PEP", "PM"], 
        "Consumer Cyclical": ["MCD", "CMG", "SBUX", "DPZ", "NKE", "RCL", "GM", "F"]
    }
    
    potential_peers = peers_map.get(sector, ["SPY", "QQQ", "DIA"])
    peers = [p for p in potential_peers if p != current_ticker][:15]
    
    peer_results = []
    for p_sym in peers:
        try:
            t = yf.Ticker(p_sym)
            inf = t.info
            peer_results.append({
                "Simbol": p_sym,
                "P/E": inf.get('trailingPE', 0),
                "ROE (%)": inf.get('returnOnEquity', 0) * 100,
                "ROA (%)": inf.get('returnOnAssets', 0) * 100,
                "Marjă Netă (%)": inf.get('profitMargins', 0) * 100,
                "Datorii/Eq (%)": inf.get('debtToEquity', 0)
            })
        except: continue
    return pd.DataFrame(peer_results)

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

def calculate_dcf_dynamic(info, growth_rate_input, discount_rate_input):
    """Calculează DCF folosind estimările tale manuale."""
    try:
        eps = info.get('trailingEps')
        if not eps or eps <= 0: return 0
        
        # Parametrii tăi din interfață
        growth_rate = growth_rate_input / 100
        discount_rate = discount_rate_input / 100
        terminal_multiple = min(info.get('trailingPE', 15), 25) 
        
        # Proiecție pe 5 ani
        cash_flows = []
        for i in range(1, 6):
            fcf = eps * ((1 + growth_rate) ** i)
            discounted_fcf = fcf / ((1 + discount_rate) ** i)
            cash_flows.append(discounted_fcf)
            
        # Valoare Terminală la finalul anului 5
        terminal_val = (eps * ((1 + growth_rate) ** 5)) * terminal_multiple
        discounted_terminal = terminal_val / ((1 + discount_rate) ** 5)
        
        return sum(cash_flows) + discounted_terminal
    except:
        return 0
# --- Pune acest bloc sus, lângă celelalte funcții (calculate_alpha, etc.) ---

def calculate_health_score_ext(info):
    score = 5
    pros = []
    cons = []
    
    try:
        # 1. Analiză Datorii
        de = info.get('debtToEquity', 0)
        if de:
            if de < 50: 
                score += 2
                pros.append("Datorii foarte mici")
            elif de > 150: 
                score -= 2
                cons.append("Îndatorare ridicată")
            elif de > 300: 
                score -= 3
                cons.append("Risc mare de insolvență")

        # 2. Analiză Rentabilitate (ROE)
        roe = info.get('returnOnEquity', 0)
        if roe > 0.15: 
            score += 2
            pros.append("Profitabilitate excelentă (ROE)")
        elif roe < 0.05: 
            score -= 1
            cons.append("Eficiență scăzută a capitalului")

        # 3. Analiză Lichiditate
        cr = info.get('currentRatio', 1)
        if cr > 1.5: 
            score += 1
            pros.append("Lichiditate solidă")
        elif cr < 1:
            score -= 1
            cons.append("Lichiditate precară")
            
    except: pass
    
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
    sector = info.get('sector', 'Unknown')
    limits = get_sector_benchmarks(sector)
    
    # Extragere sigură date (evităm erorile dacă lipsesc indicatori)
    pe = info.get('trailingPE') or 0
    roe = info.get('returnOnEquity') or 0
    de = info.get('debtToEquity') or 0
    cr = info.get('currentRatio') or 0
    
    # Protecție anti-crash pentru NVS (Transformăm None în 0 sau 1)
    safe_alpha = alpha if alpha is not None else 0
    safe_beta = beta if beta is not None else 1.0 # Beta 1.0 înseamnă risc neutru
    
    audit = []

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
    if roe > 0.25:
        audit.append(f"🚀 **PROFITABILITATE:** Eficiență de elită (ROE {roe*100:.1f}%). Management performant.")
    elif roe < 0.10:
        audit.append(f"📉 **PROFITABILITATE:** Eficiență sub-optimă ({roe*100:.1f}%). Capitalul nu produce suficient.")

    # --- 3. SOLVABILITATE (DATORII) ---
    if de > limits['de_max']:
        audit.append(f"🚩 **SOLVABILITATE:** Îndatorare peste limita sectorului ({de:.1f}%). Risc structural ridicat.")
    else:
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

    # --- 6. RISC DE PIAȚĂ (BETA - Reparat pentru NVS) ---
    if safe_beta > 1.3:
        audit.append(f"🎢 **VOLATILITATE (Beta {safe_beta:.2f}):** Risc ridicat. Mișcări mult mai ample decât piața.")
    elif safe_beta < 0.8:
        audit.append(f"🛡️ **VOLATILITATE (Beta {safe_beta:.2f}):** Profil defensiv. Stabilă în perioade de criză.")
    else:
        audit.append(f"⚖️ **VOLATILITATE (Beta {safe_beta:.2f}):** Mișcare sincronizată cu piața generală.")

    return audit

def calculate_altman_z(info):
    """Calcul Altman Z-Score recalibrat pentru a preveni erorile de miliarde."""
    try:
        total_assets = info.get('totalAssets') or info.get('totalAssetsNetModularEquity') or 1
        working_cap = (info.get('totalCurrentAssets', 0) or 0) - (info.get('totalCurrentLiabilities', 0) or 0)
        A = (working_cap / total_assets) * 1.2
        B = ((info.get('retainedEarnings', 0) or 0) / total_assets) * 1.4
        ebit = info.get('ebit', 0) or info.get('operatingIncome', 0) or 0
        C = (ebit / total_assets) * 3.3
        m_cap = info.get('marketCap', 0) or 0
        t_liab = info.get('totalLiabilitiesNetModularEquity') or info.get('totalLiabilities') or 1
        
        raw_ratio = m_cap / t_liab
        D = (raw_ratio / 1000 if raw_ratio > 100 else raw_ratio) * 0.6
        E = ((info.get('totalRevenue', 0) or 0) / total_assets) * 1.0
        
        z_score = A + B + C + D + E
        final_score = min(z_score, 15.0)

        status, color = ("Safe Zone", "#3FB950") if final_score > 2.99 else \
                        (("Grey Zone", "#D29922") if final_score >= 1.81 else ("Distress", "#F85149"))
        return final_score, status, color, "Probabilitate de faliment neglijabilă."
    except:
        return 0.0, "Eroare", "#8B949E", "Date incomplete."

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
    """Analizează dacă dividendele și profitul sunt sustenabile."""
    payout = info.get('payoutRatio', 0)
    net_income = info.get('netIncomeToCommon', 1)
    cash_flow = info.get('operatingCashflow', 0)
    
    # Calculăm Calitatea Profitului (Cash Flow / Net Income)
    # Un raport sub 0.8 indică profituri 'pe hârtie', nu în cash.
    quality_ratio = cash_flow / net_income if net_income != 0 else 0
    
    verdicts = []
    
    # Analiză Payout
    if payout > 0.80:
        verdicts.append("🚨 **Dividend Periculos:** Firma distribuie peste 80% din profit. Riscul de tăiere a dividendului este imens.")
    elif 0.30 < payout <= 0.60:
        verdicts.append("✅ **Dividend Sustenabil:** Distribuție echilibrată, lăsând loc și pentru reinvestiții.")
        
    # Analiză Calitate Profit
    if quality_ratio < 0.7:
        verdicts.append("⚠️ **Calitate Slabă a Profitului:** Firma raportează profit, dar nu încasează suficient cash. Atenție la contabilitate!")
    elif quality_ratio > 1.2:
        verdicts.append("💎 **Profit de Înaltă Calitate:** Cash-flow-ul depășește profitul net. Semn de business ultra-sănătos.")
        
    return verdicts, quality_ratio, payout * 100
        
# --- FUNCȚIE GET STOCK DATA (FINAL - SMART MODE) ---
@st.cache_data(ttl=900)
def get_stock_data(symbol):
    try:
        # FĂRĂ requests.Session, FĂRĂ curl_cffi forțat.
        # yfinance va alege singur cea mai bună metodă.

        t = yf.Ticker(symbol)
        hist = t.history(period="5y")

        # Fallback BVB
        if hist.empty and not symbol.endswith(".RO"):
            sym_ro = symbol + ".RO"
            t_ro = yf.Ticker(sym_ro)
            hist_ro = t_ro.history(period="5y")
            if not hist_ro.empty:
                return hist_ro, t_ro.info, getattr(t_ro, 'earnings_history', None), sym_ro

        if hist.empty:
            return None, None, None, symbol

        return hist, t.info, getattr(t, 'earnings_history', None), symbol

    except Exception as e:
        print(f"Eroare: {e}")
        return None, None, None, symbol

def calculate_technical_indicators(df):
    if df is None or df.empty: return df
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9).mean()
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

        st.plotly_chart(fig, use_container_width=True)
        
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
        
        st.plotly_chart(fig, use_container_width=True)

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
        'WMT', 'KO', 'PEP', 'PG', 'DXCM', 'COP', 'OXY', 'DVN', 'LNG', 'UUUU', 'FSLR', 'TTE', 'RIO', 'BHP', 'D', 'VALE', 'METC', 'MP', 'LLY', 'AMGN', 'XOM', 'CVX', 
        'PLTR', 'PANW', 'ANET', 'QCOM', 'ORCL', 'TSM', 'GS', 'CRM', 'WFC', 'NVO', 'NVS', 'MCD', 'SMR', 'CMG', 'OKLO', 'SNY', 'JNJ', 'BA', 'GD', 'RTX', 'LMT', 'KTOS', 'PM', 'COO', 'MRK', 'PFE', 'C'
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
    sheet = connect_to_gsheets()
    if sheet:
        try:
            # Luăm toate înregistrările
            data = sheet.get_all_records()
            return pd.DataFrame(data)
        except:
            # Dacă foaia e goală sau apare o eroare de citire
            return pd.DataFrame()
    return pd.DataFrame() # Fallback
# --- FUNCȚII WATCHLIST (CORECȚIE BUG) ---
def load_watchlist():
    """Citește datele din foaia 'watchlist'."""
    sheet = connect_to_gsheets() # Returnează Sheet1
    if sheet:
        try:
            # FIX: Accesăm fișierul părinte (spreadsheet) direct, apoi foaia 'watchlist'
            ws = sheet.spreadsheet.worksheet("watchlist")
            data = ws.get_all_records()
            return pd.DataFrame(data)
        except Exception as e:
            # Dacă foaia nu există sau e goală
            return pd.DataFrame()
    return pd.DataFrame()

def add_to_watchlist(symbol, target, note):
    """Adaugă o intrare nouă în watchlist."""
    sheet = connect_to_gsheets()
    if sheet:
        try:
            # FIX: Folosim 'spreadsheet' pentru a schimba tab-ul
            ws = sheet.spreadsheet.worksheet("watchlist")
            ws.append_row([symbol, float(target), note])
            st.cache_data.clear() # Resetăm cache-ul
            return True
        except Exception as e:
            st.error(f"Eroare salvare: {e}")
            return False
    return False

def remove_from_watchlist(symbol):
    """Șterge un simbol din watchlist (căutând după nume)."""
    sheet = connect_to_gsheets()
    if sheet:
        try:
            ws = sheet.spreadsheet.worksheet("watchlist")
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
        # Ordinea coloanelor trebuie să corespundă cu header-ul din Sheets: 
        # Symbol, Date, Quantity, AvgPrice, Currency
        # Le convertim explicit pentru a evita erori de serializare JSON
        row = [s, str(d), float(q), float(p), c]
        sheet.append_row(row)
        # Invalidăm cache-ul local pentru ca datele noi să apară instant la refresh
        st.cache_data.clear()

@st.cache_data(ttl=300)
def get_portfolio_history_data(tickers):
    if not tickers: return pd.DataFrame()
    data = yf.download(tickers, period="5y", group_by='ticker')
    return data

def calculate_portfolio_performance(df, history_range="1A"):
    if df.empty: return pd.DataFrame(), pd.DataFrame(), 0, 0
    
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)
    df['AvgPrice'] = pd.to_numeric(df['AvgPrice'], errors='coerce').fillna(0)
    
    tickers = df['Symbol'].unique().tolist()
    
    # --- SCHIMBARE MAJORĂ: Descărcăm totul dintr-o singură lovitură ---
    with st.spinner("Actualizăm prețurile în timp real..."):
        current_prices = fetch_portfolio_prices_parallel(tickers)
        # Descărcăm istoricul bulk pentru grafic
        hist_data = yf.download(tickers, period="5y", group_by='ticker', progress=False)

    current_vals = []
    total_daily_pl_abs = 0 
    
    for _, row in df.iterrows():
        sym = row['Symbol']
        qty = row['Quantity']
        avg_p = row['AvgPrice']
        
        curr_p = current_prices.get(sym, 0)
        
        # Calculăm prețul de ieri din hist_data pentru evoluția zilnică
        try:
            if len(tickers) > 1:
                prev_p = hist_data[sym]['Close'].dropna().iloc[-2]
            else:
                prev_p = hist_data['Close'].dropna().iloc[-2]
        except:
            prev_p = curr_p

        mkt_val = qty * curr_p
        inv_val = qty * avg_p
        profit = mkt_val - inv_val
        profit_pct = (profit / inv_val * 100) if inv_val != 0 else 0
        
        total_daily_pl_abs += (curr_p - prev_p) * qty
        
        current_vals.append({
            'Symbol': sym, 'Quantity': qty, 'AvgPrice': avg_p, 'CurrentPrice': curr_p,
            'MarketValue': mkt_val, 'Profit': profit, 'Profit %': profit_pct
        })
    
    df_result = pd.DataFrame(current_vals)
    
    # Generăm curba portofoliului (Corecție aliniere fus orar)
    portfolio_curve = pd.Series(dtype=float)
    for _, row in df.iterrows():
        sym = row['Symbol']
        qty = row['Quantity']
        try:
            # Preluăm prețurile din bulk-ul descărcat anterior
            prices = hist_data[sym]['Close'] if len(tickers) > 1 else hist_data['Close']
            # .ffill() umple zilele libere (sărbători locale) cu ultimul preț
            term = prices.ffill().bfill() * qty
            if portfolio_curve.empty: 
                portfolio_curve = term
            else: 
                # .add aliniază automat indicii de tip Datetime
                portfolio_curve = portfolio_curve.add(term, fill_value=0)
        except: pass

    days_map = {"1Z": 2, "1S": 7, "1L": 30, "3L": 90, "6L": 180, "1A": 365, "3A": 1095, "5A": 1825}
    portfolio_curve = portfolio_curve.iloc[-days_map.get(history_range, 365):]
    total_val_now = portfolio_curve.iloc[-1] if not portfolio_curve.empty else 0
    total_daily_pl_pct = (total_daily_pl_abs / (total_val_now - total_daily_pl_abs) * 100) if (total_val_now - total_daily_pl_abs) != 0 else 0
    
    return df_result, portfolio_curve, total_daily_pl_abs, total_daily_pl_pct

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
    """Calculează distribuția sectorială în procente și valoare."""
    if df_current.empty: return pd.DataFrame()
    
    tickers_list = df_current['Symbol'].unique().tolist()
    bulk_data = yf.Tickers(" ".join(tickers_list))
    sector_map = {}
    total_mkt_val = df_current['MarketValue'].sum()
    
    for sym in tickers_list:
        try:
            sec = bulk_data.tickers[sym].info.get('sector', 'Nedefinit')
            val = df_current[df_current['Symbol'] == sym]['MarketValue'].sum()
            sector_map[sec] = sector_map.get(sec, 0) + val
        except:
            val = df_current[df_current['Symbol'] == sym]['MarketValue'].sum()
            sector_map['Nedefinit'] = sector_map.get('Nedefinit', 0) + val
            
    df_sec = pd.DataFrame(list(sector_map.items()), columns=['Sector', 'MarketValue'])
    # Adăugăm coloana de procentaj
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
                 'AMD', 'INTC', 'NFLX', 'JPM', 'BAC', 'SOFI', 'MS', 'HON', 'V', 'INOD', 'MA', 'MDB', 'AIG', 'AXP', 'SCHW', 'NET', 'BIIB', 
                 'WMT', 'KO', 'PEP', 'PG', 'DXCM', 'COP', 'OXY', 'DVN', 'LNG', 'T', 'UUUU', 'FSLR', 'TTE', 'RIO', 'BHP', 'D', 'VALE', 'METC', 'MP', 'LLY', 'AMGN', 'XOM', 'CVX', 
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
            # Definim variabilele globale de diagnostic la început pentru a fi disponibile peste tot
            try:
                # Preluăm spread-ul 10Y-2Y pentru rating
                t_10y = yf.Ticker("^TNX").fast_info.last_price
                t_2y = yf.Ticker("2Y=F").fast_info.last_price
                spread = t_10y - t_2y
            except:
                spread = 0.5 # Fallback neutru dacă API-ul eșuează
            
        if hist is None or hist.empty:
            st.error("Simbol invalid sau date indisponibile.")
        else:
            # 1. Informații Generaley
            st.markdown(f"## {info.get('longName', real_sym)}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Sector", info.get('sector', 'N/A'))
            c2.metric("Industrie", info.get('industry', 'N/A'))
            c3.metric("Capitalizare", format_num(info.get('marketCap')))             
         
        # --- 1. DEFINIREA PREȚULUI (VITAL PENTRU CALCULE) ---
            # Luăm ultimul preț disponibil din istoricul deja descărcat
            curr_price = hist['Close'].iloc[-1] if not hist.empty else 0

            # --- 2. EXTRAGEREA ȚINTEI DIN WATCHLIST ---
            target_p = get_watchlist_target(real_sym)
            
            st.markdown("---") # Separator vizual între info generale și țintă

            # --- 3. AFIȘARE CARD DINAMIC ȚINTĂ ---
            if target_p and curr_price > 0:
                # Calculăm distanța procentuală (Cât de mult trebuie să mai scadă)
                # Dacă prețul e sub țintă, rezultatul va fi negativ
                dist_pct = ((curr_price - target_p) / target_p) * 100
                is_hit = curr_price <= target_p
                
                # Culoare dinamică: Verde dacă e sub țintă, Galben dacă e aproape (sub 5%)
                if is_hit:
                    t_color = "#3FB950"  # Verde (Zona de cumpărare)
                    t_status = "🚀 ZONĂ ACHIZIȚIE (Țintă Atinsă!)"
                elif dist_pct < 5:
                    t_color = "#D29922"  # Galben (Aproape de țintă)
                    t_status = f"⚠️ ATENȚIE: Doar {dist_pct:.1f}% peste țintă"
                else:
                    t_color = "#8B949E"  # Gri (Încă scump)
                    t_status = f"⏳ +{dist_pct:.1f}% peste țintă"
                
                st.markdown(f"""
                    <div style="background:#161B22; padding:25px; border-radius:15px; border-left: 10px solid {t_color}; margin-bottom:20px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <p style="color:#8B949E; margin:0; font-size:12px; text-transform:uppercase; letter-spacing:1px;">Ținta ta de intrare</p>
                                <h2 style="color:white; margin:5px 0;">{target_p:.2f} <span style="font-size:16px; color:#8B949E;">{info.get('currency', 'USD')}</span></h2>
                            </div>
                            <div style="text-align:right;">
                                <p style="color:#8B949E; margin:0; font-size:12px; text-transform:uppercase; letter-spacing:1px;">Preț Live</p>
                                <h2 style="color:{t_color}; margin:5px 0;">{curr_price:.2f}</h2>
                            </div>
                        </div>
                        <div style="background:{t_color}22; color:{t_color}; padding:8px; border-radius:8px; font-weight:bold; font-size:16px; text-align:center; margin-top:10px; border: 1px solid {t_color}44;">
                            {t_status}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                # Afișare în caz că nu ai setat o țintă pentru acest simbol
                st.markdown(f"""
                    <div style="background:#161B22; padding:20px; border-radius:15px; border:1px solid #30363D; text-align:center; margin-bottom:20px; opacity:0.7;">
                        <p style="color:#8B949E; margin:0; font-size:11px; text-transform:uppercase;">Strategie Watchlist</p>
                        <h3 style="color:#8B949E; margin:10px 0;">Fără Țintă Stabilită</h3>
                        <p style="font-size:13px; color:#58A6FF;">Adaugă o alertă în capitolul Watchlist pentru a monitoriza {real_sym}.</p>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown("---") # Separator vizual între info generale și țintă    
                        
            # 2. Grafic Tehnic (Păstrat exact cum era în original)
            hist = calculate_technical_indicators(hist)
            st.subheader("📉 Grafic Tehnic")
            col_sel, col_price_info = st.columns([1, 4])
            with col_sel:
                time_opt = st.selectbox("Interval", ["1 Lună", "3 Luni", "6 Luni", "1 An", "3 Ani", "5 Ani"], index=3)
            
            days_map = {"1 Lună": 30, "3 Luni": 90, "6 Luni": 180, "1 An": 365, "3 Ani": 1095, "5 Ani": 1825}
            subset = hist.iloc[-days_map[time_opt]:]
            
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
            st.plotly_chart(fig, use_container_width=True)

            # --- SEPARATORUL SOLICITAT (ADAUGĂ ACEASTĂ LINIE) ---
            st.markdown("---")

            # 3. Indicatori Fundamentali (Cele 4 coloane originale)
            st.subheader("📊 Indicatori Fundamentali")
            beta_val = info.get('beta')
            alpha_val = calculate_alpha(hist, beta_val)
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
                    st.metric("Current Ratio", info.get('currentRatio', 'N/A'))
                    st.metric("Quick Ratio", info.get('quickRatio', 'N/A'))
                with c_risc:
                    st.markdown("**Risc (Alpha & Beta)**")
                    st.metric("Beta", info.get('beta', 'N/A'))
                    st.metric("Alpha (1Y)", format_num(alpha_val, True))
            
            # ==================================================
            # MODUL PRO: PEER ANALYSIS (CARDURI SUS + TABEL JOS)
            # ==================================================
            st.markdown("---")
            st.subheader("🏁 Peer Review: Poziționarea față de Liderii de Sector")
            
            # Datele firmei curente
            my_pe = info.get('trailingPE', 0) or 0
            my_roe = (info.get('returnOnEquity', 0) or 0) * 100
            my_roa = (info.get('returnOnAssets', 0) or 0) * 100
            my_margin = (info.get('profitMargins', 0) or 0) * 100
            
            # --- PASUL 1: CARDURILE DE STATUS (SUS) ---
            c_p1, c_p2, c_p3, c_p4 = st.columns(4)
            
            with c_p1:
                st.metric("P/E vs Sector", f"{my_pe:.1f}", 
                          f"{'🔴 Scump' if my_pe > 25 else '🟢 Atractiv'}")
            
            with c_p2:
                st.metric("ROE vs Sector", f"{my_roe:.1f}%", 
                          f"{'🟢 Lider' if my_roe > 15 else '🟡 Mediu'}")

            with c_p3:
                # Interpretare profesională pentru ROA (peste 5% e considerat bun)
                roa_status = "💎 Excelent" if my_roa > 5 else "⚠️ Scăzut"
                st.metric("ROA vs Sector", f"{my_roa:.1f}%", roa_status)
            
            with c_p4:
                st.metric("Marjă Netă", f"{my_margin:.1f}%", 
                          f"{'🚀 Eficient' if my_margin > 15 else '⚖️ Standard'}")    

            st.write("") # Mic spațiu între carduri și tabel

            # --- PASUL 2: TABELUL COMPARATIV (JOS) ---
            st.markdown("**🔍 Comparație Detaliată cu Benchmark-urile Industriei:**")
            with st.spinner("Se analizează competitorii..."):
                df_peers = get_peers_analysis(info.get('sector'), info.get('industry'), real_sym)
                
                if not df_peers.empty:
                    # Aplicăm stilizare profesională tabelului
                    st.dataframe(df_peers.style.format({
                        "P/E": "{:.2f}",
                        "ROE (%)": "{:.1f}%",
                        "ROA (%)": "{:.1f}%",
                        "Marjă Netă (%)": "{:.1f}%",
                        "Datorii/Eq (%)": "{:.1f}%"
                    }), use_container_width=True, hide_index=True)
                else:
                    st.info("Informații despre competitori indisponibile pentru acest simbol.")

            st.caption(f"💡 Analiza compară eficiența {real_sym} cu giganții din sectorul {info.get('sector')}.")
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
                rec = info.get('recommendationKey', 'N/A').replace('_', ' ').upper()
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
                    st.dataframe(e_disp.style.applymap(style_surprise, subset=['Surpriză %']).format({'Estimare': '{:.2f}', 'Realizat': '{:.2f}', 'Diferență': '{:.2f}', 'Surpriză %': '{:.2%}'}), use_container_width=True)
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
            
            # --- MODUL REPARAT: SMART MONEY FLOW (FIX PROCENT) ---
            st.markdown("---")
            st.subheader("🐳 Smart Money Flow: Dețineri Instituționale")

            try:
                t_whale = yf.Ticker(real_sym)
                major_df = t_whale.major_holders
                
                inst_percent = 0.0
                
                if major_df is not None and not major_df.empty:
                    # Resetăm indexul pentru a procesa corect ambele coloane
                    major_df = major_df.reset_index()
                    
                    for _, row in major_df.iterrows():
                        label = str(row.iloc[1]).lower() # De obicei a doua coloană e textul
                        value = row.iloc[0]              # Prima coloană e valoarea
                        
                        if 'institutions' in label or 'institu' in label:
                            try:
                                # Conversie sigură la float
                                raw_val = float(value)
                                # FIX: Dacă valoarea este > 1, înseamnă că e deja procent (ex: 60.5)
                                # Dacă este < 1, este fracție (ex: 0.605)
                                if 0 < raw_val <= 1.0:
                                    inst_percent = raw_val * 100
                                elif 1.0 < raw_val <= 100.0:
                                    inst_percent = raw_val
                                else:
                                    # Dacă e peste 100, e probabil numărul de instituții, nu procentul
                                    # Căutăm în cealaltă coloană sau rândul următor
                                    continue
                                break
                            except: continue

                # Dacă tot e 0, încercăm o metodă secundară din info
                if inst_percent == 0:
                    inst_percent = info.get('heldPercentInstitutions', 0) * 100

                iw_col1, iw_col2 = st.columns([1, 2])
                
                with iw_col1:
                    # Determinarea culorii și mesajului pe baza pragurilor profesionale
                    if inst_percent > 70:
                        status_msg = "💎 SUPORT ELITĂ"
                        status_desc = "Instituțiile domină total. Volatilitate scăzută."
                        status_color = "#3FB950" # Verde aprins
                    elif inst_percent > 50:
                        status_msg = "✅ SUPORT SOLID"
                        status_desc = "Majoritate instituțională. Bază de investitori stabilă."
                        status_color = "#238636" # Verde închis
                    elif inst_percent > 30:
                        status_msg = "⚖️ POZIȚIONARE MIXTĂ"
                        status_desc = "Echilibru între fonduri și retail. Atenție la știri."
                        status_color = "#D29922" # Galben/Portocaliu
                    else:
                        status_msg = "🚨 EXTREM DE SLAB"
                        status_desc = "Retail dominant. Risc ridicat de mișcări speculative."
                        status_color = "#F85149" # Roșu

                    st.markdown(f"""
                        <div style="background:#161B22; padding:20px; border-radius:15px; border:2px solid {status_color}; text-align:center;">
                            <p style="color:#8B949E; margin:0; font-size:11px; text-transform:uppercase;">Dețineri Instituționale Totale</p>
                            <h1 style="color:{status_color}; margin:10px 0;">{inst_percent:.2f}%</h1>
                            <div style="background:{status_color}22; color:{status_color}; padding:5px; border-radius:5px; font-weight:bold; font-size:14px;">
                                {status_msg}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                with iw_col2:
                    st.markdown(f"### 🚩 Analiză Acționariat: {real_sym}")
                    st.write(f"**Verdict:** {status_desc}")
                    
                    # Bara de progres vizuală pentru context rapid
                    st.progress(inst_percent / 100)
                    
                    st.markdown(f"""
                    <div style="background:#21262D; padding:15px; border-radius:10px; margin-top:10px;">
                        <p style="font-size:14px; margin-bottom:5px;">💡 <b>De ce contează?</b></p>
                        <p style="font-size:13px; color:#8B949E; line-height:1.4;">
                            {'Fondurile mari (Vanguard, BlackRock) au "mâini puternice" și nu vând în panică, oferind un prag de suport prețului.' if inst_percent > 50 
                            else 'Lipsa instituțiilor înseamnă că prețul este dictat de investitori mici, care pot vinde agresiv la prima veste negativă.'}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.caption("Analiza deținerilor este optimizată pentru acțiunile listate în SUA.")
            st.markdown("---")

            # 5. Calculator Fair Value (REPARAT - REACTIVITATE TOTALĂ)
            st.subheader("🧮 Calculator Valoare Intrinsecă (Valoare justă)")
            
            # Preluare Date
            eps_f = info.get('trailingEps', 0)
            bv_f = info.get('bookValue', 0)
            price_f = info.get('currentPrice') or info.get('previousClose', 0)
            t_curr = info.get('currency', 'USD')

            st.write("⚙️ **Configurați Ipotezele: Sliderele influențează acum ambele modele!**")
            ctrl1, ctrl2 = st.columns(2)
            
            # Explicații profesionale pentru Tooltips
            eps_help = """
            **Creșterea EPS (Earnings Per Share):**
            Reprezintă rata anuală compusă cu care estimezi că vor crește profiturile companiei în următorii 5-10 ani.
            - 0-5%: Companii mature (Utility, Consumer Staples)
            - 10-20%: Companii de creștere (Tech, Healthcare)
            - 25%: Estimare foarte optimistă, greu de susținut pe termen lung.
            """

            discount_help = """
            **Rata de Scont (Discount Rate):**
            Reprezintă randamentul minim pe care îl ceri de la această investiție pentru a justifica riscul.
            - 7-9%: Companii sigure, cu cash-flow stabil (Blue Chips).
            - 10-12%: Media pieței (S&P 500).
            - 13-15%+: Companii riscante sau cu datorii mari.
            Cu cât rata de scont e mai mare, cu atât valoarea justă calculată va fi mai mică.
            """

            # Sliderele cu pas de 1% și explicații incluse
            growth_val = ctrl1.slider(
                "Creștere anuală EPS (%)", 
                -5, 40, 15, step=1, 
                help=eps_help,
                key="v_final_g"
            )
            
            discount_val = ctrl2.slider(
                "Rata de scont (%)", 
                5, 20, 9, step=1, 
                help=discount_help,
                key="v_final_d"
            )
            
            # --- LOGICĂ REACTIVĂ ---
            # 1. Graham Revizuit: V = EPS * (8.5 + 2 * Growth)
            # Folosim formula adaptată a lui Graham pentru a fi influențată de slider-ul de creștere
            graham_calc = eps_f * (8.5 + 2 * growth_val) if eps_f > 0 else 0
            
            # 2. DCF Reactiv
            dcf_calc = calculate_dcf_dynamic(info, growth_val, discount_val)

            # --- AFISARE REZULTATE ---
            if price_f > 0:
                cv1, cv2, cv3 = st.columns(3)
                css = "border: 2px solid {c}; padding: 20px; border-radius: 12px; text-align: center; background-color: #161B22; height: 180px; display: flex; flex-direction: column; justify-content: center;"
                
                with cv1:
                    st.markdown(f'<div style="{css.format(c="#30363D")}"><p style="color:#8B949E; font-size:13px; text-transform:uppercase;">Preț Curent</p><h1 style="color:white; margin:10px 0;">{price_f:.2f} <span style="font-size:14px;">{t_curr}</span></h1></div>', unsafe_allow_html=True)
                
                with cv2:
                    if graham_calc > 0:
                        diff_g = ((price_f - graham_calc) / graham_calc) * 100
                        g_col = "#3FB950" if price_f < graham_calc else "#F85149"
                        st.markdown(f'<div style="{css.format(c=g_col)}"><p style="color:#8B949E; font-size:13px; text-transform:uppercase;">Graham (Adaptat)</p><h1 style="color:{g_col}; margin:10px 0;">{graham_calc:.2f}</h1><p style="color:{g_col}; font-weight:bold; font-size:12px;">{"SUBEVALUAT" if price_f < graham_calc else "SUPRAEVALUAT"} ({abs(diff_g):.1f}%)</p></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div style="{css.format(c="#30363D")}"><p style="color:#8B949E;">Graham N/A</p></div>', unsafe_allow_html=True)

                with cv3:
                    if dcf_calc > 0:
                        diff_d = ((price_f - dcf_calc) / dcf_calc) * 100
                        d_col = "#3FB950" if price_f < dcf_calc else "#F85149"
                        st.markdown(f'<div style="{css.format(c=d_col)}"><p style="color:#8B949E; font-size:13px; text-transform:uppercase;">Valoare Justă (DCF)</p><h1 style="color:{d_col}; margin:10px 0;">{dcf_calc:.2f}</h1><p style="color:{d_col}; font-weight:bold; font-size:12px;">{"SUBEVALUAT" if price_f < dcf_calc else "SUPRAEVALUAT"} ({abs(diff_d):.1f}%)</p></div>', unsafe_allow_html=True)
         
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
            h_color = "#3FB950" if h_score >= 8 else ("#D29922" if h_score >= 5 else "#F85149")
            
            c_left, c_right = st.columns([1, 2])
            
            with c_left:
                st.markdown(f"""
                <div style="background:#161B22; padding:30px; border-radius:15px; border:2px solid {h_color}; text-align:center;">
                    <p style="color:#8B949E; margin:0; font-size:11px; text-transform:uppercase;">Scor Sănătate Financiară</p>
                    <h1 style="color:{h_color}; margin:15px 0; font-size:54px;">{h_score}<span style="font-size:18px;">/10</span></h1>
                    <hr style="border-color:#30363D;">
                    <p style="font-size:13px; color:#8B949E;">Beta: {display_beta} | Alpha: {display_alpha}</p>
                </div>
                """, unsafe_allow_html=True)
                
            with c_right:
                st.markdown(f"**Analiză de Specialist în Sectorul:** `{info.get('sector', 'N/A')}`")
                for line in audit_report:
                    st.info(line)

            # --- MODUL MARJA DE SIGURANȚĂ REPARAT ---
            st.markdown("---")
            st.subheader("🛡️ Analiza Marjei de Siguranță")
            
            current_p = info.get('currentPrice', 0)
            target_val = dcf_calc if dcf_calc and dcf_calc > 0 else 0
            
            if target_val > 0 and current_p > 0:
                mos_val = ((target_val - current_p) / target_val) * 100
                # ... restul codului de afișare rămâne la fel ...
            else:
                st.warning("⚠️ Date insuficiente pentru calculul Marjei de Siguranță.") 
            
            if target_val > 0:
                mos_val = ((target_val - current_p) / target_val) * 100
                
                # Logica de culori și mesaje detaliate
                if mos_val > 30:
                    mos_verdict = "🚀 **Oportunitate Majoră (Deep Value):** Acțiunea se vinde cu un discount masiv. Aceasta este zona ideală pentru un investitor de valoare."
                    mos_color = "#3FB950" 
                elif mos_val > 10:
                    mos_verdict = "✅ **Preț Atractiv:** Există o marjă de siguranță care te protejează împotriva erorilor de estimare în modelul DCF."
                    mos_color = "#3FB950"
                elif mos_val > -10:
                    mos_verdict = "⚖️ **Evaluare neutră:** Prețul este corect. Nu ai marjă de siguranță, deci orice veste proastă poate duce la scăderi imediate."
                    mos_color = "#D29922" 
                else:
                    mos_verdict = "🚨 **SUPRAEVALUARE CRITICĂ:** Plătești un premium periculos. Riscul de 'reversie la medie' este extrem de ridicat."
                    mos_color = "#F85149"

                m_col1, m_col2 = st.columns([1, 2])
                
                with m_col1:
                    st.markdown(f"""
                    <div style="background:#161B22; padding:25px; border-radius:15px; border:2px solid {mos_color}; text-align:center;">
                        <p style="color:#8B949E; margin:0; font-size:11px; text-transform:uppercase;">Margin of Safety</p>
                        <h1 style="color:{mos_color}; margin:10px 0; font-size:40px;">{mos_val:.1f}%</h1>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with m_col2:
                    # Afișarea verdictului explicativ
                    if mos_val < -20:
                        # Avertizarea vizuală agresivă pentru supraevaluare
                        st.error(f"⚠️ **ALERTĂ DE RISC:** {mos_verdict}")
                        st.write("👉 *Sfat Analist:* Istoric, cumpărarea în această zonă a dus la randamente negative pe termen lung.")
                    elif mos_val > 30:
                        st.success(f"🌟 **SURPRIZĂ DE VALOARE:** {mos_verdict}")
                    else:
                        st.info(mos_verdict)
                    
                    st.write(f"💵 **Preț Actual:** {current_p:.2f} | 🎯 **Fair Value:** {target_val:.2f}")
                    st.progress(max(0, min(mos_val / 100, 1.0)) if mos_val > 0 else 0)
            else:
                st.warning("⚠️ Calculul marjei de siguranță necesită un Fair Value DCF valid.")

            # --- MODUL: SUSTENABILITATE ȘI CALITATE (VERSIUNE EXTINSĂ) ---
            st.markdown("---")
            st.subheader("🧬 Sustenabilitate și Calitatea Profitului")
            
            div_verdicts, q_ratio, p_ratio = analyze_dividend_quality(info)
            q_color = "#3FB950" if q_ratio > 1 else ("#D29922" if q_ratio > 0.7 else "#F85149")
            
            c_q1, c_q2 = st.columns([1, 2])
            
            with c_q1:
                # Vizualizare principală scor cash
                st.markdown(f"""
                <div style="background:#161B22; padding:20px; border-radius:15px; border:2px solid {q_color}; text-align:center;">
                    <p style="color:#8B949E; margin:0; font-size:11px; text-transform:uppercase;">Cash-to-Income Ratio</p>
                    <h1 style="color:{q_color}; margin:10px 0; font-size:35px;">{q_ratio:.2f}x</h1>
                    <p style="font-size:12px; color:#8B949E;">Plată Dividend (Payout): {p_ratio:.1f}%</p>
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
                **Ce înseamnă {q_ratio:.2f}x (Cash-to-Income)?**
                * **Peste 1.0x:** Afacere de tip 'Cash Machine'. Firma încasează mai mulți bani reali decât profitul declarat contabil.
                * **0.7x - 1.0x:** Nivel normal pentru companii în creștere.
                * **Sub 0.7x:** Profitul este doar pe hârtie. Există riscul ca facturile să nu fie încasate.
                """)
                
            with col_info2:
                st.markdown(f"""
                **Ce înseamnă {p_ratio:.1f}% (Payout Ratio)?**
                * **30% - 60%:** Zona ideală (Sweet Spot). Dividend sigur și loc de creștere.
                * **Peste 80%:** Zona de pericol. Firma dă aproape tot profitul afară; orice scădere a vânzărilor va duce la tăierea dividendului.
                """)

            # --- MODUL: ANALIZĂ STRATEGICĂ IA (SWOT) ---
            st.markdown("---")
            st.subheader("🎯 Analiză Strategică IA (SWOT)")
            try:
                # Importuri din modulul extern
                from ai_engine import analyze_sentiment_ai, generate_ai_swot_analysis
                
                # Colectare date necesare pentru SWOT
                c_news_ai = get_company_news_rss(real_sym)
                s_score_val = analyze_sentiment_ai(c_news_ai) if c_news_ai else 0
                mos_swot = ((dcf_calc - current_p) / dcf_calc * 100) if dcf_calc > 0 else 0
                z_val_swot, _, _, _ = calculate_altman_z(info)
                
                # Generare date SWOT
                swot_res = generate_ai_swot_analysis(info, h_score, z_val_swot, mos_swot, alpha_val, s_score_val)
                
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
            # --- NOU: RADAR REGIM DE PIAȚĂ AI (K-Means) ---
            with st.spinner("AI-ul clasifică regimul de piață..."):
                from ai_engine import detect_market_regime_ai
                regime_msg, regime_color = detect_market_regime_ai(hist)
                
                st.markdown(f"""
                <div style='background:#161B22; padding:15px; border-radius:10px; border-left: 5px solid {regime_color}; margin-bottom: 20px;'>
                    <p style='margin:0; color:#8B949E; font-size: 11px; text-transform: uppercase; font-weight: bold;'>Radar AI: Regimul Curent al Pieței (K-Means)</p>
                    <h3 style='margin:5px 0 0 0; color:{regime_color}; font-size: 18px;'>{regime_msg}</h3>
                </div>
                """, unsafe_allow_html=True)
                       
            c_news_ai = get_company_news_rss(real_sym)
            cai1, cai2 = st.columns([1, 2])
            
            # --- PARTEA ACTUALIZATĂ (Scorul FinBERT cu Legendă) ---
            with cai1:
                st.write("📊 **Analiză Sentiment (FinBERT)**")
                if c_news_ai:
                    with st.spinner("AI-ul analizează contextul..."):
                        from ai_engine import analyze_sentiment_ai
                        s_score = analyze_sentiment_ai(c_news_ai)
                        
                        # Definim intervalele profesionale și etichetele
                        if s_score >= 0.25:
                            c_ai, status_text = "#3FB950", "🚀 Puternic Pozitiv"
                        elif s_score > 0.05:
                            c_ai, status_text = "#3FB950", "📈 Pozitiv (Bullish)"
                        elif s_score <= -0.25:
                            c_ai, status_text = "#F85149", "🚨 Puternic Negativ"
                        elif s_score < -0.05:
                            c_ai, status_text = "#F85149", "📉 Negativ (Bearish)"
                        else:
                            c_ai, status_text = "#8B949E", "⚖️ Neutru"

                        # Randăm Cardul și Legenda explicativă
                        st.markdown(f"""
                        <div style='background:#161B22; padding:20px; border-radius:15px; border:1px solid {c_ai}; text-align:center;'>
                            <h1 style='color:{c_ai}; margin:0; font-size:48px;'>{s_score:.2f}</h1>
                            <div style='color:{c_ai}; font-weight:bold; font-size:16px; margin-top:5px;'>{status_text}</div>
                        </div>
                        
                        <div style='margin-top: 15px; padding: 15px; background: #21262D; border-radius: 10px; border-left: 3px solid #58A6FF;'>
                            <p style='color:#8B949E; font-size: 11px; margin:0 0 8px 0; text-transform: uppercase; font-weight: bold;'>Ghid Intervale (Scală -1 la +1)</p>
                            <div style='color:#C9D1D9; font-size: 13px; line-height: 1.6;'>
                                <div><span style='color:#3FB950;'>■</span> <b>+0.25 la +1.00:</b> Euforie media</div>
                                <div><span style='color:#3FB950; opacity: 0.7;'>■</span> <b>+0.05 la +0.24:</b> Optimism moderat</div>
                                <div><span style='color:#8B949E;'>■</span> <b>-0.05 la +0.05:</b> Zgomot neutru</div>
                                <div><span style='color:#F85149; opacity: 0.7;'>■</span> <b>-0.24 la -0.06:</b> Îngrijorare / Pesimism</div>
                                <div><span style='color:#F85149;'>■</span> <b>-1.00 la -0.25:</b> Panică extremă</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            with cai2:
                st.write("📈 **Prognoză Algoritmică (Next 90 Days)**")
                if len(hist) > 100:
                    from ai_engine import predict_stock_price, render_ai_chart
                    forecast = predict_stock_price(hist)
                    render_ai_chart(forecast, hist)
            
            # ==================================================
            # CALCUL RATING FINAL (CONCLUZIA)
            # ==================================================
            st.markdown("---")
            st.subheader("🎯 Verdict Final: Rating de Investiție")

            # Pregătim variabilele (Safe check)
            # Luăm spread-ul din piață direct pentru rating
            try:
                t_10y = yf.Ticker("^TNX").fast_info.last_price
                t_2y = yf.Ticker("2Y=F").fast_info.last_price
                curr_spread = t_10y - t_2y
            except:
                curr_spread = 0.5

            s_inst = inst_percent if 'inst_percent' in locals() else 0
            s_mos = mos_val if 'mos_val' in locals() else 0
            s_rvol = rvol if 'rvol' in locals() else 1.0
            
            # Apelăm funcția PRO care returnează SCOR + DETALII
            final_score, highlights = calculate_investment_rating_pro(info, s_inst, s_rvol, curr_spread, s_mos)
            
            r_color = "#3FB950" if final_score > 70 else ("#D29922" if final_score > 40 else "#F85149")
            r_label = "STRONG BUY" if final_score > 80 else ("ACCUMULATE" if final_score > 60 else "AVOID/WATCH")

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
                        st.plotly_chart(fig_season, use_container_width=True)
                    
                    with s_col2:
                        # Extragem extremele pentru a oferi un verdict clar utilizatorului
                        best_m = df_stats.loc[df_stats['Win Rate (%)'].idxmax()]
                        worst_m = df_stats.loc[df_stats['Win Rate (%)'].idxmin()]
                        
                        st.markdown("#### 💡 Analiză Quant")
                        st.success(f"**🌟 Cea mai bună lună: {best_m['Luna']}**\n\nIstoric, prețul a crescut în **{best_m['Win Rate (%)']:.0f}%** din cazuri, aducând un randament mediu de **+{best_m['Randament Mediu (%)']:.2f}%**.")
                        
                        st.error(f"**🚨 Cea mai slabă lună: {worst_m['Luna']}**\n\nIstoric, a avut o rată de succes de doar **{worst_m['Win Rate (%)']:.0f}%**, cu o scădere medie de **{worst_m['Randament Mediu (%)']:.2f}%**.")
                        
                        st.info("📉 **Cum folosești acest modul:** Elimină ghicitul și emoția. Dacă dorești să cumperi această acțiune, dar te afli într-o lună cu Win Rate sub 40%, șansele matematice sunt împotriva ta. Așteaptă luna verde pentru a deschide o poziție la un preț statistic favorabil.")
                else:
                    # Dacă df_stats este string, înseamnă că a returnat mesajul de eroare
                    st.info(df_stats)
            
            # =================================================================
            # MODUL PRO: RAZE X - ANALIZĂ FLUX DERIVATE (FINAL & CURAT)
            # =================================================================
            st.markdown("---")
            st.subheader("🕵️‍♂️ Raze X: Fluxul de Bani din Opțiuni (Pro)")
            
            from ai_engine import get_options_analysis_ai
            with st.spinner("Se decodează contractele Market Makerilor..."):
                opt_data, opt_msg = get_options_analysis_ai(real_sym)
                
                if opt_data:
                    # --- RÂNDUL 1: VITEZOMETRU ȘI METRICE ---
                    col_gau, col_met = st.columns([1.5, 2])
                    
                    with col_gau:
                        fig_iv = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = opt_data['iv'],
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': f"Termometru IV ({opt_data['iv_status']})", 'font': {'size': 18}},
                            number = {'suffix': "%", 'font': {'color': opt_data['iv_color']}},
                            gauge = {
                                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                                'bar': {'color': opt_data['iv_color']},
                                'bgcolor': "rgba(0,0,0,0)",
                                'borderwidth': 2,
                                'bordercolor': "#30363D",
                                'steps': [
                                    {'range': [0, 20], 'color': 'rgba(63, 185, 80, 0.2)'},
                                    {'range': [20, 45], 'color': 'rgba(210, 153, 34, 0.2)'},
                                    {'range': [45, 100], 'color': 'rgba(248, 81, 73, 0.2)'}
                                ],
                                'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': opt_data['iv']}
                            }
                        ))
                        fig_iv.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_iv, use_container_width=True)

                    with col_met:
                        m1, m2 = st.columns(2)
                        m1.metric("Put/Call Ratio (OI)", f"{opt_data['oi_pc_ratio']:.2f}")
                        m2.metric("Put/Call Ratio (Volum)", f"{opt_data['vol_pc_ratio']:.2f}")
                        
                        m3, m4 = st.columns(2)
                        mp = opt_data['max_pain']
                        diff_mp = ((mp / curr_price) - 1) * 100 if curr_price > 0 else 0
                        m3.metric("Preț Max Pain", f"${mp:.1f}", f"{diff_mp:.1f}%")
                        m4.metric("Data Expirării", opt_data['expiration'])

                    # --- RÂNDUL 2: VERDICTUL INTELIGENT ---
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

                    # --- RÂNDUL 3: STRATEGIA IV (AICI EXPLICI CONTRADICȚIA) ---
                    if iv_val > 45:
                        st.warning(f"⚠️ **ALERTA IV:** Deși direcția pare {v_text.split(':')[0][2:]}, opțiunile sunt **prea scumpe** ({iv_val:.1f}%). Riscul de scădere a valorii prin volatilitate este uriaș. Recomandare: Acțiuni direct, nu derivate.")
                    elif iv_val < 20:
                        st.success(f"💎 **OPORTUNITATE IV:** Opțiunile sunt foarte ieftine ({iv_val:.1f}%). Moment ideal pentru a paria pe direcția identificată.")

                    # --- RÂNDUL 4: GHID DINAMIC (TABELUL FIXAT) ---
                    st.markdown("#### 🔍 Detalierea Indicatorilor")
                    interpretare_data = [
                        {"Indicator": "📉 Put/Call (OI)", "Valoare": f"{oi_pc:.2f}", "Interpretare": "Bullish" if oi_pc < 0.7 else "Bearish" if oi_pc > 1.1 else "Neutru"},
                        {"Indicator": "⚡ Put/Call (Volum)", "Valoare": f"{vol_pc:.2f}", "Interpretare": "Sentiment Bullish" if vol_pc < 0.7 else "Panică" if vol_pc > 1.1 else "Normal"},
                        {"Indicator": "🧲 Max Pain Price", "Valoare": f"${mp:.1f}", "Interpretare": f"Prețul tinde spre ${mp:.1f}"},
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
            
            # --- PROTECȚIE VARIABILE (Safety Net pentru date lipsă din Yahoo) ---
            s_inst = inst_percent if 'inst_percent' in locals() else 0
            s_mos = mos_val if 'mos_val' in locals() else 0
            s_rvol = rvol if 'rvol' in locals() else 1.0
            s_score_final = s_score_val if 's_score_val' in locals() else 0 
            opt_final = opt_data if 'opt_data' in locals() else None
            
            # Extragem datele specifice modulelor adiacente
            z_score_val = z_val_swot if 'z_val_swot' in locals() else 3.0
            cash_ratio = q_ratio if 'q_ratio' in locals() else 1.0
            ai_regime = regime_msg if 'regime_msg' in locals() else "Neutru"

            # Extragem Spread-ul Macro la cald
            try:
                t_10y = yf.Ticker("^TNX").fast_info.last_price
                t_2y = yf.Ticker("2Y=F").fast_info.last_price
                curr_spread = t_10y - t_2y
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
            st.markdown("---")
            
            # 7. Ultimele Știri (RESTABILITE)
            st.subheader(f"📰 Ultimele Știri despre {real_sym}")
            if c_news_ai:
                for n in c_news_ai:
                    sentiment, css_cls, icon = get_sentiment(n['title'])
                    c_t, c_i = st.columns([5, 1])
                    with c_t:
                        st.markdown(f"**[{n['title']}]({n['link']})**")
                        st.caption(f"{n['publisher']} • {n['date_str']}")
                    with c_i:
                        st.markdown(f"<span class='{css_cls}'>{icon} {sentiment}</span>", unsafe_allow_html=True)
                    st.divider()
                                
    # ==================================================
    # 3. PORTOFOLIU (MODIFICAT PENTRU MOBIL)
    # ==================================================
    elif sectiune == "3. Portofoliu":
        st.title("💼 Portofoliu Personal")
        
        with st.expander("➕ Adaugă Tranzacție Nouă"):
            with st.form("add_pf"):
                c1, c2, c3, c4 = st.columns(4)
                s = c1.text_input("Simbol (ex: AAPL, EUNL.DE)").upper()
                q = c2.number_input("Cantitate", min_value=0.01, value=1.0, format="%.4f")
                p = c3.number_input("Preț Achiziție", min_value=0.01, value=100.0, format="%.2f")
                curr = c4.selectbox("Moneda", ["USD", "EUR"]) 
                
                d_acq = st.date_input("Data", datetime.today())
                
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
                    df_calc, hist_curve, daily_abs, daily_pct = calculate_portfolio_performance(df_subset, hist_range)

                total_invested = (df_subset['Quantity'] * df_subset['AvgPrice']).sum()
                total_current = df_calc['MarketValue'].sum() if not df_calc.empty else 0
                
                total_profit_val = total_current - total_invested
                total_profit_pct = (total_profit_val / total_invested * 100) if total_invested != 0 else 0

                c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
                c_kpi1.metric(f"Total Investit ({currency_symbol})", f"{total_invested:,.2f} {currency_symbol}")
                c_kpi2.metric(f"Valoare Curentă ({currency_symbol})", f"{total_current:,.2f} {currency_symbol}")
                c_kpi3.metric(f"Profit/Pierdere ({currency_symbol})", f"{total_profit_val:,.2f} {currency_symbol}", f"{total_profit_pct:.2f}%")

                st.markdown("---")
                
                if not hist_curve.empty:
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Scatter(
                        x=hist_curve.index, y=hist_curve.values, 
                        fill='tozeroy', line=dict(color='#238636'), name=f'Valoare {currency_symbol}'
                    ))
                    fig_hist.update_layout(height=350, template="plotly_dark", margin=dict(t=10, b=10), paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_hist, use_container_width=True)
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
                    current_bench_name = "Indice BET (RON)"
                else:
                    # Rămâne STOXX 600 doar pentru portofoliul în EURO
                    current_bench_ticker = "EXW1.DE"
                    current_bench_name = "STOXX 600 (EUR)"

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
                    bench_name = "Indice BET (RON)"
                else:
                    # Rămâne STOXX 600 doar pentru portofoliul în EUR
                    bench_ticker = "EXW1.DE"
                    bench_name = "STOXX 600 (EUR)"
                
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

                st.markdown("---")

                # 2. Grafice Plăcintă: Simboluri vs Sectoare
                st.subheader("🍰 Distribuția Activelor")
                col_pie1, col_pie2 = st.columns(2)
                
                with col_pie1:
                    st.caption("**După Companie (Simbol)**")
                    if not df_calc.empty:
                        fig_sym = go.Figure(data=[go.Pie(
                            labels=df_calc['Symbol'], 
                            values=df_calc['MarketValue'], 
                            hole=.4,
                            textinfo='percent',
                            hovertemplate="<b>%{label}</b><br>Valoare: %{value:,.2f} " + currency_symbol + "<br>Pondere: %{percent}<extra></extra>"
                        )])
                        fig_sym.update_layout(height=350, margin=dict(t=0, b=0, l=0, r=0), 
                                              template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_sym, use_container_width=True, key=f"pie_sym_{currency_symbol}")

                # 4. Detaliu Poziții
                st.subheader("Detaliu Poziții")
                if not df_calc.empty:
                    display_cols = ['Symbol', 'Quantity', 'AvgPrice', 'CurrentPrice', 'MarketValue', 'Profit', 'Profit %']
                    
                    def color_profit(val):
                        color = '#3FB950' if val >= 0 else '#F85149'
                        return f'color: {color}'

                    st.dataframe(
                        df_calc[display_cols].style.map(color_profit, subset=['Profit', 'Profit %'])
                        .format({
                            'Quantity': '{:.1f}', 'AvgPrice': '{:.4f}', 'CurrentPrice': '{:.4f}',
                            'MarketValue': '{:,.4f}', 'Profit': '{:,.2f}', 'Profit %': '{:.2f}%'
                        }),
                        use_container_width=True
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
                        st.plotly_chart(fig_sec, use_container_width=True, key=f"pie_sec_{currency_symbol}")

                        # VERDICT DIVERSIFICARE
                        max_sector = df_sectors.iloc[0]
                        if max_sector['Pondere %'] > 40:
                            st.warning(f"⚠️ **Concentrare mare:** Sectorul '{max_sector['Sector']}' ocupă {max_sector['Pondere %']:.1f}% din portofoliu. Riști mult dacă acest sector scade.")
                        else:
                            st.success(f"✅ **Diversificare bună:** Niciun sector nu depășește 40%.")
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
                        hist_opt = yf.download(current_tickers, period="1y", progress=False)['Close']
                        
                        # 2. Trimitem datele la creierul AI
                        from ai_engine import optimize_portfolio_ai
                        opt_res, opt_msg = optimize_portfolio_ai(hist_opt)

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
                            st.plotly_chart(fig_opt, use_container_width=True, key=f"opt_bar_{currency_symbol}")

                            # 4. Afișăm Metricele Portofoliului Ideal
                            st.markdown("##### 🏆 Cum ar arăta Portofoliul Ideal (Conform AI):")
                            c_op1, c_opt2, c_opt3 = st.columns(3)
                            c_op1.metric("Randament Anual Așteptat", f"{opt_res['expected_return']:.2f}%")
                            c_opt2.metric("Volatilitate (Risc)", f"{opt_res['expected_volatility']:.2f}%")
                            c_opt3.metric("Sharpe Ratio Optim", f"{opt_res['sharpe_ratio']:.2f}")

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
    # 4. PIAȚĂ GLOBALĂ (CU MASTER VERDICT AI INTEGRAT)
    # =================================================================
    elif sectiune == "4. Piață Globală":
        st.title("🌐 Pulsul Pieței Globale")
        st.caption("Date în timp real (cu întârziere minimă) furnizate via Yahoo Finance.")

        # --- MODUL ACTUALIZAT: MASTER MACRO VERDICT (V2 CU VIX INTEGRAT) ---
        with st.spinner("Motorul Macro AI analizează corelațiile și frica în piață..."):
            # A. Colectăm datele existente
            macro_sectors = get_sector_performance()
            macro_risk_ratio = get_credit_risk_data("1y")
            macro_corr = get_cross_asset_correlation()
            
            # B. Obținem VIX-ul (Frica) în timp real
            try:
                vix_val = yf.Ticker("^VIX").fast_info.last_price
            except:
                vix_val = 20.0 # Valoare neutră în caz de eroare

            # C. Preluăm sentimentul global din știri
            news_samples = get_company_news_rss("^GSPC") + get_company_news_rss("^IXIC")
            from ai_engine import analyze_sentiment_ai
            macro_sentiment = analyze_sentiment_ai(news_samples) if news_samples else 0
            
            # D. Preluăm spread-ul 10Y-2Y
            try:
                t_10y = yf.Ticker("^TNX").fast_info.last_price
                t_2y = yf.Ticker("2Y=F").fast_info.last_price
                curr_yield_spread = t_10y - t_2y
            except: curr_yield_spread = 0.5

            # E. Apelăm noua funcție cu 6 parametri (am adăugat vix_val la final)
            from ai_engine import calculate_master_macro_verdict
            m_score, m_label, m_col, m_desc, m_reasons = calculate_master_macro_verdict(
                macro_sectors, macro_risk_ratio, macro_corr, macro_sentiment, curr_yield_spread, vix_val
            )

            # AFIȘARE VIZUALĂ BANNER SUPREM
            st.markdown(f"""
                <div style="background:linear-gradient(90deg, #161B22 0%, #21262D 100%); padding:30px; border-radius:15px; border-left: 10px solid {m_col}; margin-bottom:30px; box-shadow: 0 4px 15px rgba(0,0,0,0.5);">
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

            # Afișăm motivele pe două coloane
            st.markdown("#### 🔍 Argumentele Modelului (Analiză Cross-Asset):")
            c_re1, c_re2 = st.columns(2)
            for i, reason in enumerate(m_reasons):
                if i % 2 == 0: c_re1.markdown(f"{reason}")
                else: c_re2.markdown(f"{reason}")       
        st.markdown("---")

        # --- BUTON REFRESH ȘI MODULELE TALE VECHI CONTINUĂ AICI ---
        if st.button("🔄 Reîmprospătează Piața"):
            get_global_market_data.clear()
            get_macro_data_visuals.clear()
            st.rerun()

        st.subheader("🚨 Early Warning System: Risc Recesiune (10Y-2Y)")
        
        try:
            tickers_yield = ['^TNX', '^IRX'] 
            yield_data = yf.download(tickers_yield, period="5d", progress=False)['Close']
            t_10y = yf.Ticker("^TNX").fast_info.last_price
            try:
                t_2y = yf.Ticker("2Y=F").fast_info.last_price 
                spread = t_10y - t_2y
                label_spread = "Spread 10Y - 2Y"
            except:
                t_3m = yf.Ticker("^IRX").fast_info.last_price
                spread = t_10y - t_3m
                label_spread = "Spread 10Y - 3M (Fallback)"

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
                    st.error(f"⚠️ **CURBĂ INVERSATĂ:** Diferența este de {spread:.3f}. Istoric, acest semnal a precedat fiecare recesiune majoră.")
                    st.write("👉 **Strategie:** Redu expunerea pe acțiuni ciclice.")
                else:
                    st.success(f"✅ **CURBĂ NORMALĂ:** Diferența de {spread:.3f} indică expansiune economică.")
                    st.write("👉 **Strategie:** Poți menține o strategie de creștere.")
        except:
            st.info("Datele pentru curba randamentelor se încarcă...")

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
                days_map = {"1L": 30, "3L": 90, "6L": 180, "1A": 365, "3A": 1095, "5A": 1825}
                days = days_map.get(time_frame, 365)
                subset = series.iloc[-days:] # Tăiem exact cât a cerut userul
                
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
                
                st.plotly_chart(fig_macro, use_container_width=True)
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
                    st.plotly_chart(fig_rot, use_container_width=True)

                    # --- PASUL B: METRICILE ȘI EXPLICAȚIA (JOS) ---
                    st.write("") # Mic spațiu între grafic și text
                    c_inf1, c_inf2 = st.columns([1, 2]) # Organizăm detaliile pe două coloane sub grafic
                    
                    with c_inf1:
                        rot_status = "🚀 GROWTH (Nasdaq)" if rot_change > 0 else "🏭 VALUE (Dow Jones)"
                        st.metric("Trend Rotație (30z)", rot_status, f"{rot_change:+.2f}%")
                        st.write(f"Scor actual: **{current_ratio:.4f}**")
                        
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
                    st.plotly_chart(fig_sec_heat, use_container_width=True)
                
                # --- INTERPRETAREA AI (RISK-ON vs RISK-OFF) ---
                with c_heat2:
                    st.markdown("#### 🧠 Analiza AI a Fluxului")
                    
                    # Extragem extremele (Păstrăm tot Dataframe-ul pentru a avea și procentele)
                    top_3 = df_sectors.tail(3).iloc[::-1] # Cele mai mari 3, inversate să fie cel mai mare sus
                    bottom_3 = df_sectors.head(3) # Cele mai mici 3
                    
                    # Definim categoriile de risc
                    defensive = ['Utilități', 'Consum de Bază', 'Sănătate', 'Imobiliare']
                    aggressive = ['Tehnologie', 'Consum Discreționar', 'Comunicații', 'Financiar']
                    
                    def_count = sum(1 for s in top_3['Sector'] if s in defensive)
                    agg_count = sum(1 for s in top_3['Sector'] if s in aggressive)
                    
                    if agg_count >= 2:
                        st.success("**🚀 RISK-ON (Atac):**\nBanii intră masiv în sectoarele de creștere. Investitorii sunt optimiști.")
                    elif def_count >= 2:
                        st.error("**🛡️ RISK-OFF (Apărare):**\nBanii fug spre sectoarele de siguranță (defensive). Instituțiile se protejează.")
                    else:
                        st.warning("**🔄 ROTAȚIE MIXTĂ:**\nFără direcție clară de risc. Capitalul se mută la nivel individual.")
                        
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
                    st.plotly_chart(fig_bonds, use_container_width=True)
                    
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
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
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
                use_container_width=True, hide_index=True
            )
            
        with col_m2:
            st.subheader("🛢️ Mărfuri (Commodities)")
            st.dataframe(
                df_comm.style.map(color_change_val, subset=['Variație', 'Variație %'])
                .format({'Preț': '{:.2f}', 'Variație': '{:.2f}', 'Variație %': '{:.2f}%'}),
                use_container_width=True, hide_index=True
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
                    use_container_width=True, hide_index=True
                )
        
        with c_us2:
            st.markdown("**🔻 Top Scăderi (Losers)**")
            if not us_lose.empty:
                st.dataframe(
                    us_lose[['Instrument', 'Preț', 'Variație %']].style
                    .map(color_change_val, subset=['Variație %'])
                    .format({'Preț': '{:.2f}', 'Variație %': '{:.2f}%'}),
                    use_container_width=True, hide_index=True
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
                    use_container_width=True, hide_index=True
                )
        
        with c_eu2:
            st.markdown("**🔻 Top Scăderi (Losers)**")
            if not eu_lose.empty:
                st.dataframe(
                    eu_lose[['Instrument', 'Preț', 'Variație %']].style
                    .map(color_change_val, subset=['Variație %'])
                    .format({'Preț': '{:.2f}', 'Variație %': '{:.2f}%'}),
                    use_container_width=True, hide_index=True
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
            sheet = connect_to_gsheets()
            if not sheet: return pd.DataFrame()
            try:
                ws = sheet.spreadsheet.worksheet(sheet_name)
                # Folosim get_all_values pt a evita erorile de header duplicate la citire
                data = ws.get_all_values() 
                if len(data) < 2: return pd.DataFrame()
                # Transformăm în DataFrame folosind primul rând ca header
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
                        final_df, height=600, use_container_width=True,
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
                        display_df, height=600, use_container_width=True,
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
        
        now = datetime.now()
        current_hour = now.hour
        
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
            
            # 2. Top Movers (5 companii)
            col_mov1, col_mov2 = st.columns(2)
            
            with col_mov1:
                st.markdown("**🚀 Top 5 Creșteri**")
                if not gainers.empty:
                    st.dataframe(
                        gainers[['Simbol', 'Preț', 'Variație']].style
                        .format({'Preț': '{:.2f}', 'Variație': '{:+.2f}%'})
                        .map(lambda x: 'color: #3FB950', subset=['Variație']),
                        use_container_width=True, hide_index=True
                    )
                else: st.info("Date indisponibile.")
                
            with col_mov2:
                st.markdown("**🔻 Top 5 Scăderi**")
                if not losers.empty:
                    st.dataframe(
                        losers[['Simbol', 'Preț', 'Variație']].style
                        .format({'Preț': '{:.2f}', 'Variație': '{:+.2f}%'})
                        .map(lambda x: 'color: #F85149', subset=['Variație']),
                        use_container_width=True, hide_index=True
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
                    use_container_width=True, hide_index=True
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
            if current_hour < 16:
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
            
            c_us1, c_us2 = st.columns(2)
            with c_us1:
                st.markdown("**🚀 Top Creșteri (Big Caps)**")
                if not us_gainers.empty:
                    st.dataframe(
                        us_gainers[['Simbol', 'Preț', 'Variație']].style
                        .format({'Preț': '${:.2f}', 'Variație': '{:+.2f}%'})
                        .map(lambda x: 'color: #3FB950', subset=['Variație']),
                        use_container_width=True, hide_index=True
                    )
            
            with c_us2:
                st.markdown("**🔻 Top Scăderi (Big Caps)**")
                if not us_losers.empty:
                    st.dataframe(
                        us_losers[['Simbol', 'Preț', 'Variație']].style
                        .format({'Preț': '${:.2f}', 'Variație': '{:+.2f}%'})
                        .map(lambda x: 'color: #F85149', subset=['Variație']),
                        use_container_width=True, hide_index=True
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
                'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'HAL', 'MPC', 'DVN', 'UUUU', 'OKLO', 'VLO', 'T',
                'CAT', 'DE', 'BA', 'LMT', 'RTX', 'GD', 'NOC', 'GE', 'MMM', 'HON', 'UNP', 'NVO', 'AMGN', 'BIIB', 'SNY', 'NVS',
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
                'VIV.PA', 'ENG.PA', 'LR.PA', 'HO.PA', 'ML.PA', 'DGE.L', 'SU.PA', 'HO.PA', 'RI.PA', 'BN.PA', 'DSY.PA'
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
                            if "ANOMALIE" in str(row['Alertă AI']):
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
                            use_container_width=True, 
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
                target = float(row['TargetPrice'])
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
                use_container_width=True,
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
