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
        "Tehnologie": ["tehnologie", "tech", "it", "ai", "software", "hardware", "digital", "cyber", "apple", "microsoft", "google", "nvidia", "oracle", "amazon", "adobe", "asml", "tsm", "palantir", "qualcomm", "Micron", "AMD", "Meta", "Broadcom", "intel", "innodata", "crypto", "blockchain"],
        "Energie": ["energie", "petrol", "gaze", "oil", "WTI", "energy", "curent", "hidroelectrica", "omv", "romgaz", "nuclearelectrica", "electrica", "simtel", "transelectrica", "transgaz", "regenerabil", "eolian", "occidental petroleum", "exxon", "chevron", "devon", "centrus energy", "conocophillips", "LNG", "OKLO", "Shell", "vistra", "Totalenergies", "nuscale power", "fotovoltaic"],
        "Financiar": ["banca", "bank", "credit", "bursa", "finante", "fonduri", "asigurari", "bvb", "fiscal", "profit", "taxe", "buget", "wall street", "brd", "banca transilvania", "aig", "bac", "wfc", "JPM", "BNP", "unicredit", "UBS", "Deutsche Bank", "MS", "GS", "BLK", "actiuni"],
        "Farma": ["farma", "pharma", "sanatate", "medicament", "spital", "medical", "pfizer", "nvo", "sanofi", "eli lilly", "novartis", "antibiotice", "BIO", "Merk", "Biogen", "Biontech", "Medicover", "bayer", "sanofy", "Unitedhealth", "J&J", "medlife", "regina maria"],
        "Militar": ["militar", "aparare", "defense", "armata", "razboi", "nato", "arme", "securitate", "geopolitic", "taiwan", "Lockheed Martin", "raytheon", "Bae Systems", "Leonardo", "rocket lab", "thales", "vinci", "red cat", "eutelsat", "rheinmetall", "ucraina", "rusia"],
        "Alimentatie": ["alimentatie", "food", "retail", "agricultura", "horeca", "supermarket", "bauturi", "preturi alimente", "DPZ", "KO", "MCD", "PM", "P&G", "Colgate", "Pepsi", "Walmart", "carrefour", "lidl", "kaufland"],
        "Calatorii": ["turism", "calatorii", "travel", "aviatie", "aeroport", "hotel", "transport", "tarom", "airbus", "boeing", "Delta", "Royal Caribbean", "Marriot", "United Airlines", "wizz", "vacanta", "zbor"],
        "Constructii": ["constructii", "imobiliare", "impact developer", "ONE united properties", "real estate", "santier", "dezvoltator", "locuinte", "ciment", "infrastructura", "drumuri", "autostrada"],
        "Auto": ["auto", "masini", "ev", "electric", "dacia", "ford", "tesla", "volkswagen", "bmw", "mercedes", "automotive", "BYD", "Xpeng", "Nio", "Toyota", "Audi", "Ferrari", "inmatriculari"],
        "Aur/Metale": ["aur", "gold", "argint", "silver", "metal", "cupru", "precious", "aluminiu", "Ramaco Resources", "rio tinto", "BHP", "Critical Matals", "Glencore", "USA Rare Earth", "MP Materials", "otel"],
        "Marfuri": ["marfuri", "commodities", "materii prime", "grau", "porumb", "cacao", "soia", "prime materials", "gas", "cafea", "culturi"],
        "Dobânzi": ["dobanda", "robor", "ircc", "interest", "inflation", "inflatie", "banci centrale", "FED", "BCE", "BNR"],
        "Asia": ["BOJ", "Japonia", "China", "Taiwan", "Nikkei", "Topix", "Hang Seng", "banca japoniei", "guvernul japoniei", "Bank of China", "Shanghai Composite", "Bank of Japan", "Nifty", "Beijing", "yuan", "India", "yen", "KOSPI", "Nipah"],
        "Șomaj": ["somaj", "locuri de munca", "salarii", "unemployment", "jobs", "angajari", "PPI", "PCE", "CPI", "PMI", "NFP", "HR", "munca", "forta de munca"]
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
    if not keywords: return all_news
    filtered = []
    for item in all_news:
        text_full = (item['title'] + " " + item['summary']).lower()
        if any(k in text_full for k in keywords):
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
    bvb_data = yf.download(bvb_tickers, period="5d", group_by='ticker', progress=False)
    
    us_tickers = [
        '^GSPC', '^DJI', '^IXIC', '^VIX', 
        'NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'CG', 'SNOW', 'CEG', 'ASML', 'ARM', 'CRWV', 'FN', 'SNDK', 'MU', 
        'AMD', 'INTC', 'NFLX', 'JPM', 'BAC', 'SOFI', 'MS', 'HON', 'V', 'T', 'INOD', 'MA', 'MDB', 'AIG', 'AXP', 'SCHW', 'NET', 'BIIB', 
        'WMT', 'KO', 'PEP', 'PG', 'DXCM', 'COP', 'OXY', 'DVN', 'LNG', 'UUUU', 'FSLR', 'TTE', 'RIO', 'BHP', 'D', 'VALE', 'METC', 'MP', 'LLY', 'AMGN', 'XOM', 'CVX', 
        'PLTR', 'PANW', 'ANET', 'QCOM', 'ORCL', 'TSM', 'GS', 'CRM', 'WFC', 'NVO', 'NVS', 'MCD', 'SMR', 'OKLO', 'SNY', 'JNJ', 'BA', 'GD', 'RTX', 'LMT', 'KTOS', 'PM', 'COO', 'MRK', 'PFE', 'C'
    ]
    us_data = yf.download(us_tickers, period="5d", group_by='ticker', progress=False)
    
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
    """Calculează sentimentul pieței locale bazat pe deviația Indicelui BET."""
    try:
        # Preluăm datele pentru TVBETETF.RO (Proxy pentru BET)
        if isinstance(bvb_data.columns, pd.MultiIndex):
             bet_series = bvb_data['TVBETETF.RO']['Close'].dropna()
        else:
             return 50, "Neutral ⚖️"

        if bet_series.empty: return 50, "Neutral ⚖️"

        curr_bet = bet_series.iloc[-1]
        mean_bet = bet_series.mean() # Media ultimelor 5 zile
        
        # Calculăm deviația procentuală
        dev = ((curr_bet / mean_bet) - 1) * 100
        
        # Mapare pe un scor 0-100 (Proxy Fear & Greed BVB)
        # O deviație de +/- 2% în 5 zile este considerată extremă pentru BVB
        score = 50 + (dev * 25) 
        score = max(0, min(100, score))

        if score >= 75: label = "Optimism Excesiv 🚀"
        elif score >= 55: label = "Sentiment Pozitiv 📈"
        elif score >= 45: label = "Echilibru ⚖️"
        elif score >= 25: label = "Precauție ⚠️"
        else: label = "Panică / Sărituri 🚨"
        
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
                 'PLTR', 'PANW', 'ANET', 'QCOM', 'ORCL', 'TSM', 'GS', 'CRM', 'WFC', 'NVO', 'NVS', 'MCD', 'SMR', 'OKLO', 'SNY', 'JNJ', 'BA', 'GD', 'RTX', 'LMT', 'KTOS', 'PM', 'COO', 'MRK', 'PFE', 'C']
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
        "8. Watchlist 🎯" 
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
            
        if hist is None or hist.empty:
            st.error("Simbol invalid sau date indisponibile.")
        else:
            # 1. Informații Generale
            st.markdown(f"## {info.get('longName', real_sym)}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Sector", info.get('sector', 'N/A'))
            c2.metric("Industrie", info.get('industry', 'N/A'))
            c3.metric("Capitalizare", format_num(info.get('marketCap')))
            st.markdown("---")

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
            st.markdown("---")

            # 5. Calculator Fair Value (REPARAT - UNIC)
            st.subheader("🧮 Calculator Valoare Intrinsecă (Valoare justă)")
            
            # Date fundamentale unice pentru acest bloc
            eps_f = info.get('trailingEps', 0)
            bv_f = info.get('bookValue', 0)
            curr_f = info.get('currentPrice') or info.get('previousClose', 0)
            t_curr = info.get('currency', 'USD')

            st.write("⚙️ **Ajustează Ipotezele DCF**")
            c_sl1, c_sl2 = st.columns(2)
            u_growth = c_sl1.slider("Creștere Anuală EPS (%)", -10.0, 40.0, 10.0, key="slider_g_final")
            u_discount = c_sl2.slider("Rata de Scont (Risc %)", 5.0, 15.0, 9.0, key="slider_d_final")
            
            # Calcul reactiv 100%
            graham_calc = np.sqrt(22.5 * max(0, eps_f) * max(0, bv_f)) if (eps_f > 0 and bv_f > 0) else 0
            dcf_calc = calculate_dcf_dynamic(info, u_growth, u_discount)

            if curr_f > 0:
                cv1, cv2, cv3 = st.columns(3)
                box_css = "border: 2px solid {col}; padding: 15px; border-radius: 12px; text-align: center; background-color: #161B22; height: 160px; display: flex; flex-direction: column; justify-content: center;"
                
                with cv1:
                    st.markdown(f'<div style="{box_css.format(col="#30363D")}"><p style="margin:0;color:#8B949E;font-size:14px;">Preț Curent</p><h2 style="margin:10px 0;color:white;">{curr_f:.2f} {t_curr}</h2></div>', unsafe_allow_html=True)
                with cv2:
                    diff_g = ((curr_f - graham_calc) / graham_calc) * 100 if graham_calc > 0 else 0
                    g_col = "#3FB950" if curr_f < graham_calc else "#F85149"
                    st.markdown(f'<div style="{box_css.format(col=g_col)}"><p style="margin:0;color:#8B949E;font-size:14px;">Benjamin Graham</p><h2 style="margin:10px 0;color:{g_col};">{graham_calc:.2f}</h2><p style="margin:0;color:{g_col};font-weight:bold;font-size:12px;">{"SUBEVALUAT" if curr_f < graham_calc else "SUPRAEVALUAT"} ({abs(diff_g):.1f}%)</p></div>', unsafe_allow_html=True)
                with cv3:
                    diff_d = ((curr_f - dcf_calc) / dcf_calc) * 100 if dcf_calc > 0 else 0
                    d_col = "#3FB950" if curr_f < dcf_calc else "#F85149"
                    st.markdown(f'<div style="{box_css.format(col=d_col)}"><p style="margin:0;color:#8B949E;font-size:14px;">Fair Value (DCF)</p><h2 style="margin:10px 0;color:{d_col};">{dcf_calc:.2f}</h2><p style="margin:0;color:{d_col};font-weight:bold;font-size:12px;">{"SUBEVALUAT" if curr_f < dcf_calc else "SUPRAEVALUAT"} ({abs(diff_d):.1f}%)</p></div>', unsafe_allow_html=True)
            st.markdown("---")

            # 6. Terminal Intelligence AI (SENTIMENT & PROGNOZĂ)
            st.subheader("🤖 Terminal Intelligence (AI & ML)")
            c_news_ai = get_company_news_rss(real_sym)
            cai1, cai2 = st.columns([1, 2])
            
            with cai1:
                st.write("📊 **Analiză Sentiment (FinBERT)**")
                if c_news_ai:
                    with st.spinner("AI-ul analizează contextul..."):
                        from ai_engine import analyze_sentiment_ai
                        s_score = analyze_sentiment_ai(c_news_ai)
                        c_ai = "#3FB950" if s_score > 0.1 else "#F85149" if s_score < -0.1 else "#8B949E"
                        st.markdown(f"<div style='background:#161B22; padding:20px; border-radius:15px; border:1px solid {c_ai}; text-align:center;'><h1 style='color:{c_ai}; margin:0;'>{s_score:.2f}</h1><p style='color:#8B949E;'>Sentiment Scor</p></div>", unsafe_allow_html=True)
            
            with cai2:
                st.write("📈 **Prognoză Algoritmică (Next 30 Days)**")
                if len(hist) > 100:
                    from ai_engine import predict_stock_price, render_ai_chart
                    forecast = predict_stock_price(hist)
                    render_ai_chart(forecast, hist)
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
            
            tab_usd, tab_eur = st.tabs(["🇺🇸 Portofoliu USD", "🇪🇺 Portofoliu EUR"])

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
                else:
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
                else:
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
                        st.plotly_chart(fig_sym, use_container_width=True)

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
                        st.plotly_chart(fig_sec, use_container_width=True)

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
                            'Quantity': '{:.4f}', 'AvgPrice': '{:.2f}', 'CurrentPrice': '{:.2f}',
                            'MarketValue': '{:,.2f}', 'Profit': '{:,.2f}', 'Profit %': '{:.2f}%'
                        }),
                        use_container_width=True
                    )
                
                st.markdown("<br>", unsafe_allow_html=True) 

            with tab_usd:
                df_usd = df_pf[df_pf['Currency'] == 'USD']
                render_portfolio_tab(df_usd, "$")

            with tab_eur:
                df_eur = df_pf[df_pf['Currency'] == 'EUR']
                render_portfolio_tab(df_eur, "€")

            # Butonul de reset nu poate șterge datele din Google Drive, doar le ignoră temporar
            # Așa că l-am comentat sau ar trebui scos, deoarece gestionarea datelor se face acum în Sheets.
            # st.markdown("---")
            # if st.button("⚠️ Șterge TOT Portofoliul (Reset)"):
            #     os.remove(FILE_PORTOFOLIU)
            #     st.rerun()

    # ==================================================
    # 4. PIAȚĂ GLOBALĂ (CU DASHBOARD MACRO - DUAL METRICS)
    # ==================================================
    elif sectiune == "4. Piață Globală":
        st.title("🌐 Pulsul Pieței Globale")
        st.caption("Date în timp real (cu întârziere minimă) furnizate via Yahoo Finance.")
        
        # Buton refresh global (pentru macro + actiuni)
        if st.button("🔄 Reîmprospătează Piața"):
            get_global_market_data.clear()
            get_macro_data_visuals.clear()
            st.rerun()

        # --- DASHBOARD MACROECONOMIC (PARTEA NOUĂ) ---
        st.markdown("### 🧭 Indicatori Macroeconomici")
        st.info("💡 **Interpretare:** Dacă **US 10Y Yield** crește brusc, acțiunile de tehnologie (Growth) tind să scadă. Dacă **Aurul** crește, indică frică în piață.")
        
        # Apelăm funcția (acum descarcă 5 ani)
        macro_tickers, macro_data = get_macro_data_visuals()
        
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

            else:
                st.warning("Date indisponibile sau eroare conexiune Yahoo.")

        st.markdown("---")
        # ---------------------------------------------------------
        
        # --- TABELELE VECHI ---
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
                'JNJ', 'LLY', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO', 'MP', 'METC', 'RIO', 'BHP', 'AEM', 'DHR', 'BMY', 'CVS'
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
        
        # Funcție internă de calcul RVOL
        def get_rvol_data(ticker_list):
            try:
                # Descărcăm date pe 2 luni pentru a avea o medie solidă
                data = yf.download(ticker_list, period="2mo", group_by='ticker', progress=False)
                results = []
                
                for t in ticker_list:
                    try:
                        # Gestionare MultiIndex vs Single Index
                        if isinstance(data.columns, pd.MultiIndex):
                            if t not in data.columns.levels[0]: continue
                            df_t = data[t]
                        else:
                            df_t = data # Cazul unui singur ticker (rar aici)
                        
                        # Avem nevoie de Volum și Close
                        vol = df_t['Volume'].dropna()
                        close = df_t['Close'].dropna()
                        
                        if len(vol) < 25: continue # Nu avem destule date
                        
                        # 1. Volumul de AZI
                        curr_vol = vol.iloc[-1]
                        
                        # 2. Media pe ultimele 20 zile (fără azi)
                        avg_vol_20 = vol.iloc[-21:-1].mean()
                        
                        # FILTRU ZGOMOT: Ignorăm dacă media e sub 5000 unități
                        if avg_vol_20 < 5000: continue
                        
                        # 3. Calcul RVOL
                        rvol = curr_vol / avg_vol_20
                        
                        # 4. Calcul Variație Preț
                        curr_p = close.iloc[-1]
                        prev_p = close.iloc[-2]
                        change_pct = ((curr_p - prev_p) / prev_p) * 100
                        
                        results.append({
                "Simbol": t.replace('.RO', ''),
                "Preț": curr_p,
                "Variație %": change_pct,
                "Volum Azi": curr_vol,
                "Volum Mediu (20z)": avg_vol_20,
                "RVOL": rvol,
                # --- NOU: Clasificare profesională (Corectată) ---
                "Status": "🚀 BREAKOUT" if (rvol > 2.0 and change_pct > 1.5) 
                         else ("⚠️ PANIC SELL" if (rvol > 2.0 and change_pct < -1.5) 
                         else ("✅ ACUMULARE" if (rvol > 1.2 and change_pct > 0) else "Normal"))
            }) # Linia 2071 acum închide corect dicționarul și append-ul
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
                        # --- FUNCȚIE DE COLORARE PROFESIONALĂ (MODIFICATĂ) ---
                        def style_scanner_rows(row):
                            if "BREAKOUT" in row['Status']:
                                # Verde aprins pentru oportunități imediate
                                return ['background-color: rgba(63, 185, 80, 0.4); font-weight: bold'] * len(row)
                            elif "ACUMULARE" in row['Status']:
                                # Verde pal pentru acumulare discretă
                                return ['background-color: rgba(63, 185, 80, 0.15)'] * len(row)
                            elif "PANIC" in row['Status']:
                                # Roșu pentru vânzare masivă
                                return ['background-color: rgba(248, 81, 73, 0.25)'] * len(row)
                            return [''] * len(row)

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
    elif sectiune == "8. Watchlist 🎯":
        st.title("🎯 Lista de Urmărire (Watchlist)")
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




