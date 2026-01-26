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

def calculate_intrinsic_value(info):
    """
    Calculează valoarea intrinsecă cu limite de siguranță (Capping).
    """
    try:
        current_price = info.get('currentPrice') or info.get('previousClose')
        eps = info.get('trailingEps')
        book_value = info.get('bookValue')
        
        # --- 1. FORMULA BENJAMIN GRAHAM ---
        graham_val = 0
        if eps is not None and book_value is not None:
            if eps > 0 and book_value > 0:
                graham_val = np.sqrt(22.5 * eps * book_value)
            
        # --- 2. DCF SIMPLIFICAT (Cu SANITY CHECK) ---
        # Pas critic: Limităm creșterea. Nicio companie matură nu crește cu 50% pe an 5 ani la rând.
        # Yahoo poate returna valori mari, noi le plafonăm la 15% (0.15) pentru siguranță.
        raw_growth = info.get('earningsGrowth')
        
        # Logică de siguranță pentru Growth Rate
        if raw_growth is None:
            growth_rate = 0.05  # 5% conservator dacă nu avem date
        else:
            # Plafonăm la maxim 15% (0.15) sau folosim valoarea reală dacă e mai mică
            growth_rate = min(raw_growth, 0.15) 
            
            # Dacă Yahoo dă creștere negativă, folosim un minim de 2% pentru inflație
            if growth_rate < 0.02: growth_rate = 0.02

        discount_rate = 0.09 # 9% costul capitalului (standard industrie)
        terminal_multiple = info.get('trailingPE', 15) # Folosim P/E actual, dar nu mai mult de 25
        terminal_multiple = min(terminal_multiple, 25) 
        
        dcf_val = 0
        if eps is not None and eps > 0:
            future_cash_flows = []
            for i in range(1, 6): # 5 ani
                fcf = eps * ((1 + growth_rate) ** i)
                discounted_fcf = fcf / ((1 + discount_rate) ** i)
                future_cash_flows.append(discounted_fcf)
            
            # Valoare terminală
            terminal_val = (eps * ((1 + growth_rate) ** 5)) * terminal_multiple
            discounted_terminal = terminal_val / ((1 + discount_rate) ** 5)
            
            dcf_val = sum(future_cash_flows) + discounted_terminal
            
        return graham_val, dcf_val, current_price
    except Exception as e:
        return 0, 0, 0
    
@st.cache_data(ttl=900)
def get_stock_data(symbol):
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="5y")
        if hist.empty and not symbol.endswith(".RO"):
            sym_ro = symbol + ".RO"
            t_ro = yf.Ticker(sym_ro)
            hist_ro = t_ro.history(period="5y")
            if not hist_ro.empty:
                return hist_ro, t_ro.info, getattr(t_ro, 'earnings_history', None), sym_ro
        return hist, t.info, getattr(t, 'earnings_history', None), symbol
    except: return None, None, None, symbol

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
    """
    Generează o matrice de corelare optimizată cu INTERPRETARE TEXTUALĂ.
    Schimbă culorile: 
    - Albastru = RISC (Corelare Mare)
    - Verde = SIGURANȚĂ (Diversificare/Hedge)
    """
    if len(tickers) < 2: return None
    try:
        # Descărcare date
        data = yf.download(tickers, period="1y", progress=False)['Close']
        returns = data.pct_change().dropna()
        corr_matrix = returns.corr()
        
        # Pregătire matrice TEXT pentru afișare (Valoare + Interpretare)
        text_matrix = []
        for row_idx in range(len(corr_matrix)):
            row_text = []
            for col_idx in range(len(corr_matrix)):
                val = corr_matrix.iloc[row_idx, col_idx]
                
                # Logică interpretare (Threshold-uri profesionale)
                if row_idx == col_idx:
                    label = "Identic"
                    display_text = "" # Lăsăm gol pe diagonală
                elif val > 0.8:
                    label = "RISC (Duplicat)"
                    display_text = f"{val:.2f}<br>⚠️ {label}"
                elif val > 0.5:
                    label = "Corelat (Risc Mediu)"
                    display_text = f"{val:.2f}<br>{label}"
                elif val > 0.2:
                    label = "Moderat"
                    display_text = f"{val:.2f}<br>{label}"
                elif val > -0.2:
                    label = "Diversificat (BUN)"
                    display_text = f"{val:.2f}<br>✅ {label}"
                else:
                    label = "Hedge (Protecție)"
                    display_text = f"{val:.2f}<br>🛡️ {label}"
                
                row_text.append(display_text)
            text_matrix.append(row_text)

        # --- CONFIGURARE CULORI ---
        # 0.0 (Minim, -1) -> Verde lime (Siguranță)
        # 0.5 (Mijloc, 0) -> Alb (Neutru)
        # 1.0 (Maxim, 1)  -> Albastru (Risc)
        custom_colorscale = [
            [0.0, 'rgb(0, 255, 0)'],   # -1: Verde lime
            [0.5, 'rgb(255, 255, 255)'], #  0: Alb
            [1.0, 'rgb(1, 70, 77)']      # +1: Albastru
        ]

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale=custom_colorscale, 
            zmin=-1, zmax=1,
            text=text_matrix,
            texttemplate="%{text}",
            hovertemplate="<b>%{x} vs %{y}</b><br>Coeficient: %{z:.2f}<br><extra></extra>",
            showscale=True,
            colorbar=dict(title="Nivel Risc")
        ))
        
        fig.update_layout(
            title='Matrice Diversificare (Albastru = Risc Mare | Verde = Siguranță)',
            height=550,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )
        return fig
    except Exception as e: return None

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
        'SMTL.RO', 'AROBS.RO', 'AQ.RO', 'ARS.RO', 'BRK.RO', 'IARV.RO', 'TTS.RO', 'WINE.RO', 'TEL.RO', 'DN.RO', 'AG.RO', 
        'BENTO.RO', 'PE.RO', 'COTE.RO', 'PBK.RO', 'SAFE.RO', 'TBK.RO', 'CFH.RO', 'SFG.RO'
    ]
    bvb_data = yf.download(bvb_tickers, period="5d", group_by='ticker', progress=False)
    
    us_tickers = [
        '^GSPC', '^DJI', '^IXIC', '^VIX', 
        'NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'CG', 'SNOW', 'CEG', 'ASML', 'ARM', 'CRWV', 'FN', 'SNDK', 'MU', 
        'AMD', 'INTC', 'NFLX', 'JPM', 'BAC', 'SOFI', 'MS', 'HON', 'V', 'INOD', 'MA', 'MDB', 'AIG', 'AXP', 'SCHW', 'NET', 'BIIB', 
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
    
    # Asigurăm conversia tipurilor (Google Sheets poate returna string-uri uneori)
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)
    df['AvgPrice'] = pd.to_numeric(df['AvgPrice'], errors='coerce').fillna(0)
    
    tickers = df['Symbol'].unique().tolist()
    if not tickers:
        return pd.DataFrame(), pd.DataFrame(), 0, 0
        
    hist_data = yf.download(tickers, period="5y", group_by='ticker', auto_adjust=True, threads=True)
    
    current_vals = []
    total_daily_pl_abs = 0 
    
    for index, row in df.iterrows():
        sym = row['Symbol']
        qty = row['Quantity']
        avg_p = row['AvgPrice']
        
        try:
            if len(tickers) > 1:
                series = hist_data[sym]['Close']
            else:
                if isinstance(hist_data.columns, pd.MultiIndex):
                      series = hist_data[sym]['Close']
                else:
                      series = hist_data['Close']
            
            series = series.dropna()
            
            if not series.empty:
                curr_p = series.iloc[-1]
                prev_p = series.iloc[-2] if len(series) >= 2 else curr_p
            else:
                curr_p = 0
                prev_p = 0
        except Exception as e:
            curr_p = 0
            prev_p = 0
            
        mkt_val = qty * curr_p
        inv_val = qty * avg_p
        profit = mkt_val - inv_val
        profit_pct = (profit / inv_val * 100) if inv_val != 0 else 0
        
        daily_change = (curr_p - prev_p) * qty
        total_daily_pl_abs += daily_change
        
        current_vals.append({
            'Symbol': sym, 'Quantity': qty, 'AvgPrice': avg_p, 'CurrentPrice': curr_p,
            'MarketValue': mkt_val, 'Profit': profit, 'Profit %': profit_pct
        })
    
    df_result = pd.DataFrame(current_vals)
    
    portfolio_curve = None
    for index, row in df.iterrows():
        sym = row['Symbol']
        qty = row['Quantity']
        try:
            if len(tickers) > 1:
                price_series = hist_data[sym]['Close']
            else:
                if isinstance(hist_data.columns, pd.MultiIndex):
                      price_series = hist_data[sym]['Close']
                else:
                      price_series = hist_data['Close']
                      
            # FIX: Updated Pandas methods
            price_series = price_series.ffill().bfill()
            
            if portfolio_curve is None:
                portfolio_curve = price_series * qty
            else:
                portfolio_curve = portfolio_curve.add(price_series * qty, fill_value=0)
        except: pass
    
    if portfolio_curve is None:
        portfolio_curve = pd.Series()

    days_map = {"1Z": 2, "1S": 7, "1L": 30, "3L": 90, "6L": 180, "1A": 365, "3A": 1095, "5A": 1825}
    days = days_map.get(history_range, 365)
    portfolio_curve = portfolio_curve.iloc[-days:]
    
    total_val_now = portfolio_curve.iloc[-1] if not portfolio_curve.empty else 0
    total_daily_pl_pct = (total_daily_pl_abs / (total_val_now - total_daily_pl_abs) * 100) if (total_val_now - total_daily_pl_abs) != 0 else 0
    
    return df_result, portfolio_curve, total_daily_pl_abs, total_daily_pl_pct

def calculate_risk_metrics(portfolio_curve):
    """Calculează Max Drawdown și Sharpe Ratio."""
    if portfolio_curve.empty: return 0, 0
    
    # 1. Max Drawdown (Cea mai mare durere istorică)
    # Calculăm maximul atins până în fiecare punct (Running Max)
    rolling_max = portfolio_curve.cummax()
    # Calculăm cât suntem sub maxim în fiecare zi
    drawdown = (portfolio_curve - rolling_max) / rolling_max
    # Luăm cea mai negativă valoare (ex: -0.20 înseamnă -20%)
    max_dd = drawdown.min()
    
    # 2. Sharpe Ratio (Rentabilitate vs Risc)
    # Calculăm randamentele zilnice
    daily_rets = portfolio_curve.pct_change().dropna()
    if daily_rets.std() == 0: return max_dd, 0
    
    # Formula anualizată (presupunem Risk Free Rate ~ 4%)
    rf_daily = 0.04 / 252
    excess_ret = daily_rets - rf_daily
    sharpe = np.sqrt(252) * (excess_ret.mean() / daily_rets.std())
    
    return max_dd, sharpe

@st.cache_data(ttl=3600)
def get_portfolio_sectors(df_current):
    """Grupează valoarea portofoliului pe sectoare economice."""
    if df_current.empty: return pd.DataFrame()
    
    sector_map = {}
    
    for _, row in df_current.iterrows():
        sym = row['Symbol']
        val = row['MarketValue']
        
        # Încercăm să aflăm sectorul
        try:
            # Folosim fast_info sau info (optimizat)
            t = yf.Ticker(sym)
            sec = t.info.get('sector', 'Nedefinit')
        except:
            sec = 'Nedefinit'
            
        sector_map[sec] = sector_map.get(sec, 0) + val
        
    # Convertim în DataFrame pentru grafic
    df_sec = pd.DataFrame(list(sector_map.items()), columns=['Sector', 'MarketValue'])
    return df_sec

# --- FUNCȚIE GLOBAL MARKET ---
@st.cache_data(ttl=300)
def get_global_market_data():
    indices = {
        'S&P 500': '^GSPC', 'Dow Jones': '^DJI', 'Nasdaq': '^IXIC', 
        'DAX (GER)': '^GDAXI', 'FTSE 100 (UK)': '^FTSE', 'BET (RO)': 'BET.RO'
    }
    commodities = {
        'Aur (Gold)': 'GC=F', 'Argint (Silver)': 'SI=F', 
        'Petrol (WTI)': 'CL=F', 'Petrol (Brent)': 'BZ=F', 'Gaz Natural': 'NG=F'
    }
    
    us_stocks = ['NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'CG', 'SNOW', 'CEG', 'ASML', 'ARM', 'CRWV', 'FN', 'SNDK', 'MU', 
                 'AMD', 'INTC', 'NFLX', 'JPM', 'BAC', 'SOFI', 'MS', 'HON', 'V', 'INOD', 'MA', 'MDB', 'AIG', 'AXP', 'SCHW', 'NET', 'BIIB', 
                 'WMT', 'KO', 'PEP', 'PG', 'DXCM', 'COP', 'OXY', 'DVN', 'LNG', 'UUUU', 'FSLR', 'TTE', 'RIO', 'BHP', 'D', 'VALE', 'METC', 'MP', 'LLY', 'AMGN', 'XOM', 'CVX', 
                 'PLTR', 'PANW', 'ANET', 'QCOM', 'ORCL', 'TSM', 'GS', 'CRM', 'WFC', 'NVO', 'NVS', 'MCD', 'SMR', 'OKLO', 'SNY', 'JNJ', 'BA', 'GD', 'RTX', 'LMT', 'KTOS', 'PM', 'COO', 'MRK', 'PFE', 'C']
    eu_stocks = ['SAP.DE', 'MC.PA', 'ASML', 'SIE.DE', 'TTE.PA', 'AIR.PA', 'ALV.DE', 'DTE.DE', 'VOW3.DE', 'BAYN.DE', 'UCG.IT', 'ENR.DE', 'DBK.DE', 'BNP.FR', 
                 'BMW.DE', 'BNP.PA', 'SAN.PA', 'OR.PA', 'GLE.FR', 'MBG.DE', 'BSP.DE', 'LDO.IT', 'RNO.FR', 'SHEL.L', 'RACE.IT', 'AZN.L', 'HSBA.L', 'FP.PA']

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
        "5. Import Date (CSV)", 
        "6. Rezumatul Zilei",
        "7. Scanner Volum (RVOL)",
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
    # 2. ANALIZĂ COMPANIE
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
            st.markdown(f"## {info.get('longName', real_sym)}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Sector", info.get('sector', 'N/A'))
            c2.metric("Industrie", info.get('industry', 'N/A'))
            c3.metric("Capitalizare", format_num(info.get('marketCap')))
            st.markdown("---")

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

            rows_needed = 1
            if show_rsi: rows_needed += 1
            if show_macd: rows_needed += 1
            row_heights = [0.6]
            if show_rsi: row_heights.append(0.2)
            if show_macd: row_heights.append(0.2)
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

            st.subheader("📊 Indicatori Fundamentali")
            beta_val = info.get('beta')
            alpha_val = calculate_alpha(hist, beta_val)
            
            de_ratio = info.get('debtToEquity')
            if de_ratio is not None:
                de_display = f"{de_ratio:.2f}%"
            else:
                de_display = "N/A"

            # --- BLOC METRICI COMPLET ȘI CORECTAT ---
            with st.container():
                c_eval, c_prof, c_indat, c_risc = st.columns(4)
                
                with c_eval:
                    st.markdown("**Evaluare & Dividende**")
                    pe_val = info.get('trailingPE')
                    pb_val = info.get('priceToBook')
                    
                    # --- 1. CALCUL DIVIDEND (MANUAL PENTRU PRECIZIE) ---
                    div_rate = info.get('dividendRate')
                    current_price = info.get('currentPrice') or info.get('previousClose') or info.get('regularMarketPreviousClose')
                    
                    if div_rate is not None and current_price is not None and current_price > 0:
                        yield_calc = (div_rate / current_price) * 100
                        div_display = f"{div_rate} {info.get('currency', '')} ({yield_calc:.2f}%)"
                    elif div_rate is not None:
                        div_display = f"{div_rate} {info.get('currency', '')}"
                    else:
                        div_display = "N/A"
                    # ---------------------------------------------------

                    if pe_val is not None and pb_val is not None:
                        gn_val = pe_val * pb_val
                        gn_display = f"{gn_val:.2f}"
                    else:
                        gn_display = "N/A"

                    # Afișare metrici (Ordinea completă)
                    st.metric("P/E Ratio", format_num(pe_val), help="Cât plătești pentru 1$ profit.")
                    st.metric("Forward P/E", format_num(info.get('forwardPE')), help="P/E estimat pentru anul viitor.")
                    st.metric("Dividend (Randament)", div_display, help="Dividendul anual și randamentul real (Div/Preț).")
                    st.metric("P/BV", format_num(pb_val), help="Preț față de valoarea contabilă.")
                    st.metric("GN (Graham)", gn_display, help="Produsul P/E * P/BV.")
                    st.metric("EPS", format_num(info.get('trailingEps')), help="Profit net pe acțiune.")
                    st.metric("Val. Contabilă/Acțiune", format_num(info.get('bookValue')), help="Valoarea activelor nete per acțiune.")

                with c_prof:
                    st.markdown("**Profitabilitate**")
                    st.metric("ROA", format_num(info.get('returnOnAssets'), True), help="Randamentul activelor.")
                    st.metric("ROE", format_num(info.get('returnOnEquity'), True), help="Randamentul capitalului propriu.")
                    st.metric("Marjă Netă", format_num(info.get('profitMargins'), True), help="Profit net din venituri.")
                    st.metric("Marjă Operațională", format_num(info.get('operatingMargins'), True), help="EBIT / Venituri.")
                
                with c_indat:
                    st.markdown("**Îndatorare**")
                    st.metric("Datorii/Capital", de_display, help="Datorii totale la capital propriu (>100% poate indica risc).")
                    st.metric("Current Ratio", info.get('currentRatio', 'N/A'), help="Active curente / Datorii curente.")
                    st.metric("Quick Ratio", info.get('quickRatio', 'N/A'), help="Lichiditate imediată.")
                
                with c_risc:
                    st.markdown("**Risc (Alpha & Beta)**")
                    st.metric("Beta", info.get('beta', 'N/A'), help="Volatilitatea față de piață.")
                    st.metric("Alpha (1Y)", format_num(alpha_val, True), help="Performanța peste piață (vs SPY).")
            
            st.markdown("---")
            st.subheader(f"📰 Ultimele Știri despre {real_sym}")
            company_news = get_company_news_rss(real_sym)
            if company_news:
                for n in company_news:
                    sentiment, css_cls, icon = get_sentiment(n['title'])
                    c_txt, c_imp = st.columns([5, 1])
                    with c_txt:
                        st.markdown(f"**[{n['title']}]({n['link']})**")
                        st.caption(f"{n['publisher']} • {n['date_str']}")
                    with c_imp:
                        st.markdown(f"<span class='{css_cls}'>{icon} {sentiment}</span>", unsafe_allow_html=True)
                    st.divider()
            else:
                st.info(f"Nu au fost găsite știri recente pe fluxul Yahoo pentru {real_sym}.")

            # --- MODUL NOU: CALCULATOR EVALUARE ---
            st.subheader("🧮 Calculator Valoare Intrinsecă (Fair Value)")
            
            graham, dcf, curr_p = calculate_intrinsic_value(info)
            
            if curr_p and curr_p > 0:
                # Creăm 3 coloane vizuale
                c_val1, c_val2, c_val3 = st.columns(3)
                
                with c_val1:
                    st.markdown("#### Preț Curent")
                    st.markdown(f"<h2 style='color: #FFFFFF;'>{curr_p:.2f} {info.get('currency','')}</h2>", unsafe_allow_html=True)
                
                with c_val2:
                    st.markdown("#### Benjamin Graham")
                    if graham > 0:
                        diff_graham = ((curr_p - graham) / graham) * 100
                        color_g = "#F85149" if curr_p > graham else "#3FB950" # Rosu daca e scump, Verde daca e ieftin
                        status_g = "SUPRAEVALUAT" if curr_p > graham else "SUBEVALUAT"
                        st.markdown(f"<h2 style='color: {color_g};'>{graham:.2f}</h2>", unsafe_allow_html=True)
                        st.caption(f"{status_g} cu {abs(diff_graham):.1f}%")
                        st.info("Recomandat pentru: Bănci, Industrie, Energie (Active tangibile).")
                    else:
                        st.warning("Nu se poate calcula (EPS sau BV negativ).")

                with c_val3:
                    st.markdown("#### Model DCF (Growth)")
                    if dcf > 0:
                        diff_dcf = ((curr_p - dcf) / dcf) * 100
                        color_d = "#F85149" if curr_p > dcf else "#3FB950"
                        status_d = "SUPRAEVALUAT" if curr_p > dcf else "SUBEVALUAT"
                        st.markdown(f"<h2 style='color: {color_d};'>{dcf:.2f}</h2>", unsafe_allow_html=True)
                        st.caption(f"{status_d} cu {abs(diff_dcf):.1f}%")
                        st.info("Recomandat pentru: Tech, Servicii, Growth (Flux de numerar viitor).")
                    else:
                        st.warning("Date insuficiente pentru proiecție.")
                
                st.markdown("---")
            # --------------------------------------

            st.subheader("💰 Financiar & Raportări")
            st.markdown("""<div class="fin-card"><h4>Rezultate Financiare (Ultima Raportare)</h4></div>""", unsafe_allow_html=True)
            rev = info.get('totalRevenue')
            net_inc = info.get('netIncomeToCommon')
            cash = info.get('totalCash')
            
            exp = (rev - net_inc) if (rev and net_inc) else None
            
            c_f1, c_f2, c_f3, c_f4 = st.columns(4)
            c_f1.metric("Venituri Totale", format_num(rev))
            c_f2.metric("Profit Net", format_num(net_inc))
            c_f3.metric("Cheltuieli (Est.)", format_num(exp))
            c_f4.metric("Numerar Disponibil", format_num(cash))
            
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
                    score_clamped = max(1.0, min(5.0, rec_mean))
                    pos_pct = (score_clamped - 1.0) / 4.0 * 100.0
                    
                    st.markdown(f"""
                    <div style="margin-top:15px;">
                        <span style="font-size:14px; color:#ddd;">Scor Consens: <b>{rec_mean:.1f}</b></span>
                        <div class="analyst-bar-container">
                            <div class="analyst-bar-gradient"></div>
                            <div class="analyst-marker" style="left: {pos_pct}%;"></div>
                        </div>
                        <div class="analyst-labels">
                            <span>Strong Buy (1)</span>
                            <span>Hold (3)</span>
                            <span>Sell (5)</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.metric("Preț Țintă (Mediu)", f"{target} {info.get('currency','USD')}" if target else "N/A")
                
            with col_an_right:
                 st.markdown("""<div class="fin-card"><h4>🆚 Raportări vs Așteptări</h4></div>""", unsafe_allow_html=True)
                 if not earn_df.empty:
                    earn_display = earn_df[['epsEstimate', 'epsActual', 'epsDifference', 'surprisePercent']].copy()
                    earn_display.columns = ['Estimare', 'Realizat', 'Diferență', 'Surpriză %']
                    def color_surprise(val):
                        if pd.isna(val): return ''
                        color = '#3FB950' if val >= 0 else '#F85149'
                        return f'color: {color}'
                    st.dataframe(earn_display.style.map(color_surprise, subset=['Surpriză %']).format("{:.2f}"), use_container_width=True)
                 else:
                    st.info("Nu există date de earnings.")

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

        # Încărcăm datele din Google Sheets în loc de CSV local
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
                # --- MODUL NOU: ANALIZĂ DE RISC & SECTOARE ---
                # 1. Calculăm Metricile de Risc
                max_dd, sharpe = calculate_risk_metrics(hist_curve)
                
                st.markdown("#### 🛡️ Analiză Risc Portofoliu")
                c_risk1, c_risk2, c_risk3 = st.columns(3)
                
                # Max Drawdown
                c_risk1.metric(
                    "Max Drawdown (Scădere Max.)", 
                    f"{max_dd*100:.2f}%", 
                    help="Cea mai mare scădere procentuală înregistrată de portofoliu de la un maxim istoric până la minim. Un DD de -20% înseamnă că la un moment dat portofoliul a pierdut 20% din vârf."
                )
                
                # Sharpe Ratio
                c_risk2.metric(
                    "Sharpe Ratio", 
                    f"{sharpe:.2f}", 
                    help="Eficiența portofoliului. > 1 e Bun, > 2 e Excelent, < 0 e Rău. Indică cât profit faci pentru fiecare unitate de risc asumată."
                )
                
                # Volatilitate (Bonus)
                volatility = hist_curve.pct_change().std() * np.sqrt(252) * 100 if not hist_curve.empty else 0
                c_risk3.metric(
                    "Volatilitate Anualizată", 
                    f"{volatility:.2f}%",
                    help="Cât de mult fluctuează portofoliul într-un an."
                )
                
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
                            textinfo='label+percent'
                        )])
                        fig_sym.update_layout(height=350, margin=dict(t=0, b=0, l=0, r=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_sym, use_container_width=True)

                with col_pie2:
                    st.caption("**După Sector Economic**")
                    # Calculăm alocarea pe sectoare
                    with st.spinner("Analizăm sectoarele..."):
                        df_sectors = get_portfolio_sectors(df_calc)
                    
                    if not df_sectors.empty:
                        fig_sec = go.Figure(data=[go.Pie(
                            labels=df_sectors['Sector'], 
                            values=df_sectors['MarketValue'], 
                            hole=.4,
                            textinfo='label+percent'
                        )])
                        fig_sec.update_layout(height=350, margin=dict(t=0, b=0, l=0, r=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_sec, use_container_width=True)
                    else:
                        st.info("Nu există date suficiente pentru sectoare.")

                # --- NEW: CORRELATION MATRIX ---
                st.markdown("---")
                st.subheader("🧩 Analiză Diversificare (Corelare)")
                current_tickers = df_subset['Symbol'].unique().tolist()
                if len(current_tickers) > 1:
                    with st.spinner("Generăm matricea de corelare..."):
                        fig_corr = plot_correlation_matrix(current_tickers)
                        if fig_corr:
                            st.plotly_chart(fig_corr, use_container_width=True)
                            st.caption("ℹ️ Notă: O corelare de **1.00** înseamnă că activele se mișcă identic. O valoare sub **0.5** sau negativă indică o diversificare bună.")
                else:
                    st.info("Adaugă cel puțin 2 active diferite în portofoliu pentru a vedea matricea de corelare.")

                # --- ELEMENTE SUPRAPUSE PENTRU MOBIL ---
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
    # 4. PIAȚĂ GLOBALĂ (CU DASHBOARD MACRO)
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
            # Slider pentru timp (exact ca la portofoliu)
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
                # Filtrare după Slider-ul de Timp
                days_map = {"1L": 30, "3L": 90, "6L": 180, "1A": 365, "3A": 1095, "5A": 1825}
                days = days_map.get(time_frame, 365)
                subset = series.iloc[-days:] # Tăiem exact cât a cerut userul
                
                # Calcule Metrici (Delta)
                curr_val = subset.iloc[-1]
                prev_val = subset.iloc[-2]
                delta = curr_val - prev_val
                pct = (delta / prev_val) * 100
                
                # --- FORMATARE TEXT (Adăugare %) ---
                # Dacă e Yield (Titluri de stat), punem % la final
                suffix = "%" if "Yield" in selected_macro_name else ""
                val_fmt = f"{curr_val:.4f}{suffix}"
                
                # Afișare Metrică
                st.metric(f"{selected_macro_name}", val_fmt, f"{delta:.4f} ({pct:.2f}%)")
                
                # --- 3. GRAFIC PLOTLY (DETALIAT) ---
                fig_macro = go.Figure()
                
                # Linie colorată și umplută (Gradient)
                fig_macro.add_trace(go.Scatter(
                    x=subset.index, 
                    y=subset.values,
                    mode='lines',
                    fill='tozeroy', # Umple sub linie
                    line=dict(color='#58A6FF', width=2),
                    name=selected_macro_name
                ))
                
                # TRUC: Pentru Valute (EUR/RON), "Zoom-in" pe axa Y
                # Dacă variația e mică (sub 10%), nu porni axa de la 0, ci de la minim
                y_min = subset.min()
                y_max = subset.max()
                is_stable = (y_max - y_min) / y_min < 0.1 # Variație sub 10%
                
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
                        autorange=True if not range_y else False, # Auto sau Zoom forțat
                        range=range_y
                    )
                )
                
                st.plotly_chart(fig_macro, use_container_width=True)

            else:
                st.warning("Date indisponibile sau eroare conexiune Yahoo.")

        st.markdown("---")
        # ---------------------------------------------------------
        
        # --- TABELELE VECHI (PARTEA CARE ERA DEJA ÎN COD) ---
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
    elif sectiune == "5. Import Date (CSV)":
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
                fg_score, fg_label, vix_val = calculate_fear_greed_proxy(us_data)
                fg_color = "#F85149" if fg_score < 45 else "#3FB950" if fg_score > 55 else "#8B949E"
                
                st.markdown(f"""
                <div style="text-align: center; background-color: #21262D; padding: 10px; border-radius: 10px;">
                    <small style="color: #8B949E;">Fear & Greed (Est.)</small>
                    <h2 style="color: {fg_color}; margin: 0;">{int(fg_score)}</h2>
                    <div style="font-weight:bold; color: #FFFFFF;">{fg_label}</div>
                    <small style="color: #8B949E;">VIX: {vix_val:.2f}</small>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

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
    elif sectiune == "7. Scanner Volum (RVOL)":
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
                'SMTL.RO', 'AROBS.RO', 'AQ.RO', 'ARS.RO', 'BRK.RO', 'IARV.RO', 'TTS.RO', 'WINE.RO', 'TEL.RO', 'DN.RO', 'AG.RO', 
                'BENTO.RO', 'PE.RO', 'COTE.RO', 'PBK.RO', 'SAFE.RO', 'TBK.RO', 'CFH.RO', 'SFG.RO'
            ],
            
            "🇺🇸 SUA - Tech & Growth (Nasdaq 100)": [
                'NVDA', 'MSFT', 'AAPL', 'AMZN', 'META', 'GOOGL', 'TSLA', 'AVGO', 'COST', 'PEP', 'CSCO', 'TMUS',
                'CMCSA', 'INTC', 'AMD', 'CLS', 'NFLX', 'TXN', 'ANET', 'NET', 'SBUX', 'ISRG', 'MDLZ', 'GILD',
                'ARM', 'BKNG', 'AT&T', 'PANW', 'MU', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'CRWV', 'CSX', 'PYPL', 'ASML',
                'PLTR', 'CRWD', 'ZS', 'MSTR', 'QCOM', 'SNDK', 'HOOD', 'ROKU', 'INOD', 'U', 'ORCL', 'TSM', 'AFRM'
            ],
            
            "🇺🇸 SUA - Industrial & Finance (Dow/S&P)": [
                'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'AXP', 'V', 'MA', 'BRK-B',
                'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'OXY', 'HAL', 'MPC', 'DVN', 'UUUU', 'OKLO', 'VLO',
                'CAT', 'DE', 'BA', 'LMT', 'RTX', 'GD', 'NOC', 'GE', 'MMM', 'HON', 'UNP', 'NVO', 'AMGN', 'BIIB', 'SNY', 'NVS',
                'JNJ', 'LLY', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO', 'MP', 'METC', 'RIO', 'BHP', 'AEM', 'DHR', 'BMY', 'CVS'
            ],
            
            "🇪🇺 Europa - Germania (DAX 40)": [
                'SAP.DE', 'SIE.DE', 'ALV.DE', 'DTE.DE', 'AIR.DE', 'BMW.DE', 'VOW3.DE', 'MBG.DE', 'BAS.DE', 'BAYN.DE',
                'ADS.DE', 'DHL.DE', 'DB1.DE', 'MUV2.DE', 'IFX.DE', 'EOAN.DE', 'RWE.DE', 'DTG.DE', 'HEN3.DE', 'VNA.DE',
                'DBK.DE', 'CBK.DE', 'CON.DE', 'HEI.DE', 'SY1.DE', 'MTX.DE', 'BEI.DE', 'PUM.DE', 'ZAL.DE'
            ],
            
            "🇪🇺 Europa - Franța (CAC 40)": [
                'MC.PA', 'OR.PA', 'TTE.PA', 'SAN.PA', 'AIR.PA', 'SU.PA', 'AI.PA', 'BNP.PA', 'EL.PA', 'KER.PA',
                'RMS.PA', 'SAF.PA', 'CS.PA', 'DG.PA', 'STLAP.PA', 'GLE.PA', 'ACA.PA', 'ORA.PA', 'CAP.PA', 'EN.PA',
                'VIV.PA', 'ENG.PA', 'LR.PA', 'HO.PA', 'ML.PA', 'RI.PA', 'BN.PA', 'DSY.PA'
            ],
            
            "🇬🇧 UK & Others (FTSE/Global)": [
                'SHEL.L', 'AZN.L', 'HSBA.L', 'ULVR.L', 'BP.L', 'RIO.L', 'GSK.L', 'DGE.L', 'REL.L', 'BATS.L',
                'GLEN.L', 'LSEG.L', 'AAL.L', 'BARC.L', 'LLOY.L', 'NWG.L', 'VOD.L', 'RR.L', 'TSCO.L',
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
                            "RVOL": rvol
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
                        # Funcție de colorare
                        def style_scanner(row):
                            if row['Variație %'] > 0:
                                return ['color: #3FB950'] * len(row)
                            else:
                                return ['color: #F85149'] * len(row)

                        # Formatare
                        df_display = df_filtered.style.apply(style_scanner, axis=1).format({
                            "Preț": "{:.2f}",
                            "Variație %": "{:+.2f}%",
                            "Volum Azi": "{:,.0f}",
                            "Volum Mediu (20z)": "{:,.0f}",
                            "RVOL": "{:.2f}x"
                        })
                        
                        st.success(f"Găsit: {len(df_filtered)} companii cu volum neobișnuit în {market_choice}.")
                        st.dataframe(df_display, use_container_width=True, height=600)
                        st.caption("🟢 **Verde:** Breakout | 🔴 **Roșu:** Panic Sell")
                    else:
                        st.info(f"Nicio acțiune din {market_choice} nu depășește pragul de {threshold}x azi.")
                else:
                    st.warning("Eroare la preluarea datelor. Yahoo Finance ar putea limita cererile.")
# ==================================================
    # 8. WATCHLIST (NOU)
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
            
            if tickers_list:
                with st.spinner("Actualizăm prețurile..."):
                    try:
                        live_data = yf.download(tickers_list, period="1d", progress=False)['Close'].iloc[-1]
                    except:
                        live_data = pd.Series()

                # 2. Construim tabelul final
                display_rows = []
                for index, row in df_wl.iterrows():
                    sym = row['Symbol']
                    target = float(row['TargetPrice'])
                    note = row['Notes']
                    
                    # Extragem prețul curent (gestionăm cazuri de un singur ticker vs listă)
                    try:
                        if len(tickers_list) == 1:
                            curr = float(live_data) # Dacă e un singur număr
                        else:
                            curr = float(live_data[sym]) # Dacă e Series
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
                        "Status": "✅ CUMPĂRĂ ACUM" if is_buy else "⏳ Așteaptă",
                        "Notă": note,
                        "_is_buy": is_buy # Coloană ascunsă pentru colorare
                    })
                
                df_res = pd.DataFrame(display_rows)

                # 3. Stilizare și Afișare (ASCUNDEM _is_buy)
                def highlight_buy(row):
                    # Verificăm coloana ascunsă pentru a decide culoarea
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
                        "_is_buy": None, # <--- Asta ASCUNDE coloana tehnică
                        "Status": st.column_config.TextColumn("Recomandare"),
                    },
                    hide_index=True # Ascunde și indexul (0, 1, 2...) din stânga
                )
                
                # Buton ștergere rapidă
                with st.expander("🗑️ Șterge din listă"):
                    del_sym = st.selectbox("Alege simbol de șters:", tickers_list)
                    if st.button("Șterge"):
                        if remove_from_watchlist(del_sym):
                            st.warning(f"Șters {del_sym}.")
                            st.rerun()

            else:
                st.info("Lista e goală.")
        else:
            st.info("Nu ai nicio acțiune în Watchlist. Folosește formularul de sus.")

if __name__ == "__main__":
    main()


