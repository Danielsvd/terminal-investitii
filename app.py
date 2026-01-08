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

# --- 0. CONFIGURARE GLOBALĂ ---
st.set_page_config(page_title="Terminal Investiții PRO", page_icon="📈", layout="wide")
socket.setdefaulttimeout(10)

FILE_PORTOFOLIU = "portofoliu.csv"

# --- CSS MODERNIZAT (UI PREMIUM) ---
st.markdown("""
    <style>
    /* Stil general aplicație */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Carduri Principale */
    .fin-card, .news-card {
        background-color: #161B22; /* Dark Grey/Blueish */
        padding: 20px;
        border-radius: 15px; /* Rotunjire mai mare */
        border: 1px solid #30363D;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px;
        transition: transform 0.2s;
    }
    
    .fin-card:hover, .news-card:hover {
        border-color: #58A6FF; /* Highlight la hover */
    }

    /* Stilizare Metrici (KPIs) */
    div[data-testid="stMetric"] {
        background-color: #21262D;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #30363D;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #8B949E;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 600;
        color: #FFFFFF;
    }

    /* Stilizare Știri */
    .news-card {
        border-left: 5px solid #238636; /* Verde elegant */
    }
    .news-title {
        font-size: 18px;
        font-weight: 600;
        color: #58A6FF !important;
        text-decoration: none;
        margin-bottom: 8px;
        display: block;
    }
    .news-meta {
        font-size: 12px;
        color: #8B949E;
        margin-bottom: 10px;
        border-bottom: 1px solid #30363D;
        padding-bottom: 5px;
    }

    /* Bara Progres Analiști */
    .analyst-bar-container {
        width: 100%;
        background-color: #30363D;
        height: 12px;
        border-radius: 6px;
        position: relative;
        margin-top: 10px;
        margin-bottom: 5px;
    }
    .analyst-bar-gradient {
        width: 100%;
        height: 100%;
        border-radius: 6px;
        background: linear-gradient(90deg, #238636 0%, #d29922 50%, #da3633 100%);
        opacity: 0.8;
    }
    .analyst-marker {
        position: absolute;
        top: -4px;
        width: 4px;
        height: 20px;
        background-color: #FFFFFF;
        border: 1px solid #000;
        box-shadow: 0 0 5px rgba(255,255,255,0.8);
        z-index: 10;
        transform: translateX(-50%);
    }
    .analyst-labels {
        display: flex;
        justify-content: space-between;
        font-size: 10px;
        color: #8B949E;
        margin-top: 5px;
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
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC,EURUSD=X,GC=F,CL=F&region=US&lang=en-US"
    ],
    "Categorii": {
        "General": [],
        "Energie": ["energie", "petrol", "gaze", "oil", "energy", "curent", "hidroelectrica", "omv", "romgaz", "nuclearelectrica"],
        "Aur/Metale": ["aur", "gold", "argint", "silver", "metal", "cupru", "precious"],
        "Valute": ["euro", "dolar", "ron", "curs", "valutar", "forex", "eur", "usd", "bnt", "schimb"],
        "Șomaj/Muncă": ["somaj", "locuri de munca", "salarii", "unemployment", "jobs", "angajari", "hr", "munca"],
        "Bănci": ["banca", "credit", "bcr", "brd", "revolut", "tbi", "transilvania", "raiffeisen", "cec", "bank"],
        "Dobânzi": ["dobanda", "robor", "ircc", "interest", "fed", "bce", "inflation", "inflatie", "banci centrale"],
        "WallStreet": ["sua", "wall street", "nasdaq", "dow jones", "sp500", "apple", "tesla", "microsoft", "nvidia", "amazon", "google"]
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

def format_num(val, is_pct=False):
    if val is None: return "N/A"
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

# --- FUNCȚII ANALIZĂ ---
@st.cache_data(ttl=3600)
def get_market_data():
    try:
        spy = yf.Ticker("SPY").history(period="1y")['Close']
        return spy
    except: return None

def calculate_alpha(stock_hist, beta):
    try:
        spy = get_market_data()
        if spy is None or stock_hist is None: return None
        stock_close = stock_hist['Close'].iloc[-len(spy):]
        ret_stock = (stock_close.iloc[-1] / stock_close.iloc[0]) - 1
        ret_market = (spy.iloc[-1] / spy.iloc[0]) - 1
        risk_free = 0.04
        if beta is None: beta = 1.0
        alpha = ret_stock - (risk_free + beta * (ret_market - risk_free))
        return alpha
    except: return None

@st.cache_data(ttl=900)
def get_stock_data(symbol):
    # Inițializăm variabilele goale
    hist = None
    info = {}
    earnings = None
    
    try:
        t = yf.Ticker(symbol)
        
        # 1. Încercăm să luăm ISTORICUL (Graficul) - Asta e prioritatea
        try:
            hist = t.history(period="5y")
        except:
            hist = pd.DataFrame() # Dacă eșuează, facem un tabel gol
            
        # Fallback: Dacă history e gol, încercăm metoda 'download' care uneori fentează blocajul
        if hist is None or hist.empty:
            try:
                hist = yf.download(symbol, period="5y", progress=False)
            except: 
                pass

        # 2. Verificare pentru acțiuni românești (BVB)
        if (hist is None or hist.empty) and not symbol.endswith(".RO"):
            sym_ro = symbol + ".RO"
            t_ro = yf.Ticker(sym_ro)
            try:
                hist_ro = t_ro.history(period="5y")
                if not hist_ro.empty:
                    hist = hist_ro
                    symbol = sym_ro
                    t = t_ro # Comutăm obiectul Ticker pe cel de RO
            except: 
                pass

        # Dacă nici acum nu avem grafic, înseamnă că simbolul e chiar greșit
        if hist is None or hist.empty:
            return None, {}, None, symbol

        # 3. Încercăm să luăm DATELE FUNDAMENTALE (Info)
        # Aici apare des eroarea pe Cloud. O izolăm ca să nu crape tot.
        try:
            info = t.info
            # Uneori info e None, așa că ne asigurăm că e dict
            if info is None: info = {}
        except Exception:
            # Dacă Yahoo blochează info, continuăm fără el (graficul va merge)
            info = {} 

        # 4. Earnings
        try:
            earnings = getattr(t, 'earnings_history', None)
        except:
            earnings = None

        return hist, info, earnings, symbol

    except Exception as e:
        print(f"Eroare majoră: {e}")
        return None, {}, None, symbol

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

# --- FUNCȚII PORTOFOLIU ---
def load_portfolio():
    if not os.path.exists(FILE_PORTOFOLIU):
        pd.DataFrame(columns=["Symbol", "Date", "Quantity", "AvgPrice"]).to_csv(FILE_PORTOFOLIU, index=False)
    return pd.read_csv(FILE_PORTOFOLIU)

def add_trade(s, q, p, d):
    df = load_portfolio()
    new_row = pd.DataFrame({"Symbol": [s], "Date": [d], "Quantity": [q], "AvgPrice": [p]})
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(FILE_PORTOFOLIU, index=False)

@st.cache_data(ttl=300)
def get_portfolio_history_data(tickers):
    if not tickers: return pd.DataFrame()
    data = yf.download(tickers, period="5y", group_by='ticker')
    return data

def calculate_portfolio_performance(df, history_range="1A"):
    if df.empty: return pd.DataFrame(), pd.DataFrame(), 0, 0
    
    tickers = df['Symbol'].unique().tolist()
    hist_data = get_portfolio_history_data(tickers)
    
    current_vals = []
    total_daily_pl_abs = 0 
    
    for index, row in df.iterrows():
        sym = row['Symbol']
        qty = row['Quantity']
        avg_p = row['AvgPrice']
        
        try:
            if len(tickers) == 1:
                series = hist_data['Close']
            else:
                series = hist_data[sym]['Close']
            
            curr_p = series.iloc[-1]
            prev_p = series.iloc[-2]
        except:
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
            if len(tickers) == 1:
                price_series = hist_data['Close']
            else:
                price_series = hist_data[sym]['Close']
                
            price_series = price_series.fillna(method='ffill').fillna(method='bfill')
            
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
    total_daily_pl_pct = (total_daily_pl_abs / (total_val_now - total_daily_pl_abs) * 100) if total_val_now != 0 else 0
    
    return df_result, portfolio_curve, total_daily_pl_abs, total_daily_pl_pct

# --- MAIN APP ---
def main():
    st.sidebar.title("Navigare")
    sectiune = st.sidebar.radio("Mergi la:", ["1. Agregator Știri", "2. Analiză Companie", "3. Portofoliu"])
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
                 # Verificare defensivă: dacă datele sunt invalide sau goale, afișăm N/A
            if curr_price is not None and isinstance(curr_price, (int, float)):
                 m1.metric(f"Interval ({time_opt})", f"{curr_price:.2f} {info.get('currency', '')}", f"{diff_val:.2f} ({diff_pct:.2f}%)")
                 m2.metric("Evolutie Azi", f"{curr_price:.2f}", f"{day_val:.2f} ({day_pct:.2f}%)")
            else:
                 m1.metric(f"Interval ({time_opt})", "N/A", "0.00%")
                 m2.metric("Evolutie Azi", "N/A", "0.00%")

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

            with st.container():
                c_eval, c_prof, c_indat, c_risc = st.columns(4)
                with c_eval:
                    st.markdown("**Evaluare**")
                    st.metric("P/E Ratio", format_num(info.get('trailingPE')), help="Cât plătești pentru 1$ profit.")
                    st.metric("Forward P/E", format_num(info.get('forwardPE')), help="P/E estimat pentru anul viitor.")
                    st.metric("P/BV", format_num(info.get('priceToBook')), help="Preț față de valoarea contabilă.")
                    st.metric("EPS", format_num(info.get('trailingEps')), help="Profit net pe acțiune.")
                    st.metric("Val. Contabilă/Acțiune", format_num(info.get('bookValue')), help="Valoarea activelor nete per acțiune (Book Value).")

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
                
                # --- VIZUALIZARE SCOR ANALIST ---
                if rec_mean:
                    # Calcul pozitie marker: (Scor - 1) / 4 * 100. Ex: Scor 2 -> (1/4)*100 = 25%
                    # Limitam la 1-5
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
    # 3. PORTOFOLIU (FINALIZAT)
    # ==================================================
    elif sectiune == "3. Portofoliu":
        st.title("💼 Portofoliu Personal")
        
        with st.expander("➕ Adaugă Tranzacție Nouă"):
            with st.form("add_pf"):
                c1, c2, c3 = st.columns(3)
                s = c1.text_input("Simbol (ex: AAPL)").upper()
                q = c2.number_input("Cantitate", min_value=0.01, value=1.0)
                p = c3.number_input("Preț Achiziție", min_value=0.01, value=100.0)
                d_acq = st.date_input("Data", datetime.today())
                if st.form_submit_button("Salvează") and s:
                    add_trade(s, q, p, d_acq)
                    st.success(f"Adăugat {s}!")
                    st.rerun()

        df_pf = load_portfolio()
        
        if df_pf.empty:
            st.info("Portofoliul este gol. Adaugă o tranzacție mai sus.")
        else:
            st.markdown("### Perioadă Analiză")
            hist_range = st.select_slider("", options=["1Z", "1S", "1L", "3L", "6L", "1A", "3A", "5A"], value="1A", key="range_slider")
            
            with st.spinner("Se calculează performanța..."):
                df_calc, hist_curve, daily_abs, daily_pct = calculate_portfolio_performance(df_pf, hist_range)

            if not hist_curve.empty:
                start_val = hist_curve.iloc[0]
                end_val = hist_curve.iloc[-1]
                growth_abs = end_val - start_val
                growth_pct = (growth_abs / start_val * 100) if start_val != 0 else 0
                
                col_m1, col_m2 = st.columns(2)
                col_m1.metric(
                    label="Valoare Totală Curentă",
                    value=f"{end_val:,.2f}",
                    delta=f"{daily_abs:,.2f} ({daily_pct:.2f}%) Azi"
                )
                col_m2.metric(
                    label=f"Evoluție în perioada selectată ({hist_range})",
                    value=f"{growth_abs:,.2f}",
                    delta=f"{growth_pct:.2f}%",
                    delta_color="normal"
                )
                
                st.markdown("---")
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(x=hist_curve.index, y=hist_curve.values, fill='tozeroy', line=dict(color='#238636'), name='Valoare Portofoliu'))
                fig_hist.update_layout(height=400, template="plotly_dark", margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.warning("Date insuficiente pentru a genera istoricul.")

            st.markdown("---")

            col_table, col_pie = st.columns([1.5, 1])
            
            with col_table:
                st.subheader("Detaliu Poziții")
                def color_profit(val):
                    color = '#3FB950' if val >= 0 else '#F85149'
                    return f'color: {color}'

                display_cols = ['Symbol', 'Quantity', 'AvgPrice', 'CurrentPrice', 'MarketValue', 'Profit', 'Profit %']
                if not df_calc.empty:
                    df_show = df_calc[display_cols].copy()
                    st.dataframe(
                        df_show.style.map(color_profit, subset=['Profit', 'Profit %'])
                        .format({
                            'Quantity': '{:.2f}', 'AvgPrice': '{:.2f}', 'CurrentPrice': '{:.2f}',
                            'MarketValue': '{:,.2f}', 'Profit': '{:,.2f}', 'Profit %': '{:.2f}%'
                        }),
                        use_container_width=True
                    )

            with col_pie:
                st.subheader("Alocare Active")
                if not df_calc.empty:
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=df_calc['Symbol'], 
                        values=df_calc['MarketValue'], 
                        hole=.4,
                        textinfo='label+percent',
                        textposition='outside'
                    )])
                    fig_pie.update_layout(height=500, template="plotly_dark", margin=dict(t=20, b=20, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_pie, use_container_width=True)

            if st.button("Șterge Portofoliu (Reset)"):
                os.remove(FILE_PORTOFOLIU)
                st.rerun()

if __name__ == "__main__":

    main()

