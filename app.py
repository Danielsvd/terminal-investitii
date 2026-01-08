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
socket.setdefaulttimeout(15)

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
        background-color: #161B22;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #30363D;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px;
        transition: transform 0.2s;
    }
    .fin-card:hover, .news-card:hover {
        border-color: #58A6FF;
    }

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
        background: linear-gradient(90deg, #238636 0%, #d29922 50%, #da3633 100%);
        opacity: 0.8;
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
        "https://www.zf.ro/rss", "https://www.biziday.ro/feed/",
        "https://www.economica.net/rss", "https://www.bursa.ro/_rss/?t=pcaps",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC,EURUSD=X,GC=F,CL=F&region=US&lang=en-US"
    ],
    "Categorii": {
        "General": [],
        "Energie": ["energie", "petrol", "gaze", "oil", "energy", "curent", "hidroelectrica", "omv", "romgaz"],
        "Aur/Metale": ["aur", "gold", "argint", "silver", "metal", "cupru"],
        "Valute": ["euro", "dolar", "ron", "curs", "valutar", "forex", "eur", "usd"],
        "Șomaj/Muncă": ["somaj", "locuri de munca", "salarii", "jobs", "angajari"],
        "Bănci": ["banca", "credit", "bcr", "brd", "revolut", "tbi", "transilvania", "dobanda", "robor", "ircc", "inflatie"],
        "WallStreet": ["sua", "wall street", "nasdaq", "dow jones", "sp500", "apple", "tesla", "microsoft", "nvidia", "amazon", "google"]
    }
}

# --- FUNCȚII UTILITARE (FIXATE) ---
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
    # REPARATIE: Verificare strictă pentru a evita TypeError
    if val is None or val == "N/A" or isinstance(val, str):
        return "N/A"
    if isinstance(val, (pd.Series, np.ndarray)):
        if val.empty: return "N/A"
        try: val = float(val.iloc[0]) if isinstance(val, pd.Series) else float(val[0])
        except: return "N/A"
    
    if not isinstance(val, (int, float, np.number)):
        return "N/A"

    if is_pct: return f"{val * 100:.2f}%"
    if val >= 1e12: return f"{val/1e12:.2f} T"
    if val >= 1e9: return f"{val/1e9:.2f} B"
    if val >= 1e6: return f"{val/1e6:.2f} M"
    return f"{val:,.2f}"

def get_sentiment(text):
    blob = TextBlob(text)
    if blob.sentiment.polarity > 0.05: return "Pozitiv", "impact-poz", "↗"
    elif blob.sentiment.polarity < -0.05: return "Negativ", "impact-neg", "↘"
    else: return "Neutru", "impact-neu", "→"

# --- FUNCȚII ȘTIRI ---
@st.cache_data(ttl=600, show_spinner=False)
def fetch_news_data():
    all_news = []
    for url in RSS_CONFIG["Feeds"]:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:15]:
                dt = parse_date(entry)
                all_news.append({
                    "title": entry.title, "link": entry.link,
                    "summary": getattr(entry, "summary", ""),
                    "source": feed.feed.get("title", "Sursă"),
                    "date_obj": dt, "date_str": dt.strftime("%Y-%m-%d %H:%M")
                })
        except: continue
    all_news.sort(key=lambda x: x['date_obj'], reverse=True)
    return all_news

def filter_news(all_news, category):
    keywords = RSS_CONFIG["Categorii"].get(category, [])
    if not keywords: return all_news
    return [item for item in all_news if any(k in (item['title']+" "+item['summary']).lower() for k in keywords)]

def get_company_news_rss(symbol):
    try:
        feed = feedparser.parse(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US")
        news_list = []
        for entry in feed.entries[:7]:
            dt = parse_date(entry)
            news_list.append({
                "title": entry.title, "link": entry.link,
                "publisher": "Yahoo Finance", "date_str": dt.strftime("%Y-%m-%d %H:%M")
            })
        return news_list
    except: return []

# --- FUNCȚII ANALIZĂ (REPARATE) ---
@st.cache_data(ttl=900)
def get_stock_data(symbol):
    symbol = symbol.strip().upper()
    try:
        t = yf.Ticker(symbol)
        hist = pd.DataFrame()
        
        # 1. Încercare standard
        try: hist = t.history(period="5y")
        except: pass
        
        # 2. Fallback download explicit
        if hist.empty:
            try: hist = yf.download(symbol, period="5y", progress=False, threads=False)
            except: pass
            
        # 3. Fallback BVB
        if hist.empty and not symbol.endswith(".RO"):
            try:
                t_ro = yf.Ticker(symbol + ".RO")
                hist_ro = t_ro.history(period="5y")
                if not hist_ro.empty:
                    return hist_ro, t_ro.info, getattr(t_ro, 'earnings_history', None), symbol + ".RO"
            except: pass
            
        if hist.empty: return None, {}, None, symbol
        
        # Info safe fetch
        info = {}
        try: info = t.info
        except: pass
        
        return hist, info, getattr(t, 'earnings_history', None), symbol
    except: return None, {}, None, symbol

def calculate_technical_indicators(df):
    if df is None or df.empty: return df
    df = df.copy()
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

def calculate_alpha(stock_hist, beta):
    try:
        spy = yf.Ticker("SPY").history(period="1y")['Close']
        if spy.empty or stock_hist is None or stock_hist.empty: return None
        
        stock_close = stock_hist['Close'].iloc[-len(spy):]
        if len(stock_close) < 10: return None
        
        # REPARATIE: Conversie float explicită
        val_start = float(stock_close.iloc[0])
        val_end = float(stock_close.iloc[-1])
        spy_start = float(spy.iloc[0])
        spy_end = float(spy.iloc[-1])
        
        if val_start == 0 or spy_start == 0: return None

        ret_stock = (val_end / val_start) - 1
        ret_market = (spy_end / spy_start) - 1
        
        if beta is None or beta == "N/A": beta = 1.0
        return ret_stock - (0.04 + beta * (ret_market - 0.04))
    except: return None

# --- FUNCȚII PORTOFOLIU (REPARATE) ---
def load_portfolio():
    if not os.path.exists(FILE_PORTOFOLIU):
        pd.DataFrame(columns=["Symbol", "Date", "Quantity", "AvgPrice"]).to_csv(FILE_PORTOFOLIU, index=False)
    df = pd.read_csv(FILE_PORTOFOLIU)
    # Normalizăm simbolurile la majuscule
    if not df.empty and 'Symbol' in df.columns:
        df['Symbol'] = df['Symbol'].astype(str).str.upper().str.strip()
    return df

def add_trade(s, q, p, d):
    s = str(s).upper().strip()
    df = load_portfolio()
    new_row = pd.DataFrame({"Symbol": [s], "Date": [d], "Quantity": [q], "AvgPrice": [p]})
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(FILE_PORTOFOLIU, index=False)

@st.cache_data(ttl=300)
def get_portfolio_history_data(tickers):
    if not tickers: return pd.DataFrame()
    unique = list(set([str(t).upper().strip() for t in tickers]))
    try:
        # REPARATIE: auto_adjust=True și group_by pentru structură consistentă
        data = yf.download(unique, period="2y", group_by='ticker', auto_adjust=True, threads=True)
        if len(unique) == 1:
            data.columns = pd.MultiIndex.from_product([unique, data.columns])
        return data
    except: return pd.DataFrame()

def calculate_portfolio_performance(df, history_range="1A"):
    if df.empty: return pd.DataFrame(), pd.DataFrame(), 0, 0
    
    tickers = df['Symbol'].unique().tolist()
    hist_data = get_portfolio_history_data(tickers)
    
    current_vals = []
    total_daily_pl_abs = 0 
    
    for index, row in df.iterrows():
        sym = str(row['Symbol']).upper().strip()
        qty = float(row['Quantity'])
        avg_p = float(row['AvgPrice'])
        
        curr_p = 0.0
        prev_p = 0.0
        
        try:
            if not hist_data.empty and sym in hist_data.columns.levels[0]:
                series = hist_data[sym]['Close'].dropna()
                if not series.empty:
                    curr_p = float(series.iloc[-1])
                    prev_p = float(series.iloc[-2]) if len(series) > 1 else curr_p
        except: pass
            
        mkt_val = qty * curr_p
        profit = mkt_val - (qty * avg_p)
        profit_pct = (profit / (qty * avg_p) * 100) if avg_p else 0
        
        daily_chg = (curr_p - prev_p) * qty
        daily_pct = ((curr_p - prev_p) / prev_p * 100) if prev_p else 0
        total_daily_pl_abs += daily_chg
        
        current_vals.append({
            'Symbol': sym, 'Quantity': qty, 'AvgPrice': avg_p, 'CurrentPrice': curr_p,
            'MarketValue': mkt_val, 'TotalProfit': profit, 'Total%': profit_pct,
            'DailyProfit': daily_chg, 'Daily%': daily_pct
        })
    
    df_result = pd.DataFrame(current_vals)
    
    # Calcul curbă istorică
    portfolio_curve = None
    try:
        for index, row in df.iterrows():
            sym = str(row['Symbol']).upper().strip()
            qty = float(row['Quantity'])
            if not hist_data.empty and sym in hist_data.columns.levels[0]:
                series = hist_data[sym]['Close'].fillna(method='ffill').fillna(method='bfill')
                if portfolio_curve is None: portfolio_curve = series * qty
                else: portfolio_curve = portfolio_curve.add(series * qty, fill_value=0)
    except: pass
    
    if portfolio_curve is None: portfolio_curve = pd.Series()
    
    days_map = {"1Z": 2, "1S": 7, "1L": 30, "3L": 90, "6L": 180, "1A": 365, "3A": 1095, "5A": 1825}
    days = days_map.get(history_range, 365)
    portfolio_curve = portfolio_curve.iloc[-days:]
    
    total_val_now = float(portfolio_curve.iloc[-1]) if not portfolio_curve.empty else 0.0
    start_val = total_val_now - total_daily_pl_abs
    total_daily_pl_pct = (total_daily_pl_abs / start_val * 100) if start_val != 0 else 0
    
    return df_result, portfolio_curve, total_daily_pl_abs, total_daily_pl_pct

# --- MAIN APP ---
def main():
    st.sidebar.title("Navigare")
    sectiune = st.sidebar.radio("Mergi la:", ["1. Agregator Știri", "2. Analiză Companie", "3. Portofoliu"])
    st.sidebar.markdown("---")

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
                            <div style="color:#B0B8C4; font-size:14px;">{item['summary'][:250]}...</div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.info(f"Nu există știri recente pentru: {cat}.")

    elif sectiune == "2. Analiză Companie":
        st.sidebar.header("Căutare")
        sym = st.sidebar.text_input("Simbol (ex: AAPL, TLV):", "AAPL").upper()
        
        st.sidebar.markdown("### Indicatori Grafic")
        show_sma20 = st.sidebar.checkbox("SMA 20", True)
        show_sma50 = st.sidebar.checkbox("SMA 50", True)
        show_sma200 = st.sidebar.checkbox("SMA 200", True)
        show_rsi = st.sidebar.checkbox("RSI 14", True)
        show_macd = st.sidebar.checkbox("MACD", True)

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
            
            # REPARATIE: Calcul defensiv cu float()
            curr_price = float(subset['Close'].iloc[-1]) if not subset.empty else 0.0
            start_price = float(subset['Close'].iloc[0]) if not subset.empty else 0.0
            
            diff_val = curr_price - start_price
            diff_pct = (diff_val / start_price * 100) if start_price != 0 else 0.0
            
            prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else curr_price
            day_val = curr_price - prev_close
            day_pct = (day_val / prev_close * 100) if prev_close != 0 else 0.0

            with col_price_info:
                 m1, m2 = st.columns(2)
                 if curr_price > 0:
                     m1.metric(f"Interval ({time_opt})", f"{curr_price:.2f} {info.get('currency', '')}", f"{diff_val:.2f} ({diff_pct:.2f}%)")
                     m2.metric("Evoluție Azi", f"{curr_price:.2f}", f"{day_val:.2f} ({day_pct:.2f}%)")
                 else:
                     m1.metric("Status", "Date Insuficiente")

            rows_needed = 1 + (1 if show_rsi else 0) + (1 if show_macd else 0)
            row_heights = [0.6] + ([0.2] * (rows_needed - 1))
            total = sum(row_heights)
            row_heights = [r/total for r in row_heights]

            fig = make_subplots(rows=rows_needed, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights)
            fig.add_trace(go.Candlestick(x=subset.index, open=subset['Open'], high=subset['High'], low=subset['Low'], close=subset['Close'], name='Preț'), row=1, col=1)
            
            if show_sma20: fig.add_trace(go.Scatter(x=subset.index, y=subset['SMA20'], line=dict(color='orange', width=1), name='SMA 20'), row=1, col=1)
            if show_sma50: fig.add_trace(go.Scatter(x=subset.index, y=subset['SMA50'], line=dict(color='cyan', width=1), name='SMA 50'), row=1, col=1)
            if show_sma200: fig.add_trace(go.Scatter(x=subset.index, y=subset['SMA200'], line=dict(color='purple', width=1.5), name='SMA 200'), row=1, col=1)

            curr_row = 2
            if show_rsi:
                fig.add_trace(go.Scatter(x=subset.index, y=subset['RSI'], line=dict(color='yellow'), name='RSI'), row=curr_row, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color="red", row=curr_row, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color="green", row=curr_row, col=1)
                curr_row += 1

            if show_macd:
                fig.add_trace(go.Scatter(x=subset.index, y=subset['MACD'], line=dict(color='cyan'), name='MACD'), row=curr_row, col=1)
                fig.add_trace(go.Scatter(x=subset.index, y=subset['Signal'], line=dict(color='orange'), name='Signal'), row=curr_row, col=1)
                fig.add_trace(go.Bar(x=subset.index, y=subset['MACD']-subset['Signal'], name='Hist'), row=curr_row, col=1)

            fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False, paper_bgcolor='#0E1117', plot_bgcolor='#0E1117')
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📊 Indicatori Fundamentali")
            beta_val = info.get('beta')
            alpha_val = calculate_alpha(hist, beta_val)
            
            with st.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("P/E Ratio", format_num(info.get('trailingPE')))
                c2.metric("Profit Margin", format_num(info.get('profitMargins'), True))
                c3.metric("Beta", format_num(beta_val))
                c4.metric("Alpha (1Y)", format_num(alpha_val, True))
                
                c5, c6, c7, c8 = st.columns(4)
                c5.metric("ROA", format_num(info.get('returnOnAssets'), True))
                c6.metric("Datorii/Capital", format_num(info.get('debtToEquity')))
                c7.metric("Venituri", format_num(info.get('totalRevenue')))
                c8.metric("Numerar", format_num(info.get('totalCash')))

            st.markdown("---")
            st.subheader(f"📰 Știri {real_sym}")
            company_news = get_company_news_rss(real_sym)
            if company_news:
                for n in company_news:
                    st.markdown(f"**[{n['title']}]({n['link']})** • {n['date_str']}")
            else:
                st.info("Fără știri recente.")

            st.markdown("<br>", unsafe_allow_html=True)
            col_an_left, col_an_right = st.columns([1, 2])
            with col_an_left:
                st.markdown("""<div class="fin-card"><h4>Analiști</h4></div>""", unsafe_allow_html=True)
                rec = info.get('recommendationKey', 'N/A').replace('_', ' ').upper()
                rec_mean = info.get('recommendationMean')
                color_rec = "#3FB950" if "BUY" in rec else "#F85149" if "SELL" in rec else "#8B949E"
                st.markdown(f"Recomandare: <span style='color:{color_rec}; font-weight:bold;'>{rec}</span>", unsafe_allow_html=True)
                if rec_mean:
                    st.caption(f"Scor Consens: {rec_mean:.1f} (1=Buy, 5=Sell)")

            with col_an_right:
                 st.markdown("""<div class="fin-card"><h4>Earnings</h4></div>""", unsafe_allow_html=True)
                 if earn_df is not None and not earn_df.empty:
                    earn_display = earn_df[['epsEstimate', 'epsActual', 'epsDifference', 'surprisePercent']].copy()
                    earn_display.columns = ['Est', 'Act', 'Diff', 'Surpriză %']
                    st.dataframe(earn_display, use_container_width=True)
                 else:
                    st.info("Lipsă date earnings.")

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
            st.info("Portofoliul este gol.")
        else:
            st.markdown("### Perioadă Analiză")
            hist_range = st.select_slider("", options=["1Z", "1S", "1L", "3L", "6L", "1A", "3A", "5A"], value="1A", key="range_slider")
            
            with st.spinner("Se calculează performanța..."):
                df_calc, hist_curve, daily_abs, daily_pct = calculate_portfolio_performance(df_pf, hist_range)

            if not hist_curve.empty:
                val_now = float(hist_curve.iloc[-1])
                val_start = float(hist_curve.iloc[0])
                growth_abs = val_now - val_start
                growth_pct = (growth_abs / val_start * 100) if val_start != 0 else 0
                
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("Valoare Totală", f"{val_now:,.2f}", f"{daily_abs:,.2f} ({daily_pct:.2f}%) Azi")
                col_m2.metric(f"Evoluție ({hist_range})", f"{growth_abs:,.2f}", f"{growth_pct:.2f}%", delta_color="normal")
                
                st.area_chart(hist_curve, color="#238636")

            st.markdown("---")
            col_table, col_pie = st.columns([1.5, 1])
            
            with col_table:
                st.subheader("Detaliu Poziții")
                if not df_calc.empty:
                    df_show = df_calc[['Symbol', 'Quantity', 'AvgPrice', 'CurrentPrice', 'DailyProfit', 'Daily%', 'TotalProfit', 'Total%']].copy()
                    def color_profit(val):
                        color = '#3FB950' if val >= 0 else '#F85149'
                        return f'color: {color}'
                    st.dataframe(df_show.style.format({'Quantity':'{:.2f}','AvgPrice':'{:.2f}','CurrentPrice':'{:.2f}','DailyProfit':'{:+.2f}','Daily%':'{:+.2f}%','TotalProfit':'{:+.2f}','Total%':'{:+.2f}%'}).map(color_profit, subset=['DailyProfit','Daily%','TotalProfit','Total%']), use_container_width=True)

            with col_pie:
                st.subheader("Alocare Active")
                if not df_calc.empty:
                    fig_pie = go.Figure(data=[go.Pie(labels=df_calc['Symbol'], values=df_calc['MarketValue'], hole=.4)])
                    fig_pie.update_layout(height=400, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("---")
            st.subheader("📝 Editare Registru")
            df_edit = st.data_editor(load_portfolio(), num_rows="dynamic", key="pf_edit")
            if st.button("💾 Salvează Modificările"):
                df_edit.to_csv(FILE_PORTOFOLIU, index=False)
                st.success("Actualizat!")
                st.rerun()

if __name__ == "__main__":
    main()
