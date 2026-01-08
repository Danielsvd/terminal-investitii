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
socket.setdefaulttimeout(15) # Timeout mai mare pentru conexiuni lente

FILE_PORTOFOLIU = "portofoliu.csv"

# --- CSS MODERNIZAT (UI PREMIUM) ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    .fin-card, .news-card {
        background-color: #161B22; padding: 20px; border-radius: 15px;
        border: 1px solid #30363D; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); margin-bottom: 15px;
    }
    div[data-testid="stMetric"] {
        background-color: #21262D; padding: 15px; border-radius: 12px; border: 1px solid #30363D;
    }
    div[data-testid="stMetricLabel"] { font-size: 14px; color: #8B949E; }
    div[data-testid="stMetricValue"] { font-size: 24px; font-weight: 600; color: #FFFFFF; }
    .news-title { font-size: 18px; font-weight: 600; color: #58A6FF !important; text-decoration: none; }
    .news-meta { font-size: 12px; color: #8B949E; margin-bottom: 10px; border-bottom: 1px solid #30363D; }
    .impact-poz { color: #3FB950; font-weight: bold; background: rgba(63, 185, 80, 0.1); padding: 2px 6px; border-radius: 4px; }
    .impact-neg { color: #F85149; font-weight: bold; background: rgba(248, 81, 73, 0.1); padding: 2px 6px; border-radius: 4px; }
    .impact-neu { color: #8B949E; font-weight: bold; background: rgba(139, 148, 158, 0.1); padding: 2px 6px; border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. CONFIGURARE AGREGATOR ---
RSS_CONFIG = {
    "Feeds": [
        "https://www.zf.ro/rss", "https://www.biziday.ro/feed/", "https://www.economica.net/rss",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC,EURUSD=X,GC=F&region=US&lang=en-US"
    ],
    "Categorii": {
        "General": [],
        "Energie": ["energie", "petrol", "gaze", "hidroelectrica", "omv", "nuclearelectrica"],
        "Bănci": ["banca", "bcr", "brd", "transilvania", "dobanda", "robor"],
        "IT & Tech": ["apple", "microsoft", "google", "nvidia", "ai", "tech"],
        "Internațional": ["sua", "ue", "fed", "bce", "germania", "franta"]
    }
}

# --- FUNCȚII UTILITARE (REPARATE) ---
def parse_date(entry):
    try:
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            return datetime.fromtimestamp(time.mktime(entry.published_parsed))
        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
            return datetime.fromtimestamp(time.mktime(entry.updated_parsed))
    except: pass
    return datetime.now()

def format_num(val, is_pct=False):
    # ACEASTA ESTE FIXAREA ERORII TYPEERROR
    if val is None or val == "N/A" or isinstance(val, str):
        return "N/A"
    
    # Verificăm dacă e număr valid
    if not isinstance(val, (int, float, np.number)):
        return "N/A"
        
    if is_pct:
        return f"{val * 100:.2f}%"
        
    if val >= 1e12: return f"{val/1e12:.2f} T"
    if val >= 1e9: return f"{val/1e9:.2f} B"
    if val >= 1e6: return f"{val/1e6:.2f} M"
    return f"{val:,.2f}"

def get_sentiment(text):
    blob = TextBlob(text)
    if blob.sentiment.polarity > 0.05: return "Pozitiv", "impact-poz", "↗"
    if blob.sentiment.polarity < -0.05: return "Negativ", "impact-neg", "↘"
    return "Neutru", "impact-neu", "→"

# --- FUNCȚII DATE ---
@st.cache_data(ttl=600)
def fetch_news_data():
    all_news = []
    for url in RSS_CONFIG["Feeds"]:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
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

@st.cache_data(ttl=900)
def get_stock_data(symbol):
    # Reset variables
    hist, info, earnings = pd.DataFrame(), {}, None
    symbol = symbol.strip().upper()
    
    try:
        t = yf.Ticker(symbol)
        # Încercăm history
        try:
            hist = t.history(period="5y")
        except: pass
        
        # Fallback BVB
        if hist.empty and not symbol.endswith(".RO"):
            try:
                t_ro = yf.Ticker(symbol + ".RO")
                hist_ro = t_ro.history(period="5y")
                if not hist_ro.empty:
                    hist = hist_ro; symbol = symbol + ".RO"; t = t_ro
            except: pass
            
        # Fallback download explicit
        if hist.empty:
            try:
                hist = yf.download(symbol, period="5y", progress=False, threads=False)
            except: pass

        if hist.empty:
            return pd.DataFrame(), {}, None, symbol

        # Info - cu protecție extra
        try:
            info = t.info
            if info is None: info = {}
        except: info = {}
        
        return hist, info, getattr(t, 'earnings_history', None), symbol

    except Exception as e:
        print(f"Err {symbol}: {e}")
        return pd.DataFrame(), {}, None, symbol

def calculate_alpha(stock_hist, beta):
    try:
        spy = yf.Ticker("SPY").history(period="1y")['Close']
        if spy.empty or stock_hist.empty: return None
        
        # Aliniem datele
        stock_close = stock_hist['Close'].iloc[-len(spy):]
        if len(stock_close) < 10: return None

        ret_stock = (stock_close.iloc[-1] / stock_close.iloc[0]) - 1
        ret_market = (spy.iloc[-1] / spy.iloc[0]) - 1
        
        if beta is None or beta == "N/A": beta = 1.0
        
        risk_free = 0.04
        alpha = ret_stock - (risk_free + beta * (ret_market - risk_free))
        return alpha
    except: return None

# --- FUNCȚII PORTOFOLIU ---
def load_portfolio():
    if not os.path.exists(FILE_PORTOFOLIU):
        pd.DataFrame(columns=["Symbol", "Date", "Quantity", "AvgPrice"]).to_csv(FILE_PORTOFOLIU, index=False)
    return pd.read_csv(FILE_PORTOFOLIU)

def add_trade(s, q, p, d):
    s = str(s).upper().strip()
    df = load_portfolio()
    new_row = pd.DataFrame({"Symbol": [s], "Date": [d], "Quantity": [q], "AvgPrice": [p]})
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(FILE_PORTOFOLIU, index=False)

@st.cache_data(ttl=300)
def get_portfolio_data_bulk(tickers):
    if not tickers: return pd.DataFrame()
    unique_tickers = list(set([str(t).upper().strip() for t in tickers]))
    try:
        # Folosim auto_adjust=True pentru prețuri corecte
        data = yf.download(unique_tickers, period="2y", group_by='ticker', auto_adjust=True, threads=True)
        if len(unique_tickers) == 1:
            # Fix structura pentru un singur ticker
            data.columns = pd.MultiIndex.from_product([unique_tickers, data.columns])
        return data
    except: return pd.DataFrame()

def calculate_portfolio_performance(df, history_range="1A"):
    if df.empty: return pd.DataFrame(), pd.Series(), 0, 0
    
    tickers = df['Symbol'].unique().tolist()
    hist_data = get_portfolio_data_bulk(tickers)
    
    vals = []
    total_daily_pl_abs = 0
    
    for _, row in df.iterrows():
        sym = row['Symbol'].upper().strip()
        qty, avg_p = row['Quantity'], row['AvgPrice']
        
        curr_p, prev_p = 0.0, 0.0
        try:
            if sym in hist_data.columns.levels[0]:
                series = hist_data[sym]['Close'].dropna()
                if not series.empty:
                    curr_p = series.iloc[-1]
                    prev_p = series.iloc[-2] if len(series) > 1 else curr_p
        except: pass
        
        mkt_val = qty * curr_p
        profit_total = mkt_val - (qty * avg_p)
        profit_pct = (profit_total / (qty * avg_p) * 100) if avg_p else 0
        
        # Fluctuație zilnică
        daily_diff = curr_p - prev_p
        daily_pl_abs = daily_diff * qty
        daily_pl_pct = (daily_diff / prev_p * 100) if prev_p else 0
        
        total_daily_pl_abs += daily_pl_abs
        
        vals.append({
            'Symbol': sym, 'Quantity': qty, 'AvgPrice': avg_p,
            'CurrentPrice': curr_p, 'MarketValue': mkt_val,
            'ProfitTotal': profit_total, 'ProfitTotal%': profit_pct,
            'DailyProfit': daily_pl_abs, 'Daily%': daily_pl_pct
        })
        
    res_df = pd.DataFrame(vals)
    
    # Calcul curbă istorică
    curve = pd.Series(0, index=pd.date_range(end=datetime.today(), periods=2)) # Default
    try:
        first = True
        for _, row in df.iterrows():
            sym = row['Symbol'].upper().strip()
            qty = row['Quantity']
            if sym in hist_data.columns.levels[0]:
                s_close = hist_data[sym]['Close'].fillna(method='ffill')
                if first:
                    curve = s_close * qty
                    first = False
                else:
                    curve = curve.add(s_close * qty, fill_value=0)
        
        curve = curve.dropna()
        days_map = {"1S":7, "1L":30, "3L":90, "6L":180, "1A":365, "3A":1095}
        d = days_map.get(history_range, 365)
        curve = curve.iloc[-d:]
    except: pass

    current_total = curve.iloc[-1] if not curve.empty else 0
    daily_total_pct = (total_daily_pl_abs / (current_total - total_daily_pl_abs) * 100) if current_total else 0

    return res_df, curve, total_daily_pl_abs, daily_total_pct

# --- MAIN APP ---
def main():
    st.sidebar.title("Meniu")
    sectiune = st.sidebar.radio("Navigare:", ["Agregator Știri", "Analiză Companie", "Portofoliu"])
    
    # 1. ȘTIRI
    if sectiune == "Agregator Știri":
        st.title("🌍 Știri Financiare")
        if st.button("Actualizează"): fetch_news_data.clear(); st.rerun()
        
        news = fetch_news_data()
        tabs = st.tabs(RSS_CONFIG["Categorii"].keys())
        for i, (cat, kws) in enumerate(RSS_CONFIG["Categorii"].items()):
            with tabs[i]:
                # Filtrare simplă
                filtered = [n for n in news if not kws or any(k in (n['title']+n['summary']).lower() for k in kws)]
                if not filtered: st.info("Fără știri recente.")
                for n in filtered:
                    st.markdown(f"""<div class="news-card"><a href="{n['link']}" class="news-title">{n['title']}</a><br><small>{n['source']} • {n['date_str']}</small></div>""", unsafe_allow_html=True)

    # 2. ANALIZĂ
    elif sectiune == "Analiză Companie":
        sym = st.sidebar.text_input("Simbol:", "AAPL").upper()
        
        hist, info, earn, real_sym = get_stock_data(sym)
        
        if hist.empty:
            st.error("Nu s-au găsit date. Încearcă alt simbol sau verifică conexiunea.")
        else:
            st.title(f"{info.get('longName', real_sym)}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Preț", format_num(hist['Close'].iloc[-1]))
            c2.metric("Sector", info.get('sector', 'N/A'))
            c3.metric("Industrie", info.get('industry', 'N/A'))
            
            # Grafic
            st.subheader("Grafic Tehnic")
            fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'])])
            fig.update_layout(height=500, template="plotly_dark", paper_bgcolor='#0E1117', plot_bgcolor='#0E1117')
            st.plotly_chart(fig, use_container_width=True)
            
            # Indicatori
            beta = info.get('beta')
            alpha = calculate_alpha(hist, beta)
            
            st.subheader("Indicatori Cheie")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("P/E", format_num(info.get('trailingPE')))
            m2.metric("Profit Marg", format_num(info.get('profitMargins'), True))
            m3.metric("Beta", format_num(beta))
            # Aici era eroarea! Acum e protejată de format_num
            m4.metric("Alpha (1Y)", format_num(alpha, True))

    # 3. PORTOFOLIU
    elif sectiune == "Portofoliu":
        st.title("💼 Portofoliu")
        
        with st.expander("➕ Adaugă Tranzacție"):
            with st.form("pf_add"):
                c1, c2, c3 = st.columns(3)
                s = c1.text_input("Simbol").upper()
                q = c2.number_input("Cantitate", 1.0)
                p = c3.number_input("Preț Achiziție", 100.0)
                if st.form_submit_button("Salvează") and s:
                    add_trade(s, q, p, datetime.now())
                    st.rerun()

        df_pf = load_portfolio()
        if df_pf.empty:
            st.info("Portofoliu gol.")
        else:
            df_calc, curve, day_abs, day_pct = calculate_portfolio_performance(df_pf)
            
            if not curve.empty:
                val_now = curve.iloc[-1]
                val_start = curve.iloc[0]
                total_growth = val_now - val_start
                total_pct = (total_growth / val_start * 100) if val_start else 0
                
                k1, k2 = st.columns(2)
                k1.metric("Valoare Totală", f"{val_now:,.2f}", f"{day_abs:,.2f} ({day_pct:.2f}%) Azi")
                k2.metric("Evoluție Totală", f"{total_growth:,.2f}", f"{total_pct:.2f}% Total")
                
                st.area_chart(curve, color="#238636")
            
            # TABEL DETALIAT CU FLUCTUAȚII
            st.subheader("Detaliere Poziții")
            if not df_calc.empty:
                # Selectăm și redenumim coloanele pentru afișare clară
                display_df = df_calc[[
                    'Symbol', 'Quantity', 'AvgPrice', 'CurrentPrice', 
                    'DailyProfit', 'Daily%', 'ProfitTotal', 'ProfitTotal%'
                ]].copy()
                
                # Formatare culori pentru profit/pierdere
                def color_vals(val):
                    color = '#3FB950' if val >= 0 else '#F85149'
                    return f'color: {color}'

                st.dataframe(
                    display_df.style.format({
                        'Quantity': '{:.2f}', 'AvgPrice': '{:.2f}', 'CurrentPrice': '{:.2f}',
                        'DailyProfit': '{:+,.2f}', 'Daily%': '{:+.2f}%',
                        'ProfitTotal': '{:+,.2f}', 'ProfitTotal%': '{:+.2f}%'
                    }).map(color_vals, subset=['DailyProfit', 'Daily%', 'ProfitTotal', 'ProfitTotal%']),
                    use_container_width=True
                )
                
            # Editor
            st.markdown("---")
            st.subheader("Editare Manuală")
            edited = st.data_editor(df_pf, num_rows="dynamic", key="edit_pf")
            if st.button("Salvează Modificări"):
                edited.to_csv(FILE_PORTOFOLIU, index=False)
                st.success("Salvat!")
                st.rerun()

if __name__ == "__main__":
    main()
