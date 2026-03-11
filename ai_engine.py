import pandas as pd
import numpy as np
from prophet import Prophet
from transformers import pipeline
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from sklearn.ensemble import IsolationForest
import yfinance as yf
from datetime import date
from datetime import datetime
import google.generativeai as genai
import streamlit as st

# Citim cheia din fișierul de secrete (configurat la Pasul 3)
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    st.error("Lipsește cheia API! Verifică fișierul .streamlit/secrets.toml")

# Incarcare model de sentiment specializat pe finante (FinBERT)
@st.cache_resource
def get_sentiment_pipeline():
    """Incarcă modelul FinBERT optimizat pentru analiză financiară."""
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_sentiment_ai(news_list):
    """
    Analiză de sentiment specializată pe piețe financiare.
    Înțelege diferența dintre 'boom' (creștere) și 'crash', 
    și ponderea rezultatelor financiare vs titluri senzaționaliste.
    """
    if not news_list: return 0
    
    # Combinăm titlurile pentru o singură interogare (economisim tokeni și timp)
    titles = "\n".join([f"- {n['title']}" for n in news_list[:10]])
    
    prompt = f"""
    Ești un analist financiar senior la o bancă de investiții de top. 
    Sarcina ta este să evaluezi sentimentul acestor știri pentru un acționar.
    
    REGULI CRITICE:
    1. "Shares up", "Surging", "Beat expectations", "AI boom", "Price objective raised" sunt ÎNTOTDEAUNA POZITIVE.
    2. "Cash flow negative" este un risc, dar dacă e însoțit de "Hyper growth", sentimentul poate fi mixt/neutru, nu pur negativ.
    3. Ignoră zgomotul și concentrează-te pe impactul asupra PREȚULUI ACȚIUNII.

    Iată știrile:
    {titles}

    Răspunde strict cu un număr între -1 (extrem de negativ) și 1 (extrem de pozitiv). 
    0 este neutru. Nu adăuga text explicativ.
    """

    try:
        # Aici apelezi modelul tău (OpenAI/Gemini/etc)
        # Exemplu pentru structura ta:
        response = model.generate_content(prompt) # Sau ce funcție de apel folosești
        sentiment_score = float(response.text.strip())
        return sentiment_score
    except:
        return 0

# Funcție utilitară pentru a crea un "cheie" de timp (ziua curentă)
def get_daily_hash():
    return str(date.today())

@st.cache_data(ttl=86400, show_spinner=False) # Cache valabil 24 ore (86400 secunde)
def get_cached_prophet_prediction(symbol, _hist_data, daily_hash):
    """
    Antrenează modelul ML o singură dată pe zi per simbol.
    Prefixul '_' la _hist_data îi spune lui Streamlit să NU facă hashing pe tot tabelul (viteză extra).
    """
    try:
        # Pregătire date (identic cu ce aveai, dar optimizat)
        df_p = _hist_data.reset_index()[['Date', 'Close', 'Volume']]
        df_p.columns = ['ds', 'y', 'volume_regressor']
        df_p['ds'] = df_p['ds'].dt.tz_localize(None)

        m = Prophet(
            daily_seasonality=False, 
            weekly_seasonality=True, 
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        m.add_regressor('volume_regressor')
        m.fit(df_p)
        
        future = m.make_future_dataframe(periods=90)
        avg_recent_vol = df_p['volume_regressor'].tail(20).mean()
        historical_vols = df_p['volume_regressor'].values
        future_vols = np.full(90, avg_recent_vol)
        future['volume_regressor'] = np.concatenate((historical_vols, future_vols))
        
        forecast = m.predict(future)
        return forecast
    except Exception as e:
        print(f"Eroare Prophet: {e}")
        return None

def render_ai_chart(forecast, hist):
    """Generare grafic profesional: Preț (sus) + Volum (jos) cu predicție AI."""
    if forecast is None: return
    
    # 1. Creăm un layout cu 2 etaje (Preț 70%, Volum 30%)
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.7, 0.3]
    )
    
    hist_recent = hist.iloc[-120:]
    
    # --- RÂNDUL 1: PREȚ REAL ȘI PREDICȚIE AI ---
    fig.add_trace(go.Scatter(
        x=hist_recent.index, y=hist_recent['Close'], 
        name='Preț Real', line=dict(color='#58A6FF', width=2)
    ), row=1, col=1)
    
    future_dates = forecast['ds'].iloc[-90:]
    yhat = forecast['yhat'].iloc[-90:]
    
    # Linia de predicție
    fig.add_trace(go.Scatter(
        x=future_dates, y=yhat, 
        line=dict(dash='dash', color='#3FB950'), name='Proiecție AI (90z)'
    ), row=1, col=1)
    
    # Intervalul de încredere (Conul verde)
    fig.add_trace(go.Scatter(
        x=future_dates, y=forecast['yhat_upper'].iloc[-90:], 
        line=dict(width=0), showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=future_dates, y=forecast['yhat_lower'].iloc[-90:], 
        fill='tonexty', fillcolor='rgba(63, 185, 80, 0.1)', 
        line=dict(width=0), name='Interval Probabilitate'
    ), row=1, col=1)

    # --- RÂNDUL 2: VOLUMUL ISTORIC (Datele folosite de Regresor) ---
    # Colorăm barele: Verde dacă prețul a închis pe plus, Roșu pe minus
    colors = ['#3FB950' if row['Close'] >= row['Open'] else '#F85149' for index, row in hist_recent.iterrows()]
    
    fig.add_trace(go.Bar(
        x=hist_recent.index, y=hist_recent['Volume'], 
        marker_color=colors, name='Volum Tranzacționat'
    ), row=2, col=1)
    
    # --- STILIZARE FINALĂ ---
    fig.update_layout(
        height=450, template="plotly_dark", 
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified",
        showlegend=False # Ascundem legenda ca să fie mai curat, e explicat în tooltip
    )
    
    # Ascundem axa X dintre cele două grafice pentru continuitate
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def generate_ai_swot_analysis(info, h_score, z_val, mos_val, alpha, s_score):
    """
    Sintetizează datele fundamentale cu contextul de business.
    """
    swot = {"Strengths": [], "Weaknesses": [], "Opportunities": [], "Threats": []}
    sector = info.get('sector', 'General')
    
    # --- STRENGTHS ---
    if h_score >= 8: swot["Strengths"].append("Bilanț 'Safe Haven' (Sănătate financiară de elită).")
    if info.get('profitMargins', 0) > 0.15: swot["Strengths"].append(f"Profitabilitate peste medie în sectorul {sector}.")
    if s_score > 0.2: swot["Strengths"].append("Narațiune extrem de pozitivă în media financiară.")

    # --- WEAKNESSES ---
    if z_val < 1.8: swot["Weaknesses"].append("Z-Score în zona de stres (Risc structural).")
    if alpha and alpha < -0.05: swot["Weaknesses"].append("Subperformanță cronică (Acțiunea pierde momentum).")
    if info.get('currentRatio', 1) < 1.0: swot["Weaknesses"].append("Probleme potențiale de lichiditate pe termen scurt.")

    # --- OPPORTUNITIES ---
    if mos_val > 25: swot["Opportunities"].append(f"Fereastră de achiziție sub-evaluată ({mos_val:.1f}% discount).")
    if sector == "Technology": swot["Opportunities"].append("Expansiune prin integrare AI și automatizare.")
    if sector == "Energy": swot["Opportunities"].append("Tranziția către surse regenerabile și eficiență carbon.")

    # --- THREATS ---
    if mos_val < -20: swot["Threats"].append("Bula de evaluare (Risc major de corecție a prețului).")
    if yield_spread := info.get('yield_spread', 0.5) < 0: swot["Threats"].append("Recesiune iminentă (Curba dobânzilor inversată).")
    if sector == "Financial Services": swot["Threats"].append("Risc sistemic de credit și reglementări stricte.")
    
    return swot

def detect_market_regime_ai(hist_data):
    """
    Folosește K-Means pentru clasificare și calculează Win Rate-ul istoric al regimului.
    """
    try:
        if hist_data is None or len(hist_data) < 60:
            return "Date insuficiente", "#8B949E", "ℹ️ Minim 60 de zile necesare pentru backtest."

        # 1. Feature Engineering
        df_ml = pd.DataFrame()
        df_ml['Return'] = hist_data['Close'].pct_change() * 100
        df_ml['Volatility'] = df_ml['Return'].rolling(window=10).std()
        
        # Calculăm "Forward Return" la 20 de zile (ce s-a întâmplat după semnal)
        # Shift(-20) aduce prețul din viitor în rândul curent pentru calcul
        df_ml['Fwd_20d_Ret'] = df_ml['Return'].shift(-20).rolling(20).sum()
        df_ml = df_ml.dropna(subset=['Return', 'Volatility'])
        
        # 2. Clustering K-Means (3 regimuri)
        X = df_ml[['Return', 'Volatility']].values
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df_ml['Cluster'] = kmeans.fit_predict(X)
        
        centers = kmeans.cluster_centers_
        panic_cluster = np.argmax(centers[:, 1]) # Volatilitate maximă
        bull_cluster = np.argmax(centers[:, 0])  # Randament maxim
        current_cluster = df_ml['Cluster'].iloc[-1]
        
        # 3. LOGICA DE BACKTESTING
        # Analizăm toate zilele din trecut care au fost în același cluster cu ziua de azi
        cluster_history = df_ml[df_ml['Cluster'] == current_cluster].dropna(subset=['Fwd_20d_Ret'])
        
        if not cluster_history.empty:
            win_rate = (cluster_history['Fwd_20d_Ret'] > 0).mean() * 100
            avg_fwd_ret = cluster_history['Fwd_20d_Ret'].mean()
            backtest_msg = f"📊 **Backtest:** Rata de succes istorică {win_rate:.1f}% | Profit mediu forward (20z): {avg_fwd_ret:+.2f}%"
        else:
            backtest_msg = "ℹ️ Date istorice insuficiente pentru un backtest relevant."

        # Verdict
        if current_cluster == panic_cluster:
            return "🚨 REGIM PANICĂ (Volatilitate Extremă)", "#F85149", backtest_msg
        elif current_cluster == bull_cluster:
            return "🚀 REGIM TREND ASCENDENT (Bullish)", "#3FB950", backtest_msg
        else:
            return "⚖️ REGIM CONSOLIDARE (Lateral)", "#D29922", backtest_msg

    except Exception as e:
        return "Model Offline", "#8B949E", f"Eroare: {str(e)}"
    
def optimize_portfolio_ai(hist_data):
    """
    Optimizarea portofoliului (Modern Portfolio Theory - Markowitz).
    Algoritmul găsește ponderile care maximizează Sharpe Ratio.
    """
    try:
        if hist_data is None or len(hist_data.columns) < 2:
            return None, "Ai nevoie de cel puțin 2 active."

        # Curățăm datele (completăm zilele lipsă, ex: sărbători diferite pe burse)
        df_clean = hist_data.ffill().dropna()
        
        # Randamente zilnice
        returns = df_clean.pct_change().dropna()
        
        # Parametri anuali (252 zile de tranzacționare/an)
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        num_assets = len(mean_returns)
        risk_free_rate = 0.04 # Presupunem 4% rata fără risc (yield-ul titlurilor de stat)

        # Funcția obiectiv: Vrem să MAXIMIZĂM Sharpe. 
        # Pentru că scriptul caută minimul matematic, îi dăm Sharpe Ratio NEGATIV.
        def negative_sharpe(weights, mean_ret, cov_mat, rf_rate):
            p_ret = np.sum(mean_ret * weights)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
            return -(p_ret - rf_rate) / p_vol

        # Constrângeri: Toți banii trebuie investiți (Suma procentelor = 100%)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Limite: Nu putem avea alocare negativă (Fără Short-Selling: limite 0 - 1)
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Punct de plecare (Presupunem că ești egal ponderat)
        init_guess = np.array(num_assets * [1. / num_assets])

        # Rulăm motorul de optimizare ML (SLSQP = Sequential Least Squares Programming)
        opt_results = minimize(negative_sharpe, init_guess, args=(mean_returns, cov_matrix, risk_free_rate), 
                               method='SLSQP', bounds=bounds, constraints=constraints)

        if not opt_results.success:
            return None, "AI-ul nu a găsit o soluție convergentă."

        # Extragem ponderile "magice" găsite de AI
        optimal_weights = opt_results.x
        opt_ret = np.sum(mean_returns * optimal_weights)
        opt_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        opt_sharpe = (opt_ret - risk_free_rate) / opt_vol

        # Formatăm rezultatele pentru interfață
        tickers = df_clean.columns.tolist()
        allocation = {tickers[i]: round(optimal_weights[i] * 100, 2) for i in range(num_assets)}

        return {
            "allocation": allocation,
            "expected_return": opt_ret * 100,
            "expected_volatility": opt_vol * 100,
            "sharpe_ratio": opt_sharpe
        }, "Optimizare finalizată."

    except Exception as e:
        print(f"Eroare AI Portofoliu: {e}")
        return None, str(e)

def detect_volume_anomaly_ai(df_t):
    """
    Machine Learning (Isolation Forest): Învață comportamentul 'normal'
    al acțiunii (Preț vs Volum vs Volatilitate) și detectează dacă
    mișcarea de azi este o anomalie generată de fonduri instituționale.
    """
    try:
        if len(df_t) < 30: return False # Prea puține date pentru antrenament
            
        # 1. Extragem datele relevante pentru AI
        df_ml = pd.DataFrame()
        df_ml['Return'] = df_t['Close'].pct_change() * 100
        df_ml['Volume'] = df_t['Volume']
        df_ml['Volatility'] = df_ml['Return'].rolling(window=5).std()
        
        df_ml = df_ml.dropna()
        if len(df_ml) < 20: return False

        # 2. Antrenăm modelul pe istoric (Zilele anterioare)
        X = df_ml[['Return', 'Volume', 'Volatility']].values
        
        # Setăm 5% rata de anomalii ('contamination' - cât de strict e filtrul)
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        df_ml['Anomaly'] = iso_forest.fit_predict(X)
        
        # 3. IsolationForest returnează -1 pentru o anomalie, 1 pentru normal
        current_status = df_ml['Anomaly'].iloc[-1]
        
        return current_status == -1
    except:
        return False

def calculate_and_plot_seasonality(hist_data):
    """
    Analizează istoricul prețurilor pentru a găsi tipare recurente (Sezonalitate).
    Calculează Win Rate-ul și Randamentul Mediu pentru fiecare lună a anului.
    """
    try:
        if hist_data is None or len(hist_data) < 200:
            return None, "Nu există suficient istoric pentru a calcula sezonalitatea (minim 1 an)."
            
        # Pregătim datele extrăgând luna și anul
        df = hist_data.copy()
        df['Month'] = df.index.month
        df['Year'] = df.index.year
        
        monthly_returns = []
        
        # Calculăm randamentul exact pentru fiecare lună din fiecare an istoric
        for year in df['Year'].unique():
            for month in range(1, 13):
                df_m = df[(df['Year'] == year) & (df['Month'] == month)]
                if not df_m.empty:
                    # Randamentul se calculează de la primul Open la ultimul Close din lună
                    ret = (df_m['Close'].iloc[-1] / df_m['Open'].iloc[0] - 1) * 100
                    monthly_returns.append({'Year': year, 'Month': month, 'Return': ret})
                    
        df_ret = pd.DataFrame(monthly_returns)
        if df_ret.empty: return None, "Eroare la calculul randamentelor lunare."
        
        months_labels = ['Ian', 'Feb', 'Mar', 'Apr', 'Mai', 'Iun', 'Iul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        stats = []
        
        # Agregăm datele pentru a obține statistici finale per lună
        for i in range(1, 13):
            m_data = df_ret[df_ret['Month'] == i]
            if len(m_data) > 0:
                win_rate = (m_data['Return'] > 0).sum() / len(m_data) * 100
                avg_ret = m_data['Return'].mean()
                stats.append({
                    'Luna': months_labels[i-1], 
                    'Win Rate (%)': win_rate, 
                    'Randament Mediu (%)': avg_ret, 
                    'Ani': len(m_data)
                })
                
        df_stats = pd.DataFrame(stats)
        
        # --- DESENĂM GRAFICUL INSTITUȚIONAL ---
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, 
                            subplot_titles=("Win Rate (%) - Frecvența Creșterilor", "Randament Mediu Lunar (%)"))
        
        # Grafic 1: Win Rate (Cât de des crește prețul în acea lună?)
        # Verde dacă > 60% șanse de câștig, Roșu dacă < 40%, Galben între
        colors_win = ['#3FB950' if w >= 60 else ('#F85149' if w <= 40 else '#D29922') for w in df_stats['Win Rate (%)']]
        fig.add_trace(go.Bar(x=df_stats['Luna'], y=df_stats['Win Rate (%)'], marker_color=colors_win, 
                             texttemplate='%{y:.0f}%', textposition='auto', name="Win Rate"), row=1, col=1)
        # Linia de "Echilibru" (50% șanse - dă cu banul)
        fig.add_hline(y=50, line_dash="dot", line_color="#8B949E", row=1, col=1) 
        
        # Grafic 2: Randament Mediu (Cât de mult crește/scade de obicei?)
        colors_ret = ['#3FB950' if r > 0 else '#F85149' for r in df_stats['Randament Mediu (%)']]
        fig.add_trace(go.Bar(x=df_stats['Luna'], y=df_stats['Randament Mediu (%)'], marker_color=colors_ret, 
                             texttemplate='%{y:.2f}%', textposition='auto', name="Randament"), row=2, col=1)
                             
        fig.update_layout(height=450, showlegend=False, template="plotly_dark", 
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=40, b=0))
        fig.update_yaxes(ticksuffix="%", row=1, col=1)
        fig.update_yaxes(ticksuffix="%", row=2, col=1)
        
        return fig, df_stats
    except Exception as e:
        return None, f"Eroare procesare sezonalitate: {e}"

@st.cache_data(ttl=900, show_spinner=False) # Cache activ 15 minute
def get_options_analysis_ai(ticker_sym):
    try:
        t = yf.Ticker(ticker_sym)
        
        # Pasul 1: Obținem datele de expirare (Primul apel API)
        expirations = t.options
        if not expirations: 
            return None, "Fără date de opțiuni disponibile pentru acest simbol."
            
        # Pasul 2: Obținem lanțul de opțiuni (Al doilea apel API - cel mai greu)
        opts = t.option_chain(expirations[0])
        calls, puts = opts.calls, opts.puts
        
        # Preluăm prețul curent rapid
        spot = t.fast_info.last_price
        if spot == 0: spot = calls['strike'].median() # Fallback dacă lipsește prețul

        # --- Calcule GEX & IV (identic cu ce ai deja) ---
        calls['gex'] = calls['openInterest'] * (spot / ((calls['strike'] - spot)**2 + 1))
        puts['gex'] = puts['openInterest'] * (spot / ((puts['strike'] - spot)**2 + 1)) * -1
        total_gex = calls['gex'].sum() + puts['gex'].sum()
        
        avg_iv = (calls['impliedVolatility'].mean() + puts['impliedVolatility'].mean()) / 2 * 100
        
        gex_verdict = "🛡️ GAMMA LUNGĂ: Stabilitate" if total_gex > 0 else "🔥 GAMMA SCURTĂ: Volatilitate"
        gex_color = "#3FB950" if total_gex > 0 else "#F85149"
        
        iv_txt_color = "#3FB950" if avg_iv < 25 else ("#D29922" if avg_iv < 50 else "#F85149")

        return {
            "expiration": expirations[0],
            "oi_pc_ratio": puts['openInterest'].sum() / calls['openInterest'].sum() if calls['openInterest'].sum() > 0 else 0,
            "vol_pc_ratio": puts['volume'].sum() / calls['volume'].sum() if calls['volume'].sum() > 0 else 0,
            "max_pain": spot, 
            "iv": avg_iv,
            "iv_color": iv_txt_color,
            "net_gex": total_gex,
            "gex_verdict": gex_verdict,
            "gex_color": gex_color,
            "timestamp": datetime.now().strftime("%H:%M") # Adăugăm ora salvării în cache
        }, "Succes"

    except Exception as e:
        # Dacă Yahoo ne blochează (Eroare 429), returnăm un mesaj specific
        error_msg = str(e)
        if "429" in error_msg or "Too Many Requests" in error_msg:
            return None, "🚨 LIMITĂ ATINSĂ: Yahoo Finance a restricționat temporar accesul. Reîncearcă în 15 minute."
        return None, f"Eroare Opțiuni: {error_msg}"

def calculate_master_ai_score(info, hist, h_score, mos_val, inst_pct, rvol, s_score, opt_data, spread, z_score, q_ratio, regime_msg):
    """
    Sistem de Decizie Multicriterial (10 Piloni).
    Punctaj maxim 100. Include penalizări drastice (Red Flags) pentru a preveni 'capcanele valorii'.
    """
    score = 0
    reasons = []

    # 1. EVALUARE (DCF Margin of Safety) - MAX 20 puncte
    if mos_val > 25:
        score += 20
        reasons.append("✅ **Subevaluare Masivă:** Discount excelent față de valoarea intrinsecă (Model DCF).")
    elif mos_val > 5:
        score += 10
        reasons.append("✅ **Preț Corect:** Acțiunea se tranzacționează în limite rezonabile de evaluare.")
    elif mos_val < -15:
        score += 0
        reasons.append("🚨 **Supraevaluare Critică:** Prețul este nejustificat de mare matematic. Premium periculos.")
    else:
        score += 5
        reasons.append("⚖️ **Evaluare Neutră:** Fără marjă de siguranță clară împotriva erorilor.")

    # 2. SĂNĂTATE FINANCIARĂ & FALIMENT (Max 15 puncte)
    if h_score >= 8:
        score += 15
        reasons.append(f"✅ **Bilanț Fortăreață:** Scor de sănătate extrem de solid ({h_score}/10).")
    elif h_score >= 5:
        score += 7
        reasons.append(f"⚖️ **Bilanț Mediu:** Datorii și lichiditate în limite acceptabile ({h_score}/10).")
    else:
        reasons.append(f"🚨 **Risc de Bilanț:** Sănătate financiară precară, grad de îndatorare mare ({h_score}/10).")
        
    if z_score < 1.8:
        score -= 20 # Penalizare fatală
        reasons.append("🚨 **ALERTĂ ALTMAN Z:** Risc statistic sever de faliment sau restructurare în următorii 2 ani!")

    # 3. CALITATEA PROFITULUI (Cash-Flow) (Max 10 puncte)
    if q_ratio > 1.0:
        score += 10
        reasons.append("✅ **Cash Machine:** Compania generează mai mult cash real (în bancă) decât profit contabil.")
    elif q_ratio > 0.7:
        score += 5
        reasons.append("✅ **Profit Calitativ:** Fluxurile de numerar susțin bine câștigurile raportate.")
    else:
        score -= 10 # Penalizare
        reasons.append("⚠️ **Profit pe Hârtie:** Firma raportează profit, dar nu încasează cash. Risc de contabilitate creativă.")

    # 4. SMART MONEY & VOLUM (Max 15 puncte)
    if inst_pct > 60:
        score += 10
        reasons.append(f"✅ **Dominare Instituțională:** Balenele dețin {inst_pct:.0f}%, oferind stabilitate prețului.")
    elif inst_pct < 30:
        reasons.append(f"⚠️ **Lipsă Instituții:** Doar {inst_pct:.0f}% la fonduri. Preț expus la speculă de retail.")
    else:
        score += 5
        
    if rvol > 1.3:
        score += 5
        reasons.append("✅ **Momentum de Volum:** Activitate anormal de mare recent. Banii inteligenți se mișcă.")

    # 5. PIAȚA DE OPȚIUNI & IV (Max 15 puncte)
    if opt_data:
        oi_pc = opt_data.get('oi_pc_ratio', 1)
        iv = opt_data.get('iv', 50)
        
        if oi_pc < 0.7:
            score += 10
            reasons.append("✅ **Opțiuni Bullish:** Market Makerii văd pariuri masive pe creștere.")
        elif oi_pc > 1.1:
            score -= 10
            reasons.append("🚨 **Frică în Opțiuni:** Număr disproporționat de contracte Put. Se așteaptă o cădere.")
        else:
            score += 5
            
        if iv < 30:
            score += 5
            reasons.append(f"✅ **Volatilitate (IV) Scăzută:** Primele de asigurare sunt ieftine ({iv:.1f}%).")
        elif iv > 50:
            score -= 10 # Penalizare
            reasons.append(f"⚠️ **IV Extem ({iv:.1f}%):** Opțiunile sunt foarte scumpe. Piața așteaptă șocuri de preț.")

    # 6. SENTIMENT NLP (Max 10 puncte)
    if s_score > 0.15:
        score += 10
        reasons.append("✅ **Sentiment Media AI:** Știrile procesate de FinBERT sunt clar optimiste.")
    elif s_score < -0.15:
        score -= 5
        reasons.append("🚨 **Presă Negativă:** Titlurile recente generează frică și presiune de vânzare.")
    else:
        score += 5

    # 7. TEHNIC & TREND (Max 10 puncte)
    try:
        current_price = hist['Close'].iloc[-1]
        sma50 = hist['SMA50'].iloc[-1]
        rsi = hist['RSI'].iloc[-1]
        
        if current_price > sma50:
            score += 5
            reasons.append("✅ **Trend Tehnic:** Acțiunea navighează curat peste media mobilă (SMA 50).")
        else:
            reasons.append("⚠️ **Trend Descendent:** Prețul a rupt suportul de 50 de zile. Momentum negativ.")
            
        if rsi < 35:
            score += 5
            reasons.append("✅ **Oversold (Supra-vândut):** Indicatorul RSI sugerează o panică exagerată. Bun de intrare.")
        elif rsi > 75:
            score -= 15 # Penalizare gravă
            reasons.append(f"🚨 **Overbought (RSI {rsi:.0f}):** Acțiunea este extrem de supra-cumpărată. Corecție iminentă!")
        else:
            score += 5
    except: pass

    # 8. MACROECONOMIE & REGIM (Max 5 puncte)
    if spread > 0:
        score += 5
    else:
        score -= 10
        reasons.append("🚨 **Avertisment Macro:** Curba randamentelor 10Y-2Y inversată. Risc sistemic!")
        
    if "PANICĂ" in str(regime_msg).upper():
        score -= 15 # Penalizare drastică
        reasons.append("🚨 **Regim de Piață K-Means:** AI-ul detectează vânzări emoționale generalizate în piață.")

    # Asigurăm intervalul strict 1-100
    final_score = max(1, min(100, score))

    # Ponderare Verdict Final
    if final_score >= 75:
        action, color = "CUMPĂRĂ (STRONG BUY)", "#3FB950"
        advice = "Toate filtrele cuantice și fundamentale sunt aliniate pozitiv. Risc minim, potențial maxim."
    elif final_score >= 55:
        action, color = "ACUMULEAZĂ / HOLD", "#238636"
        advice = "Companie robustă, dar prezintă 1-2 puncte slabe (ex: preț tehnic ridicat sau volatilitate). Intrări treptate."
    elif final_score >= 40:
        action, color = "ATENȚIE (NEUTRU)", "#D29922"
        advice = "Semnale contradictorii puternice. De exemplu: fundamente bune, dar regim de piață panicat."
    else:
        action, color = "EVITĂ SAU VINDE", "#F85149"
        advice = "Steaguri roșii critice: Risc de faliment, supraevaluare masivă, lipsă cash sau fugă instituțională."

    # Sortăm lista ca să punem Steagurile Roșii (🚨) sus de tot
    reasons.sort(key=lambda x: "🚨" not in x)

    return final_score, action, color, advice, reasons

def calculate_master_macro_verdict(df_sectors, credit_ratio_series, corr_matrix, sentiment_score, yield_spread, vix_val):
    """
    Algoritm de Sinteză Macro V2 (Quant Grade).
    Adaugă VIX și corelația Dolarului ca filtre de siguranță.
    """
    score = 50 
    reasons = []

    # 1. ANALIZA PIEȚEI DE CREDIT (HYG/IEF) - 30%
    if not credit_ratio_series.empty:
        curr_r = credit_ratio_series.iloc[-1]
        sma_20 = credit_ratio_series.rolling(20).mean().iloc[-1]
        if curr_r < sma_20:
            score -= 20
            reasons.append("🚨 **Risc de Credit:** Piața de obligațiuni (banii deștepți) se retrage spre siguranță.")
        else:
            score += 10
            reasons.append("✅ **Credit Sănătos:** Băncile finanțează activ economia.")

    # 2. PILONUL VIX (Frica / Complăcere) - 20%
    if vix_val > 25:
        score -= 20
        reasons.append(f"🚨 **Panică (VIX {vix_val:.1f}):** Frica este ridicată. Piața este vulnerabilă la căderi bruște.")
    elif vix_val < 15:
        score += 10
        reasons.append(f"✅ **Liniște (VIX {vix_val:.1f}):** Volatilitatea este scăzută, favorizând trendul ascendent.")
    else:
        reasons.append(f"⚖️ **VIX Stabil ({vix_val:.1f}):** Frica este în limite normale.")

    # 3. PILONUL DOLAR (Corelația SPY/UUP) - 15%
    try:
        spy_uup = corr_matrix.loc['Acțiuni (SPY)', 'Dolar (UUP)']
        if spy_uup < -0.7:
            score -= 15
            reasons.append(f"⚠️ **Dolar Dominant:** Corelație inversă severă ({spy_uup:.2f}). Orice creștere a Dolarului va lovi bursa.")
        else:
            score += 5
    except: pass

    # 4. MONEY FLOW SECTORIAL - 15%
    if not df_sectors.empty:
        top_sectors = df_sectors.tail(3)['Sector'].tolist()
        agg_count = sum(1 for s in top_sectors if s in ['Tehnologie', 'Consum Discreționar', 'Financiar', 'Comunicații'])
        if agg_count >= 2:
            score += 15
            reasons.append("🚀 **Apetit Risc:** Capitalul migrează spre sectoarele de creștere.")
        elif any(s in top_sectors for s in ['Utilități', 'Consum de Bază']):
            score -= 10
            reasons.append("🛡️ **Rotație Defensivă:** Investitorii caută adăpost în sectoarele sigure.")

    # 5. MACRO & SENTIMENT - 20%
    if yield_spread < 0:
        score -= 15
        reasons.append("🚩 **Yield Curve:** Curba inversată semnalează risc de recesiune.")
    if sentiment_score > 0.15: score += 10
    elif sentiment_score < -0.15: score -= 10

    # Normalizare și Verdict
    final_score = max(1, min(100, score))
    
    if final_score >= 70:
        label, color, desc = "INVESTEȘTE", "#3FB950", "Condiții ideale: Lichiditate mare, frica scăzută. Mediul macro este extrem de favorabil. Riscurile sistemice sunt scăzute."
    elif final_score >= 45:
        label, color, desc = "AȘTEAPTĂ", "#D29922", "Echilibru fragil. Nu forța intrări noi azi. Piața este în tranziție. Există semnale contradictorii."
    else:
        label, color, desc = "CASH / SELL", "#F85149", "🚨 PERICOL: Riscurile sistemice domină piața. Condițiile macro sunt periculoase. Capitalul se retrage rapid"

    return final_score, label, color, desc, reasons

def calculate_iv_rank_percentile(ticker_sym, current_iv):
    """
    Calculează IV Rank și IV Percentile folosind datele istorice de volatilitate.
    """
    try:
        t = yf.Ticker(ticker_sym)
        # Avem nevoie de istoric pentru a calcula volatilitatea istorică (HV) ca proxy dacă IV lipsește
        hist = t.history(period="1y")
        if hist.empty: return 0, 0
        
        # Calculăm volatilitatea realizată (HV) pe 252 de zile
        returns = np.log(hist['Close'] / hist['Close'].shift(1))
        vol_hist = returns.rolling(window=21).std() * np.sqrt(252) * 100
        vol_hist = vol_hist.dropna()

        if vol_hist.empty: return 0, 0

        # IV Rank: (IV_curent - IV_min) / (IV_max - IV_min)
        iv_min = vol_hist.min()
        iv_max = vol_hist.max()
        iv_rank = ((current_iv - iv_min) / (iv_max - iv_min)) * 100
        
        # IV Percentile: Procentul de zile din an cu IV mai mic decât cel de azi
        iv_percentile = (vol_hist < current_iv).mean() * 100
        
        return max(0, min(100, iv_rank)), max(0, min(100, iv_percentile))
    except:
        return 0, 0

def get_detailed_ownership_and_execs(ticker_sym):
    try:
        t = yf.Ticker(ticker_sym)
        info = t.info
        
        # 1. Extragere CEO
        ceo_name = "N/A"
        officers = info.get('companyOfficers', [])
        for officer in officers:
            if 'ceo' in officer.get('title', '').lower():
                ceo_name = officer.get('name', 'N/A')
                break

        # 2. PROCESARE SIGURĂ ACȚIONARI
        major = t.major_holders
        insider_pct, inst_pct = 0.0, 0.0
        
        if major is not None and not major.empty:
            major_df = major.reset_index().astype(str)
            for _, row in major_df.iterrows():
                row_str = " ".join(row.values).lower()
                # Extragem valoarea numerică curată
                raw_val = row.iloc[0].replace('%', '').replace(',', '') if isinstance(row.iloc[0], str) else row.iloc[0]
                val = pd.to_numeric(raw_val, errors='coerce')
                
                if not pd.isna(val):
                    # --- LOGICĂ DE NORMALIZARE ANTI-1400% ---
                    # Dacă valoarea e sub 1.0 (ex: 0.51), e fracție și o înmulțim cu 100
                    # Dacă e peste 1.0 (ex: 51.78), e deja procent și o lăsăm așa
                    val_clean = val * 100 if 0 < val <= 1.0 else val
                    
                    if 'insider' in row_str: insider_pct = val_clean
                    if 'instituti' in row_str or 'held by inst' in row_str: inst_pct = val_clean
        
        # Fallback: Dacă tabelul major_holders e gol, luăm din info direct
        if inst_pct == 0:
            val_info = info.get('heldPercentInstitutions', 0)
            inst_pct = val_info * 100 if val_info <= 1.0 else val_info

        retail_pct = max(0, 100 - (insider_pct + inst_pct))
        df_summary = pd.DataFrame({
            "Tip Acționar": ["Insideri", "Instituții", "Retail / Alții"],
            "Procent (%)": [insider_pct, inst_pct, retail_pct]
        })

        # 3. Lista nominală (Top 10)
        inst_holders = t.institutional_holders
        if inst_holders is not None and not inst_holders.empty:
            df_details = inst_holders[['Holder', 'pctHeld', 'Value']].copy()
            df_details['pctHeld'] = df_details['pctHeld'] * 100
            df_details.columns = ['Nume Acționar', 'Deținere (%)', 'Valoare ($)']
        else:
            df_details = pd.DataFrame()

        return df_summary, df_details, ceo_name, inst_pct
    except:
        return None, None, "N/A", 0.0

def generate_macro_ai_summary(vix_val, yield_spread, credit_ratio_series, df_sectors, corr_matrix, macro_data, df_fred):
    """
    Sinteză Macro de nivel Instituțional Suprem (Comprehensive). 
    Corelează datele și raportează statusul pentru ABSOLUT TOȚI parametrii vitali (Piață + Economie Reală).
    """
    bullets = []
    
    # =========================================================
    # 1. EXTRAGERE DATE PIAȚĂ (Trend pe o lună / 20 zile)
    # =========================================================
    def get_trend(ticker):
        try:
            if isinstance(macro_data.columns, pd.MultiIndex):
                if ticker in macro_data.columns.levels[0]:
                    series = macro_data[ticker]['Close'].dropna()
                else: return 0
            else:
                series = macro_data['Close'].dropna()
            
            if len(series) >= 20: 
                return (series.iloc[-1] / series.iloc[-20]) - 1
            elif len(series) >= 5:
                return (series.iloc[-1] / series.iloc[-5]) - 1
            return 0
        except: return 0

    gold_trend = get_trend('GC=F')
    oil_trend = get_trend('CL=F')
    copper_trend = get_trend('HG=F')
    dxy_trend = get_trend('DX-Y.NYB')

    # =========================================================
    # 2. EXTRAGERE DATE FRED (Economia Reală)
    # =========================================================
    fred_dict = {}
    if df_fred is not None and not df_fred.empty:
        fred_dict = {row['_simbol']: row for _, row in df_fred.iterrows()}
        
    cpi_change = fred_dict.get('CPIAUCSL', {}).get('_change_raw', 0)
    unrate_curr = fred_dict.get('UNRATE', {}).get('Valoare Curentă', 4.0)
    unrate_change = fred_dict.get('UNRATE', {}).get('_change_raw', 0)
    nfp_change = fred_dict.get('PAYEMS', {}).get('_change_raw', 0)
    retail_change = fred_dict.get('RSAFS', {}).get('_change_raw', 0)
    fed_change = fred_dict.get('FEDFUNDS', {}).get('_change_raw', 0)
    indpro_change = fred_dict.get('INDPRO', {}).get('_change_raw', 0)

    cross_asset_flags = 0 

    # =========================================================
    # STRATUL 1: SUPRA-CORELAȚII (BOMBE MACROECONOMICE)
    # =========================================================
    if oil_trend > 0.05 and cpi_change > 0.2:
        bullets.append({"icon": "🌪️", "color": "#F85149", "text": f"BUCLA INFLAȚIEI: Petrolul crește violent (+{oil_trend*100:.1f}%), iar datele CPI confirmă scumpirile. FED este blocată. Mediu toxic pentru acțiunile de creștere (Growth)."})
        cross_asset_flags += 1
        
    if cpi_change > 0.2 and unrate_change > 0:
        bullets.append({"icon": "☠️", "color": "#F85149", "text": "ALERTĂ STAGFLAȚIE: Șomajul crește, dar prețurile (CPI) refuză să scadă. Este cel mai prost mediu economic: acțiunile și obligațiunile pot pierde valoare simultan."})
        cross_asset_flags += 1

    if gold_trend > 0.03 and dxy_trend > 0.015:
        bullets.append({"icon": "🥇", "color": "#FFAB00", "text": f"PARADOXUL FRICII: Aurul (+{gold_trend*100:.1f}%) și Dolarul (+{dxy_trend*100:.1f}%) cresc simultan. Marile fonduri acumulează orice activ de siguranță, semn de panică sistemică."})
        cross_asset_flags += 1

    if copper_trend < -0.04 and indpro_change < -0.5:
        bullets.append({"icon": "📉", "color": "#F85149", "text": "CONTRACȚIE EXTREMĂ: Scăderea cuprului se aliniază cu prăbușirea producției industriale (FRED). Motorul global frânează puternic."})
        cross_asset_flags += 1

    # =========================================================
    # STRATUL 2: ECONOMIA REALĂ (Datele FRED)
    # =========================================================
    # Inflație
    if cpi_change > 0.2:
        bullets.append({"icon": "🔥", "color": "#F85149", "text": f"INFLAȚIE (CPI): Accelerare ({cpi_change:+.2f} pct). Prețurile nu cedează, menținând presiunea pe consumatori și pe dobânzile FED."})
    elif cpi_change <= 0:
        bullets.append({"icon": "❄️", "color": "#3FB950", "text": f"INFLAȚIE (CPI): Dezinflație confirmată ({cpi_change:+.2f} pct). Oferă băncii centrale motive să reducă dobânzile (Dovish)."})
    else:
        bullets.append({"icon": "⚖️", "color": "#8B949E", "text": f"INFLAȚIE (CPI): Creștere sub control ({cpi_change:+.2f} pct). Prețurile se stabilizează fără șocuri majore."})

    # Piața Muncii
    if unrate_change > 0 or nfp_change < 50:
        bullets.append({"icon": "⚠️", "color": "#D29922", "text": f"PIAȚA MUNCII: Se răcește. Șomajul e la {unrate_curr:.1f}%, iar joburile noi scârțâie ({nfp_change:+.0f}K). Risc de recesiune dacă trendul continuă."})
    elif nfp_change > 150:
        bullets.append({"icon": "💪", "color": "#3FB950", "text": f"PIAȚA MUNCII: Robustă (Soft Landing). Crearea a {nfp_change:+.0f}K joburi arată o economie de fier care ignoră dobânzile mari."})
    else:
        bullets.append({"icon": "⚖️", "color": "#8B949E", "text": f"PIAȚA MUNCII: Echilibrată ({nfp_change:+.0f}K joburi adăugate). Angajările sunt la un nivel sustenabil, fără surprize."})

    # Consum
    if retail_change < -0.5:
        bullets.append({"icon": "🛒", "color": "#F85149", "text": f"CONSUM: Fragil. Vânzările de retail au scăzut ({retail_change:+.2f} pct). Motorul economiei americane dă semne de oboseală."})
    elif retail_change > 0.5:
        bullets.append({"icon": "🛍️", "color": "#3FB950", "text": f"CONSUM: Puternic. Populația cheltuie masiv ({retail_change:+.2f} pct), ceea ce va susține direct profiturile companiilor."})
    else:
        bullets.append({"icon": "⚖️", "color": "#8B949E", "text": f"CONSUM: Stabil. Vânzările de retail sunt constante ({retail_change:+.2f} pct). Fără anomalii în comportamentul de cumpărare."})

    # Decizia FED
    if fed_change < -0.1:
        bullets.append({"icon": "💸", "color": "#3FB950", "text": f"POLITICA FED: Ciclu de relaxare (Tăieri de {fed_change:+.2f} pct). Banii mai ieftini acționează ca un stimulent major pentru bursa de acțiuni."})
    elif fed_change > 0.1:
        bullets.append({"icon": "🏦", "color": "#F85149", "text": f"POLITICA FED: Restrictivă (Creșteri de {fed_change:+.2f} pct). Dobânzile mari atrag capitalul dinspre acțiuni spre depozite și obligațiuni."})
    else:
        bullets.append({"icon": "⚖️", "color": "#8B949E", "text": "POLITICA FED: Dobânda menținută. Banca Centrală este în faza de așteptare ('Wait and See')."})

    # =========================================================
    # STRATUL 3: PIAȚA FINANCIARĂ (Dolar, Petrol, Cupru, Opțiuni)
    # =========================================================
    
    if dxy_trend > 0.015 and cross_asset_flags == 0:
        bullets.append({"icon": "💵", "color": "#F85149", "text": f"DOLAR: Puternic (+{dxy_trend*100:.1f}% într-o lună). Moneda americană acționează ca un obstacol major pentru S&P 500."})
    elif dxy_trend < -0.015:
        bullets.append({"icon": "💸", "color": "#3FB950", "text": f"DOLAR: Slab (-{abs(dxy_trend)*100:.1f}%). Oferă o fereastră de oportunitate masivă pentru piețele globale și cripto."})
    else:
        bullets.append({"icon": "⚖️", "color": "#8B949E", "text": f"DOLAR: Consolidare ({dxy_trend*100:+.1f}%). Valuta se tranzacționează în interval neutru."})

    if oil_trend > 0.06:
        bullets.append({
            "icon": "🛢️", 
            "color": "#F85149", # Schimbăm în Roșu (era Galben) pentru că +40% e critic
            "text": f"ȘOC PETROLIER: Țițeiul a explodat (+{oil_trend*100:.1f}%). Această creștere violentă a costurilor la energie va pune o presiune imensă pe inflație și va forța FED-ul să rămână Hawkish."
        })
    elif oil_trend < -0.06:
        bullets.append({
            "icon": "⛽", 
            "color": "#3FB950", 
            "text": f"RELAXARE ENERGETICĂ: Petrolul scade semnificativ (-{abs(oil_trend)*100:.1f}%). Este un semnal puternic de dezinflație care ajută marjele de profit ale companiilor."
        })
    else:
        # Mesajul de stabilitate apare DOAR dacă variația e mică
        bullets.append({
            "icon": "⚖️", 
            "color": "#8B949E", 
            "text": f"PETROL: Preț stabil ({oil_trend*100:+.1f}%). Sectorul energetic nu prezintă șocuri imediate pe lanțul global."
        })

    if yield_spread < 0:
        bullets.append({"icon": "🚨", "color": "#F85149", "text": f"OBLIGAȚIUNI: Curba 10Y-3M rămâne inversată ({yield_spread:.2f}). Semnalul istoric de recesiune este în continuare aprins."})
    else:
        bullets.append({"icon": "📈", "color": "#3FB950", "text": f"OBLIGAȚIUNI: Curba dobânzilor e normală ({yield_spread:.2f}). Creditarea corporativă este sănătoasă."})

    if vix_val > 22:
        bullets.append({"icon": "⚡", "color": "#F85149", "text": f"VIX (OPȚIUNI): Panică ({vix_val:.1f}). Cererea pentru contracte Put explodează. Volatilitatea e la putere."})
    elif vix_val < 14:
        bullets.append({"icon": "🧘", "color": "#3FB950", "text": f"VIX (OPȚIUNI): Liniște absolută ({vix_val:.1f}). Piața e calmă, dar favorizează apariția unor „Black Swans”."})
    else:
        bullets.append({"icon": "⚖️", "color": "#8B949E", "text": f"VIX (OPȚIUNI): Moderație ({vix_val:.1f}). Investitorii nu evaluează riscuri imediate majore."})

    # =========================================================
    # SORTARE (Sistem Profesional de Prioritizare Vizuală)
    # =========================================================
    def sort_key(b):
        if b['color'] == "#F85149": return 0 # Roșu Critic (Pe primul loc)
        if b['color'] == "#FFAB00": return 1 # Portocaliu Avertisment
        if b['color'] == "#D29922": return 2 # Galben Atenție
        if b['color'] == "#3FB950": return 3 # Verde (Pozitiv)
        return 4 # Gri (Neutru/Stabil) pus la final ca informare
        
    bullets.sort(key=sort_key)
    
    # Returnăm TOATE argumentele posibile (până la 15)
    return bullets[:15]

def get_gemini_sentiment_label(title):
    """
    Analiză individuală Gemini pentru etichetare UI.
    Returnează: (text_etichetă, clasă_css, iconiță)
    """
    prompt = f"""
    Ești un analist financiar senior. Evaluează sentimentul strict pentru acest titlu: "{title}"
    
    REGULI:
    - "Shares up", "Surge", "AI Boom", "Beat expectations", "Target raised" sunt POZITIVE.
    - "Shares down", "Missed", "Job cuts", "Lawsuit" sunt NEGATIVE.
    
    Răspunde DOAR cu un număr: 1 (Pozitiv), -1 (Negativ) sau 0 (Neutru).
    """
    try:
        response = model.generate_content(prompt)
        val = response.text.strip()
        # Curățăm răspunsul în caz că AI-ul mai scrie text pe lângă număr
        score = 1 if "1" in val and "-" not in val else (-1 if "-1" in val else 0)
        
        if score == 1: return "Pozitiv", "impact-poz", "📈"
        if score == -1: return "Negativ", "impact-neg", "📉"
        return "Neutru", "impact-neu", "➡️"
    except:
        return "Neutru", "impact-neu", "➡️"    
