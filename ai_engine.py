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

# Incarcare model de sentiment specializat pe finante (FinBERT)
@st.cache_resource
def get_sentiment_pipeline():
    """Incarcă modelul FinBERT optimizat pentru analiză financiară."""
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_sentiment_ai(news_list):
    """
    Analizează contextul știrilor (FinBERT), cu Time Decay 
    (știrile noi au un impact matematic mai mare decât cele vechi).
    """
    if not news_list: return 0.0
    try:
        pipe = get_sentiment_pipeline()
        
        # Luăm primele 10 titluri (presupunând că sunt sortate de la nou la vechi)
        titles = [n['title'] for n in news_list[:10]]
        results = pipe(titles)
        
        weighted_score = 0
        total_weight = 0
        
        # Iterăm prin rezultate. 'i' este indexul (0 e cea mai nouă știre, 9 e cea mai veche)
        for i, r in enumerate(results):
            # Exponențial: 0.8^0 = 1.0 (100% pondere pt ultima știre)
            # 0.8^1 = 0.8 (80% pondere pt a doua), etc.
            decay_factor = (0.8) ** i 
            
            # Convertim eticheta text într-un scor numeric (-1 la 1)
            if r['label'] == 'positive':
                raw_val = r['score']
            elif r['label'] == 'negative':
                raw_val = -r['score']
            else:
                raw_val = 0 # Neutru
            
            weighted_score += (raw_val * decay_factor)
            total_weight += decay_factor
            
        # Returnăm media ponderată
        return weighted_score / total_weight if total_weight > 0 else 0.0
    except Exception as e:
        print(f"Eroare AI Sentiment: {e}")
        return 0.0

def predict_stock_price(df):
    """
    Predicție Machine Learning pe 90 de zile folosind Facebook Prophet.
    Acum modelul este MULTIVARIAT (include și variațiile de Volum).
    """
    try:
        # 1. Fallback de siguranță: dacă nu avem date de volum (se poate întâmpla la unii indici)
        if 'Volume' not in df.columns or df['Volume'].sum() == 0:
            df_p = df.reset_index()[['Date', 'Close']]
            df_p.columns = ['ds', 'y']
            df_p['ds'] = df_p['ds'].dt.tz_localize(None)
            m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True, changepoint_prior_scale=0.05)
            m.fit(df_p)
            future = m.make_future_dataframe(periods=90)
            return m.predict(future)

        # 2. Modelul Avansat (cu Volum ca Regresor Extern)
        df_p = df.reset_index()[['Date', 'Close', 'Volume']]
        df_p.columns = ['ds', 'y', 'volume_regressor']
        df_p['ds'] = df_p['ds'].dt.tz_localize(None)
        
        m = Prophet(
            daily_seasonality=False, 
            weekly_seasonality=True, 
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        # Învățăm modelul să coreleze prețul cu volumele tranzacționate
        m.add_regressor('volume_regressor')
        m.fit(df_p)
        
        # 3. Pregătim dataframe-ul pentru proiecția pe următoarele 90 de zile
        future = m.make_future_dataframe(periods=90)
        
        # Estimăm volumul viitor ca fiind media volumelor din ultimele 20 de zile
        avg_recent_vol = df_p['volume_regressor'].tail(20).mean()
        historical_vols = df_p['volume_regressor'].values
        future_vols = np.full(90, avg_recent_vol) # Umplem cele 90 de zile viitoare cu media
        
        # Combinăm array-ul istoric cu cel viitor și îl adăugăm în DF-ul 'future'
        future['volume_regressor'] = np.concatenate((historical_vols, future_vols))
        
        # Generăm predicția
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
    Sintetizează datele fundamentale din app.py cu datele AI.
    Include și s_score (Sentiment Score) în logica SWOT.
    """
    swot = {"Strengths": [], "Weaknesses": [], "Opportunities": [], "Threats": []}
    
    # --- STRENGTHS ---
    if h_score >= 8: swot["Strengths"].append("Sănătate financiară de elită.")
    if z_val > 3.0: swot["Strengths"].append("Risc de insolvență neglijabil (Z-Score safe).")
    if s_score > 0.3: swot["Strengths"].append("Sentiment pozitiv puternic în media financiară.")

    # --- WEAKNESSES ---
    de = info.get('debtToEquity', 0) or 0
    if de > 150: swot["Weaknesses"].append("Levier ridicat (Îndatorare peste medie).")
    if s_score < -0.2: swot["Weaknesses"].append("Sentiment negativ dominant în presă.")
    if alpha and alpha < -0.05: swot["Weaknesses"].append("Subperformanță cronică față de piață.")

    # --- OPPORTUNITIES ---
    if mos_val > 25: swot["Opportunities"].append(f"Sub-evaluare masivă ({mos_val:.1f}% discount).")
    if s_score > 0.1 and mos_val > 10: swot["Opportunities"].append("Convergență: Preț atractiv + Sentiment în creștere.")

    # --- THREATS ---
    if mos_val < -20: swot["Threats"].append("Supra-evaluare critică (Risc de corecție).")
    if z_val < 1.8: swot["Threats"].append("Vulnerabilitate structurală (Zona de risc financiar).")
    
    return swot

def detect_market_regime_ai(hist_data):
    """
    Folosește K-Means Clustering (Machine Learning) pentru a clasifica
    automat regimul curent al pieței pe baza volatilității și randamentului.
    """
    try:
        if hist_data is None or len(hist_data) < 50:
            return "Date insuficiente pentru ML", "#8B949E"

        # 1. Extragem "Trăsăturile" (Features) pentru modelul AI
        df_ml = pd.DataFrame()
        # Randamentul zilnic
        df_ml['Return'] = hist_data['Close'].pct_change() * 100
        # Volatilitatea (cât de agresiv se mișcă prețul)
        df_ml['Volatility'] = df_ml['Return'].rolling(window=10).std() 
        df_ml = df_ml.dropna()
        
        # 2. Antrenăm modelul K-Means (Găsește 3 comportamente distincte ale pieței)
        X = df_ml[['Return', 'Volatility']].values
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df_ml['Cluster'] = kmeans.fit_predict(X)
        
        # 3. Analizăm centrele clusterelor pentru a ști care e "Panică" și care e "Bull"
        centers = kmeans.cluster_centers_
        
        # Clusterul cu cea mai mare volatilitate este clar regimul de Panică
        panic_cluster = np.argmax(centers[:, 1])
        # Clusterul cu cel mai mare randament mediu este regimul Bull
        bull_cluster = np.argmax(centers[:, 0])
        
        # Ce a decis AI-ul pentru ZIUA DE AZI?
        current_cluster = df_ml['Cluster'].iloc[-1]
        
        # 4. Formulăm Verdictul
        if current_cluster == panic_cluster:
            return "🚨 REGIM PANICĂ (Volatilitate Extremă - Prudență maximă)", "#F85149"
        elif current_cluster == bull_cluster:
            return "🚀 REGIM TREND ASCENDENT (Acumulare instituțională)", "#3FB950"
        else:
            return "⚖️ REGIM CONSOLIDARE (Zgomot de piață / Așteptare)", "#D29922"

    except Exception as e:
        print(f"Eroare AI Market Regime: {e}")
        return "Model AI Offline", "#8B949E"
    
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
