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

def get_options_analysis_ai(ticker_sym):
    """
    Analiză cantitativă PRO a pieței de opțiuni.
    Extrage OI, Volume, IV și calculează Max Pain + Stare IV.
    """
    try:
        t = yf.Ticker(ticker_sym)
        expirations = t.options
        if not expirations:
            return None, "Nu există date despre opțiuni."
            
        target_exp = expirations[0] 
        opts = t.option_chain(target_exp)
        calls, puts = opts.calls, opts.puts
        
        # 1. Calcule Put/Call
        total_call_oi = calls['openInterest'].sum()
        total_put_oi = puts['openInterest'].sum()
        oi_pc_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        total_call_vol = calls['volume'].sum()
        total_put_vol = puts['volume'].sum()
        vol_pc_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 0
        
        # 2. Volatilitate Implicită (IV) medie
        avg_iv = (calls['impliedVolatility'].mean() + puts['impliedVolatility'].mean()) / 2
        iv_val = avg_iv * 100
        
        # Determinăm "starea" IV pentru vitezometru
        # Praguri standard: <20% Mic (Ieftin), 20-45% Mediu, >45% Mare (Scump)
        if iv_val < 20:
            iv_status = "IEFTIN"
            iv_color = "#3FB950"
        elif iv_val < 45:
            iv_status = "MODERAT"
            iv_color = "#D29922"
        else:
            iv_status = "SCUMP"
            iv_color = "#F85149"
        
        # 3. Calcul MAX PAIN
        all_strikes = sorted(list(set(calls['strike']).union(set(puts['strike']))))
        def calculate_pain(strike):
            c_loss = ((strike - calls[calls['strike'] < strike]['strike']) * calls[calls['strike'] < strike]['openInterest']).sum()
            p_loss = ((puts[puts['strike'] > strike]['strike'] - strike) * puts[puts['strike'] > strike]['openInterest']).sum()
            return c_loss + p_loss
        
        pain_values = [calculate_pain(s) for s in all_strikes]
        max_pain_price = all_strikes[pain_values.index(min(pain_values))]
        
        return {
            "expiration": target_exp,
            "oi_pc_ratio": oi_pc_ratio,
            "vol_pc_ratio": vol_pc_ratio,
            "max_pain": max_pain_price,
            "iv": iv_val,
            "iv_status": iv_status,
            "iv_color": iv_color,
            "total_calls": total_call_oi,
            "total_puts": total_put_oi
        }, "Succes"
    except Exception as e:
        return None, f"Eroare: {str(e)}"

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
