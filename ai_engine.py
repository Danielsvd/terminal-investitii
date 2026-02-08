import pandas as pd
import numpy as np
from prophet import Prophet
from transformers import pipeline
import plotly.graph_objects as go
import streamlit as st

# Incarcare model de sentiment specializat pe finante (FinBERT)
@st.cache_resource
def get_sentiment_pipeline():
    """Incarcă modelul FinBERT optimizat pentru analiză financiară."""
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_sentiment_ai(news_list):
    """Analizează contextul știrilor folosind NLP Transformer (FinBERT)."""
    if not news_list: return 0.0
    try:
        pipe = get_sentiment_pipeline()
        # Luăm primele 10 titluri pentru relevanță
        titles = [n['title'] for n in news_list[:10]]
        results = pipe(titles)
        
        score = 0
        for r in results:
            if r['label'] == 'positive': score += r['score']
            elif r['label'] == 'negative': score -= r['score']
        
        return score / len(results)
    except Exception as e:
        print(f"Eroare AI Sentiment: {e}")
        return 0.0

def predict_stock_price(df):
    """Predicție Machine Learning pe 90 de zile folosind Facebook Prophet."""
    try:
        df_p = df.reset_index()[['Date', 'Close']]
        df_p.columns = ['ds', 'y']
        df_p['ds'] = df_p['ds'].dt.tz_localize(None)
        
        m = Prophet(
            daily_seasonality=False, 
            weekly_seasonality=True, 
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        m.fit(df_p)
        
        future = m.make_future_dataframe(periods=90)
        forecast = m.predict(future)
        return forecast
    except:
        return None

def render_ai_chart(forecast, hist):
    """Generare grafic profesional cu interval de încredere."""
    if forecast is None: return
    fig = go.Figure()
    
    hist_recent = hist.iloc[-120:]
    fig.add_trace(go.Scatter(
        x=hist_recent.index, y=hist_recent['Close'], 
        name='Preț Real', line=dict(color='#58A6FF', width=2)
    ))
    
    future_dates = forecast['ds'].iloc[-90:]
    yhat = forecast['yhat'].iloc[-90:]
    
    fig.add_trace(go.Scatter(
        x=future_dates, y=yhat, 
        line=dict(dash='dash', color='#3FB950'), name='Proiecție AI (90z)'
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates, y=forecast['yhat_upper'].iloc[-90:], 
        line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=forecast['yhat_lower'].iloc[-90:], 
        fill='tonexty', fillcolor='rgba(63, 185, 80, 0.1)', 
        line=dict(width=0), name='Interval Probabilitate'
    ))
    
    fig.update_layout(
        height=400, template="plotly_dark", 
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
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
