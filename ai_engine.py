import pandas as pd
import numpy as np
from prophet import Prophet
from transformers import pipeline
import plotly.graph_objects as go
import streamlit as st

# Incarcare model de sentiment specializat pe finante (FinBERT)
@st.cache_resource
def get_sentiment_pipeline():
    # Folosim versiunea optimizată pentru analiză financiară profesională
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_sentiment_ai(news_list):
    """Analizează contextul știrilor folosind NLP Transformer (FinBERT)."""
    if not news_list: return 0.0
    try:
        pipe = get_sentiment_pipeline()
        # Luăm primele 10 titluri pentru o relevanță mai mare
        titles = [n['title'] for n in news_list[:10]]
        results = pipe(titles)
        
        # Calculăm un scor ponderat: Positive = 1, Negative = -1, Neutral = 0
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
    # Pregătire date pentru Prophet (necesită coloanele 'ds' și 'y')
    df_p = df.reset_index()[['Date', 'Close']]
    df_p.columns = ['ds', 'y']
    df_p['ds'] = df_p['ds'].dt.tz_localize(None)
    
    # Model Prophet ajustat pentru volatilitatea pieței
    m = Prophet(
        daily_seasonality=False, 
        weekly_seasonality=True, 
        yearly_seasonality=True,
        changepoint_prior_scale=0.05 # Flexibilitate pentru schimbări de trend
    )
    m.fit(df_p)
    
    future = m.make_future_dataframe(periods=90)
    forecast = m.predict(future)
    return forecast

def render_ai_chart(forecast, hist):
    """Generare grafic profesional cu interval de încredere."""
    fig = go.Figure()
    
    # Date istorice (ultimele 120 zile)
    hist_recent = hist.iloc[-120:]
    fig.add_trace(go.Scatter(
        x=hist_recent.index, y=hist_recent['Close'], 
        name='Preț Real', line=dict(color='#58A6FF', width=2)
    ))
    
    # Predicție (ultimele 90 zile din forecast)
    future_dates = forecast['ds'].iloc[-90:]
    yhat = forecast['yhat'].iloc[-90:]
    
    fig.add_trace(go.Scatter(
        x=future_dates, y=yhat, 
        line=dict(dash='dash', color='#3FB950'), name='Proiecție AI (90z)'
    ))
    
    # Bandă de incertitudine (Aria de risc)
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
