import pandas as pd
import numpy as np
from prophet import Prophet
from transformers import pipeline
import plotly.graph_objects as go
import streamlit as st

# Incarcare model de sentiment specializat pe finante
@st.cache_resource
def get_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_sentiment_ai(news_list):
    """Analizează contextul știrilor folosind NLP Transformer."""
    if not news_list: return 0.0
    pipe = get_sentiment_pipeline()
    titles = [n['title'] for n in news_list[:5]]
    results = pipe(titles)
    
    score = 0
    for r in results:
        if r['label'] == 'positive': score += r['score']
        elif r['label'] == 'negative': score -= r['score']
    return score / len(results)

def predict_stock_price(df):
    """Predicție Machine Learning pe 30 de zile."""
    df_p = df.reset_index()[['Date', 'Close']]
    df_p.columns = ['ds', 'y']
    df_p['ds'] = df_p['ds'].dt.tz_localize(None)
    
    # Model Prophet cu sezonalitate automata
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(df_p)
    
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    return forecast

def render_ai_chart(forecast, hist):
    """Generare grafic profesional cu benzi de incredere."""
    fig = go.Figure()
    # Date istorice recente
    hist_recent = hist.iloc[-90:]
    fig.add_trace(go.Scatter(x=hist_recent.index, y=hist_recent['Close'], name='Istoric Real'))
    # Predictie
    fig.add_trace(go.Scatter(x=forecast['ds'].iloc[-30:], y=forecast['yhat'].iloc[-30:], 
                             line=dict(dash='dash', color='cyan'), name='Proiecție AI'))
    # Banda de incertitudine
    fig.add_trace(go.Scatter(x=forecast['ds'].iloc[-30:], y=forecast['yhat_upper'].iloc[-30:], 
                             line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast['ds'].iloc[-30:], y=forecast['yhat_lower'].iloc[-30:], 
                             fill='tonexty', fillcolor='rgba(0,255,255,0.1)', line=dict(width=0), name='Interval Încredere'))
    
    fig.update_layout(height=400, template="plotly_dark", margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig, use_container_width=True)