import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_technical_indicators(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    
    if hist.empty:
        raise ValueError(f"No data found for ticker: {ticker}")
    
    # Price Momentum Indicators
    hist['50_MA'] = hist['Close'].rolling(window=50).mean()
    hist['200_MA'] = hist['Close'].rolling(window=200).mean()
    hist['Price_Above_50MA'] = (hist['Close'] > hist['50_MA']).astype(int)
    hist['Price_Above_200MA'] = (hist['Close'] > hist['200_MA']).astype(int)
    
    # RSI (Relative Strength Index)
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    hist['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    hist['12_EMA'] = hist['Close'].ewm(span=12, adjust=False).mean()
    hist['26_EMA'] = hist['Close'].ewm(span=26, adjust=False).mean()
    hist['MACD'] = hist['12_EMA'] - hist['26_EMA']
    hist['Signal_Line'] = hist['MACD'].ewm(span=9, adjust=False).mean()
    hist['MACD_Hist'] = hist['MACD'] - hist['Signal_Line']
    
    # Volume Analysis
    hist['Volume_MA'] = hist['Volume'].rolling(window=20).mean()
    hist['Volume_Spike'] = (hist['Volume'] > 2 * hist['Volume_MA']).astype(int)
    
    # Latest values
    current_price = hist['Close'].iloc[-1]
    ma_50 = hist['50_MA'].iloc[-1]
    ma_200 = hist['200_MA'].iloc[-1]
    rsi = hist['RSI'].iloc[-1]
    macd_hist = hist['MACD_Hist'].iloc[-1]
    volume_spike = hist['Volume_Spike'].iloc[-1]
    
    return {
        'current_price': current_price,
        'ma_50': ma_50,
        'ma_200': ma_200,
        'rsi': rsi,
        'macd_hist': macd_hist,
        'volume_spike': volume_spike,
        'price_above_50ma': hist['Price_Above_50MA'].iloc[-1],
        'price_above_200ma': hist['Price_Above_200MA'].iloc[-1]
    }

def normalize(value, min_val, max_val, inverse=False):
    score = (value - min_val) / (max_val - min_val)
    score = max(0, min(score, 1))
    return (1 - score if inverse else score)

def calculate_technical_score(indicators):
    # Weighted scoring system
    scores = {
        'price_vs_50ma': normalize(indicators['current_price'] - indicators['ma_50'], -0.1*indicators['ma_50'], 0.1*indicators['ma_50']),
        'price_vs_200ma': normalize(indicators['current_price'] - indicators['ma_200'], -0.15*indicators['ma_200'], 0.15*indicators['ma_200']),
        'rsi_score': normalize(indicators['rsi'], 30, 70, inverse=True),  # RSI between 30-70 is ideal
        'macd_score': normalize(indicators['macd_hist'], -2, 2),
        'volume_score': indicators['volume_spike'] * 0.5,  # Binary score for volume spike
        'trend_score': (indicators['price_above_50ma'] + indicators['price_above_200ma']) * 0.25
    }
    
    # Weighted average
    weights = {
        'price_vs_50ma': 0.25,
        'price_vs_200ma': 0.25,
        'rsi_score': 0.2,
        'macd_score': 0.15,
        'volume_score': 0.1,
        'trend_score': 0.05
    }
    
    total_score = sum(scores[k] * weights[k] for k in scores)
    return round(total_score * 100)  # Convert to percentage