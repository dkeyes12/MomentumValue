import streamlit as st
import os

# --- CRITICAL FIX: DISABLE OPENBB AUTO-BUILD ---
# This prevents OpenBB from trying to write to the read-only file system
os.environ["OPENBB_AUTO_BUILD"] = "false"

# Now it is safe to import openbb
from openbb import obb
# -----------------------------------------------

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --- CONFIGURATION ---
st.set_page_config(page_title="Stock Momentum Dashboard (OpenBB v4)", layout="wide")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter Stock Ticker", value="NVDA").upper()
time_period = st.sidebar.selectbox("Time Period", ["1y", "2y", "3y", "5y"], index=0)
# User selection (Capitalized for UI, mapped to lowercase later)
rsi_source = st.sidebar.selectbox("RSI Source Data", ["Close", "Open", "High", "Low"], index=0)

# --- FUNCTIONS ---

def get_start_date(period):
    today = datetime.now()
    days_map = {"1y": 365, "2y": 730, "3y": 1095, "5y": 1825}
    delta = days_map.get(period, 365)
    return (today - timedelta(days=delta)).strftime('%Y-%m-%d')

def get_stock_data(ticker, period, source_column):
    try:
        start_date = get_start_date(period)

        # 1. Load Historical Price Data (v4 syntax)
        # OpenBB v4 returns columns in LOWERCASE: open, high, low, close, volume
        df = obb.equity.price.historical(
            symbol=ticker, 
            start_date=start_date, 
            provider="yfinance"
        ).to_df()

        if df is None or df.empty:
            return None, None

        # Ensure index is datetime
        df.index = pd.to_datetime(df.index)

        # 2. Get Fundamental Snapshot
        pe_ratio = None
        try:
            metrics = obb.equity.fundamental.metrics(
                symbol=ticker, 
                provider="yfinance"
            ).to_df()
            # Check for common P/E column names
            if 'pe_ratio' in metrics.columns:
                pe_ratio = metrics['pe_ratio'].iloc[0]
        except Exception:
            pass # Fail silently on fundamentals if missing

        # 3. Technical Analysis Calculations
        # Convert user selection (e.g., "Close") to lowercase ("close")
        target_col = source_column.lower()
        
        # Simple Moving Averages
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['SMA_200'] = df['close'].rolling(window=200).mean()

        # RSI Calculation (Manual Pandas implementation)
        delta = df[target_col].diff()
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))
        
        avg_gain = gain.ewm(com=13, min_periods=14).mean()
        avg_loss = loss.ewm(com=13, min_periods=14).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        return df, pe_ratio

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None

def determine_signal(price, sma50, sma200, rsi):
    if pd.isna(sma50) or pd.isna(sma200) or pd.isna(rsi):
        return "INSUFFICIENT DATA", "gray"

    trend_buy = sma50 > sma200
    buy_cond = (price > sma200) and (rsi > 50) and trend_buy
    short_cond = (price < sma50) and (rsi < 50) and (not trend_buy)
    
    if buy_cond:
        return "BUY", "green"
    elif short_cond:
        return "SELL / SHORT", "red"
    else:
        return "WAIT", "orange"

# --- MAIN DASHBOARD LOGIC ---

st.title(f"📊 {ticker_input} Dashboard (OpenBB v4)")

with st.spinner('Fetching market data via OpenBB v4...'):
    df, pe_ratio = get_stock_data(ticker_input, time_period, rsi_source)

if df is not None and not df.empty:
    current_price = df['close'].iloc[-1]
    
    # Calculate Signal
    if len(df) > 200:
        last_rsi = df['RSI'].iloc[-1]
        last_sma50 = df['SMA_50'].iloc[-1]
        last_sma200 = df['SMA_200'].iloc[-1]
        signal, signal_color = determine_signal(current_price, last_sma50, last_sma200, last_rsi)
    else:
        last_rsi = 50
        signal, signal_color = "LOADING...", "gray"
        st.warning("Not enough history for 200-day SMA.")

    # --- KPI METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        prev_close = df['close'].iloc[-2]
        change = current_price - prev_close
        st.metric("Current Price", f"${current_price:.2f}", f"{change:.2f}")
    with col2:
        st.metric(f"RSI (14)", f"{last_rsi:.2f}")
    with col3:
        pe_display = f"{pe_ratio:.2f}" if pe_ratio else "N/A"
        st.metric("P/E Ratio", pe_display)
    with col4:
        st.markdown(
            f"<div style='background-color:#f0f2f6; padding:10px; border-radius:5px; text-align:center;'>"
            f"<h3 style='margin:0; color:{signal_color};'>{signal}</h3></div>", 
            unsafe_allow_html=True
        )

    # --- TECHNICAL PRICE AND TRENDS CHART ---
    st.subheader("Technical Analysis")
    
    # Create Subplots: Row 1 = Price (70%), Row 2 = RSI (30%)
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        row_heights=[0.7, 0.3]
    )

    # 1. Candlestick Chart (Row 1)
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], 
        name='OHLC'
    ), row=1, col=1)
    
    # 2. SMAs (Row 1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA_50'], 
        line=dict(color='orange', width=1.5), 
        name='50 Day MA'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA_200'], 
        line=dict(color='blue', width=1.5), 
        name='200 Day MA'
    ), row=1, col=1)

    # 3. RSI Chart (Row 2)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['RSI'], 
        line=dict(color='purple', width=2), 
        name='RSI'
    ), row=2, col=1)
    
    # RSI Thresholds (70/30)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1, line_color="red", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dot", row=2, col=1, line_color="green", annotation_text="Oversold")
    
    # Fill RSI Background for "Zones"
    fig.add_hrect(y0=70, y1=100, row=2, col=1, fillcolor="red", opacity=0.1, layer="below", line_width=0)
    fig.add_hrect(y0=0, y1=30, row=2, col=1, fillcolor="green", opacity=0.1, layer="below", line_width=0)

    # Chart Layout Fixes
    fig.update_layout(
        height=600, 
        xaxis_rangeslider_visible=False, # Important: Hides the bottom slider which often breaks layouts
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Remove gaps in non-trading days (optional, often cleaner for stocks)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])]) 

    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("No data found. Please check ticker spelling or internet connection.")