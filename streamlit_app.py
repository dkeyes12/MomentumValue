import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --- OPENBB v4 IMPORT ---
from openbb import obb

# --- CONFIGURATION ---
st.set_page_config(page_title="Stock Momentum Dashboard (OpenBB v4)", layout="wide")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter Stock Ticker", value="NVDA").upper()
time_period = st.sidebar.selectbox("Time Period", ["1y", "2y", "3y", "5y"], index=0)
rsi_source = st.sidebar.selectbox("RSI Source Data", ["close", "open", "high", "low"], index=0)

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
        # using yfinance provider by default
        df = obb.equity.price.historical(
            symbol=ticker, 
            start_date=start_date, 
            provider="yfinance"
        ).to_df()

        if df.empty:
            return None, None

        # 2. Get Fundamental Snapshot (v4 syntax)
        # v4 uses 'metrics' for key ratios
        try:
            metrics = obb.equity.fundamental.metrics(
                symbol=ticker, 
                provider="yfinance"
            ).to_df()
            # Try to find PE ratio (keys vary by provider, usually 'pe_ratio' or similar)
            # yfinance provider often returns 'pe_ratio'
            pe_ratio = metrics['pe_ratio'].iloc[0] if 'pe_ratio' in metrics.columns else None
        except:
            pe_ratio = None

        # 3. Technical Analysis (RSI)
        # OpenBB v4 currently relies on extensions or standard pandas for TA.
        # It is safer/faster to calculate RSI manually here than rely on the v4 extensions 
        # which might not be installed by default.
        
        # Simple Moving Averages
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['SMA_200'] = df['close'].rolling(window=200).mean()

        # RSI Calculation (Standard Formula)
        # source_column is lowercase in v4 df (e.g. 'close')
        series = df[source_column.lower()]
        delta = series.diff()
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
    
    if len(df) > 200:
        last_rsi = df['RSI'].iloc[-1]
        last_sma50 = df['SMA_50'].iloc[-1]
        last_sma200 = df['SMA_200'].iloc[-1]
        signal, signal_color = determine_signal(current_price, last_sma50, last_sma200, last_rsi)
    else:
        last_rsi = 50
        signal, signal_color = "LOADING...", "gray"
        st.warning("Not enough history for 200-day SMA.")

    # KPI Metrics
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
        st.markdown(f"<h2 style='color:{signal_color}'>{signal}</h2>", unsafe_allow_html=True)

    # Charts
    st.subheader("Technical Analysis")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])

    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index,
                        open=df['open'], high=df['high'],
                        low=df['low'], close=df['close'], name='OHLC'), row=1, col=1)
    
    # SMAs
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange'), name='50 Day MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='blue'), name='200 Day MA'), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1, line_color="red")
    fig.add_hline(y=30, line_dash="dot", row=2, col=1, line_color="green")
    
    fig.update_layout(xaxis_rangeslider_visible=False, height=600)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("No data found or API error.")