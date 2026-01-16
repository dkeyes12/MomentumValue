import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CONFIGURATION ---
st.set_page_config(page_title="Stock Momentum Dashboard", layout="wide")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter Stock Ticker", value="NVDA").upper()
time_period = st.sidebar.selectbox("Time Period", ["1y", "2y", "3y", "5y", "max"], index=1)
rsi_source = st.sidebar.selectbox("RSI Source Data", ["Close", "Open", "High", "Low"], index=0)

# --- FUNCTIONS ---
def calculate_rsi(data_series, window=14):
    """
    Calculates the Relative Strength Index (RSI) for a given data series.
    """
    delta = data_series.diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.ewm(com=window-1, adjust=False).mean()
    avg_loss = loss.ewm(com=window-1, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_stock_data(ticker, period, source_column):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            return None, None
            
        info = stock.info
        
        # Calculate Indicators (SMA calculations use Close price by default)
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Calculate RSI based on user selection
        if source_column in df.columns:
            df['RSI'] = calculate_rsi(df[source_column])
        else:
            st.warning(f"Data source '{source_column}' not found, defaulting RSI to 'Close'.")
            df['RSI'] = calculate_rsi(df['Close'])
            
        return df, info
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None

def determine_signal(price, sma50, sma200, rsi):
    trend_buy = sma50 > sma200
    buy_cond = (price > sma200) and (rsi > 50) and trend_buy
    short_cond = (price < sma50) and (rsi < 50) and (not trend_buy)
    
    if buy_cond:
        return "BUY", "success"
    elif short_cond:
        return "SELL / SHORT", "error"
    else:
        return "WAIT", "warning"

# --- MAIN DASHBOARD LOGIC ---

st.title(f"📊 {ticker_input} Interactive Dashboard")

# 1. Fetch Data
with st.spinner('Fetching market data...'):
    df, info = get_stock_data(ticker_input, time_period, rsi_source)

if df is not None and not df.empty:
    # Get latest values
    current_price = df['Close'].iloc[-1]
    last_rsi = df['RSI'].iloc[-1]
    last_sma50 = df['SMA_50'].iloc[-1]
    last_sma200 = df['SMA_200'].iloc[-1]
    
    # Get P/E (Handle missing data)
    pe_ratio = info.get('trailingPE')
    if pe_ratio is None:
        pe_ratio = info.get('forwardPE') 
    
    # Determine Signal
    signal, signal_color = determine_signal(current_price, last_sma50, last_sma200, last_rsi)

    # 2. Display KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        prev_close = df['Close'].iloc[-2]
        st.metric("Current Price", f"${current_price:.2f}", f"{current_price - prev_close:.2f}")
    with col2:
        st.metric(f"RSI (14) on {rsi_source}", f"{last_rsi:.2f}")
    with col3:
        pe_display = f"{pe_ratio:.2f}" if pe_ratio else "N/A"
        st.metric("P/E Ratio", pe_display)
    with col4:
        # Dynamic color logic for HTML
        color_map = {'BUY': 'green', 'SELL / SHORT': 'red', 'WAIT': 'orange'}
        text_color = color_map.get(signal, 'orange')
        
        st.markdown(f"""
            <div style="text-align: center; padding: 10px; background-color: #f0f2f6; border-radius: 10px;">
                <h3 style="margin:0; color: black;">Signal</h3>
                <h2 style="margin:0; color: {text_color};">{signal}</h2>
            </div>
            """, unsafe_allow_html=True)
        
    # --- ROW 2: 2x2 VALUE VS MOMENTUM PLOT ---
    st.subheader("Value vs. Momentum Matrix (P/E vs RSI)")
    
    if pe_ratio is not None and pe_ratio > 0:
        PE_THRESHOLD = 25  
        RSI_THRESHOLD = 50 
        
        fig_quad = go.Figure()

        # Add the stock point
        fig_quad.add_trace(go.Scatter(
            x=[pe_ratio], y=[last_rsi],
            mode='markers+text',
            text=[ticker_input],
            textposition="top right",
            marker=dict(size=20, color='blue', line=dict(width=2, color='white')),
            name='Current Status'
        ))

        # Add Quadrant Backgrounds
        # Q1: Top Left (Value + Momentum) - Green
        fig_quad.add_shape(type="rect", x0=0, y0=RSI_THRESHOLD, x1=PE_THRESHOLD, y1=100,
                           fillcolor="green", opacity=0.1, layer="below", line_width=0)
        # Q2: Top Right (Growth/Expensive) - Yellow
        fig_quad.add_shape(type="rect", x0=PE_THRESHOLD, y0=RSI_THRESHOLD, x1=max(pe_ratio*2, 100), y1=100,
                           fillcolor="yellow", opacity=0.1, layer="below", line_width=0)
        # Q3: Bottom Left (Value Trap/Weak) - Yellow
        fig_quad.add_shape(type="rect", x0=0, y0=0, x1=PE_THRESHOLD, y1=RSI_THRESHOLD,
                           fillcolor="yellow", opacity=0.1, layer="below", line_width=0)
        # Q4: Bottom Right (Expensive & Weak) - Red
        fig_quad.add_shape(type="rect", x0=PE_THRESHOLD, y0=0, x1=max(pe_ratio*2, 100), y1=RSI_THRESHOLD,
                           fillcolor="red", opacity=0.1, layer="below", line_width=0)

        # Add Crosshair Lines
        fig_quad.add_vline(x=PE_THRESHOLD, line_width=1, line_dash="dash", line_color="gray")
        fig_quad.add_hline(y=RSI_THRESHOLD, line_width=1, line_dash="dash", line_color="gray")

        # Quadrant Labels
        fig_quad.add_annotation(x=PE_THRESHOLD/2, y=85, text="VALUE + MOMENTUM", showarrow=False, font=dict(color="green", size=14, weight="bold"))
        fig_quad.add_annotation(x=PE_THRESHOLD*1.5, y=85, text="EXPENSIVE MOMENTUM", showarrow=False, font=dict(color="orange", size=12))
        fig_quad.add_annotation(x=PE_THRESHOLD/2, y=25, text="WEAK / VALUE TRAP", showarrow=False, font=dict(color="orange", size=12))
        fig_quad.add_annotation(x=PE_THRESHOLD*1.5, y=25, text="EXPENSIVE & WEAK", showarrow=False, font=dict(color="red", size=14, weight="bold"))

        max_x = max(50, pe_ratio * 1.2)
        fig_quad.update_xaxes(title_text="P/E Ratio (Value)", range=[0, max_x])
        fig_quad.update_yaxes(title_text=f"RSI ({rsi_source} data) (Momentum)", range=[0, 100])
        fig_quad.update_layout(height=500, title="Quadrant Analysis: Is it Cheap? Is it Moving?")
        
        # --- FIX 1: Corrected use_container_width ---
        st.plotly_chart(fig_quad, use_container_width=True)
    else:
        st.warning("Cannot generate Value vs Momentum plot: P/E Ratio data is missing or negative.")
   
    # --- ROW 3: TECHNICAL ANALYSIS CHART ---
    st.subheader("Technical Analysis (Price & Trends)")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Price Chart
    fig.add_trace(go.Candlestick(x=df.index,
                        open=df['Open'], high=df['High'],
                        low=df['Low'], close=df['Close'], name='OHLC'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], 
                             line=dict(color='orange', width=2), name='50 Day MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], 
                             line=dict(color='blue', width=2), name='200 Day MA'), row=1, col=1)

    # RSI Chart
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], 
                             line=dict(color='purple', width=2), name=f'RSI ({rsi_source})'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1, line_color="red")
    fig.add_hline(y=30, line_dash="dot", row=2, col=1, line_color="green")
    fig.add_hline(y=50, line_dash="solid", row=2, col=1, line_color="gray", opacity=0.5)

    fig.update_layout(xaxis_rangeslider_visible=False, height=600, margin=dict(l=20, r=20, t=30, b=20))
    
    # --- FIX 2: Corrected use_container_width ---
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("No data found. Please check the ticker symbol.")