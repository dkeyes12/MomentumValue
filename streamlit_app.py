import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CONFIGURATION ---
st.set_page_config(page_title="Portfolio Optimizer [Maximize MomentumValue]", layout="wide")

# --- DATASETS ---
ETF_TICKERS = [
    {"Ticker": "XLK", "Sector": "Technology"},
    {"Ticker": "XLV", "Sector": "Health Care"},
    {"Ticker": "XLF", "Sector": "Financials"},
    {"Ticker": "XLRE", "Sector": "Real Estate"},
    {"Ticker": "XLE", "Sector": "Energy"},
    {"Ticker": "XLB", "Sector": "Materials"},
    {"Ticker": "XLY", "Sector": "Cons. Discretionary"},
    {"Ticker": "XLP", "Sector": "Cons. Staples"},
    {"Ticker": "XLI", "Sector": "Industrials"},
    {"Ticker": "XLU", "Sector": "Utilities"},
    {"Ticker": "XLC", "Sector": "Communication"}
]

STOCK_TICKERS = [
    {"Ticker": "NVDA", "Sector": "Technology"},
    {"Ticker": "AAPL", "Sector": "Technology"},
    {"Ticker": "MSFT", "Sector": "Technology"},
    {"Ticker": "AMZN", "Sector": "Cons. Disc."},
    {"Ticker": "GOOGL", "Sector": "Comm. Svcs"},
    {"Ticker": "META", "Sector": "Comm. Svcs"},
    {"Ticker": "TSLA", "Sector": "Cons. Disc."},
    {"Ticker": "JPM", "Sector": "Financials"},
    {"Ticker": "V", "Sector": "Financials"},
    {"Ticker": "JNJ", "Sector": "Health Care"},
    {"Ticker": "LLY", "Sector": "Health Care"},
    {"Ticker": "XOM", "Sector": "Energy"}
]

# --- CACHE MANAGEMENT ---
def clear_cache_callback():
    """Forces a complete reset of cache and state when switching modes"""
    st.cache_data.clear()
    if "market_data" in st.session_state:
        del st.session_state["market_data"]
    if "opt_results" in st.session_state:
        del st.session_state["opt_results"]
    if "historical_data" in st.session_state:
        del st.session_state["historical_data"]

# --- INITIALIZE STATE ---
if "market_data" not in st.session_state: st.session_state["market_data"] = None
if "opt_results" not in st.session_state: st.session_state["opt_results"] = None
if "historical_data" not in st.session_state: st.session_state["historical_data"] = {}

# --- HELPER FUNCTIONS ---
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@st.cache_data
def process_bulk_data(tickers, sector_map, mode, period="2y"):
    """
    Fetches data. 
    mode="ETF" -> Fetches P/E
    mode="Stock" -> Fetches PEG
    """
    ticker_list = [t.upper().strip() for t in tickers if t.strip()]
    if not ticker_list: return None, None

    # Bulk download
    bulk_data = yf.download(ticker_list, period=period, group_by='ticker', auto_adjust=False)
    
    snapshot_data = []
    hist_data = {}

    for t in ticker_list:
        try:
            # Handle yfinance multi-index structure
            if len(ticker_list) == 1:
                df = bulk_data.copy()
            else:
                df = bulk_data[t].copy()
            
            if df.empty: continue

            df = df.dropna(how='all')
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            df['RSI'] = calculate_rsi(df['Close'])
            
            hist_data[t] = df

            # Metrics
            current_price = df['Close'].iloc[-1]
            start_price = df['Close'].iloc[0]
            pct_return = (current_price - start_price) / start_price if start_price > 0 else 0
            
            rsi = df['RSI'].iloc[-1]
            volatility = df['Close'].pct_change().std() * np.sqrt(252)
            
            # --- CONDITIONAL VALUATION METRIC ---
            val_metric = None
            try:
                stock_info = yf.Ticker(t).info
                if mode == "Stock (PEG)":
                    val_metric = stock_info.get('pegRatio')
                else: # ETF (PE)
                    val_metric = stock_info.get('trailingPE')
                    if val_metric is None: val_metric = stock_info.get('forwardPE')
            except:
                val_metric = None
            
            # Store if valid
            if val_metric is not None and val_metric > 0:
                row = {
                    "Ticker": t,
                    "Sector": sector_map.get(t, "Unknown"),
                    "Price": current_price,
                    "RSI": rsi,
                    "Volatility": volatility,
                    "Return": pct_return
                }
                # Assign to correct column based on mode
                if mode == "Stock (PEG)":
                    row["PEG"] = val_metric
                    row["PE"] = np.nan # Placeholder
                else:
                    row["PE"] = val_metric
                    row["PEG"] = np.nan # Placeholder
                    
                snapshot_data.append(row)

        except Exception:
            continue
            
    return pd.DataFrame(snapshot_data), hist_data

# --- OPTIMIZATION ENGINE ---
def optimize_portfolio(df, objective_type, max_weight_per_asset, mode):
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver: return None

    weights = []
    for i in range(len(df)):
        weights.append(solver.NumVar(0.0, max_weight_per_asset, f'w_{i}'))

    # Constraint: Sum of weights = 1.0
    constraint_sum = solver.Constraint(1.0, 1.0)
    for w in weights:
        constraint_sum.SetCoefficient(w, 1)

    objective = solver.Objective()
    
    # --- ADAPTIVE SCORING LOGIC ---
    # Switches formula based on the selected mode
    if mode == "Stock (PEG)":
        # PEG Score: 1/PEG + RSI/100 (No multiplier needed, scales are similar)
        scores = (df['RSI'] / 100) + (1 / df['PEG']) 
    else:
        # PE Score: 1/PE * 50 + RSI/100 (Multiplier needed to match scales)
        scores = (df['RSI'] / 100) + ((1 / df['PE']) * 50)
    
    if objective_type == "Maximize Gain (Score)":
        for i, w in enumerate(weights):
            objective.SetCoefficient(w, scores.iloc[i])
        objective.SetMaximization()
    elif objective_type == "Minimize Loss (Volatility)":
        avg_score = scores.mean()
        constraint_quality = solver.Constraint(avg_score, solver.infinity())
        for i, w in enumerate(weights):
            constraint_quality.SetCoefficient(w, scores.iloc[i])
        for i, w in enumerate(weights):
            objective.SetCoefficient(w, df['Volatility'].iloc[i])
        objective.SetMinimization()

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        results = []
        for i, w in enumerate(weights):
            if w.solution_value() > 0.001: 
                row = {
                    "Ticker": df['Ticker'].iloc[i],
                    "Sector": df['Sector'].iloc[i],
                    "Weight": w.solution_value(),
                    "RSI": df['RSI'].iloc[i],
                    "Volatility": df['Volatility'].iloc[i]
                }
                # Add the correct valuation metric to the results
                if mode == "Stock (PEG)":
                    row["PEG"] = df['PEG'].iloc[i]
                else:
                    row["PE"] = df['PE'].iloc[i]
                    
                results.append(row)
        return pd.DataFrame(results)
    else:
        return pd.DataFrame()

# --- DASHBOARD UI ---
st.title("⚖️ Portfolio Optimizer: Amongst S&P500 ETFs or stocks")

# 1. SIDEBAR SETTINGS
with st.sidebar:
    st.header("1. Choose Stocks to Optimize")
    
    # TRIGGER CACHE CLEAR ON CHANGE
    mode_select = st.radio(
        "Pre-Selected Stock Sectors Or Stocks:", 
        ["S&P 500 Sectors (P/E)", "Popular and widely followed stocks (P/E/G)"], 
        on_change=clear_cache_callback
    )
    
    st.divider()
    st.header("2. Optimization Settings")
    obj_choice = st.radio("Goal", ["Maximize Gain (MomentumValue)", "Minimize Loss (Volatility)"])
    max_concentration = st.slider("Max Allocation per Asset", 0.05, 1.0, 0.25, 0.05)
    
    # Explicit Cache Button
    st.divider()
    if st.button("⚠️ Force Clear Cache"):
        clear_cache_callback()
        st.rerun()

# 2. EDITABLE INPUT
st.subheader(f"1. Define Universe: {mode_select}")

# Load default based on mode
current_defaults = STOCK_TICKERS if mode_select == "Stock (PEG)" else ETF_TICKERS

# Session State for Editable DF
if "user_tickers" not in st.session_state or st.session_state.get("last_mode") != mode_select:
    st.session_state["user_tickers"] = pd.DataFrame(current_defaults)
    st.session_state["last_mode"] = mode_select

col_input, col_action = st.columns([3, 1])
with col_input:
    edited_df = st.data_editor(
        st.session_state["user_tickers"], 
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "Sector": st.column_config.TextColumn("Sector", width="medium"),
        },
        num_rows="dynamic", use_container_width=True, key="ticker_editor"
    )

with col_action:
    st.write("### ") 
    if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
        tickers = edited_df["Ticker"].tolist()
        sector_map = dict(zip(edited_df['Ticker'], edited_df['Sector']))
        
        if len(tickers) < 2:
            st.error("Need 2+ tickers.")
        else:
            with st.spinner("Fetching data..."):
                df_mkt, hist_data = process_bulk_data(tickers, sector_map, mode_select)
                
                if df_mkt is not None and not df_mkt.empty:
                    df_res = optimize_portfolio(df_mkt, obj_choice, max_concentration, mode_select)
                    st.session_state["market_data"] = df_mkt
                    st.session_state["historical_data"] = hist_data
                    st.session_state["opt_results"] = df_res
                else:
                    st.error("Data fetch failed.")

# CSS centering
st.markdown("""<style>[data-testid="stDataFrame"] th, [data-testid="stDataFrame"] td { text-align: center !important; }</style>""", unsafe_allow_html=True)

# 3. EXECUTION
if st.session_state["market_data"] is not None:
    df_market = st.session_state["market_data"]
    df_opt = st.session_state["opt_results"]
    hist_map = st.session_state["historical_data"]
    
    metric_col = "PEG" if mode_select == "Stock (PEG)" else "PE"

    if df_opt is not None and not df_opt.empty:
        # --- PLOT ---
        st.subheader("2. Portfolio Analysis")
        
        # Scaling Logic
        if mode_select == "Stock (PEG)":
            VAL_THRESHOLD = 1.5
            MAX_X = 4.0
            x_label = "PEG Ratio (Growth Value)"
        else:
            VAL_THRESHOLD = 25
            MAX_X = 50
            x_label = "P/E Ratio (Value)"
            
        fig_quad = go.Figure()
        
        # Universe
        remaining = df_market[~df_market['Ticker'].isin(df_opt['Ticker'])]
        fig_quad.add_trace(go.Scatter(x=remaining[metric_col].clip(upper=MAX_X), y=remaining['RSI'],
            mode='markers+text', text=remaining['Ticker'], textposition="top center",
            textfont=dict(size=11, color="black"), marker=dict(size=12, color='rgba(128,128,128,0.5)', line=dict(width=1, color='dimgray')), name='Universe'))

        # Portfolio
        fig_quad.add_trace(go.Scatter(x=df_opt[metric_col].clip(upper=MAX_X), y=df_opt['RSI'],
            mode='markers+text', text=df_opt['Ticker'], textposition="top center",
            textfont=dict(family="Arial Black", size=12, color="black"), marker=dict(size=18, color='blue', line=dict(width=2, color='black')), name='Selected'))

        # Backgrounds
        fig_quad.add_shape(type="rect", x0=0, y0=50, x1=VAL_THRESHOLD, y1=100, fillcolor="green", opacity=0.1, layer="below", line_width=0)
        fig_quad.add_shape(type="rect", x0=VAL_THRESHOLD, y0=50, x1=MAX_X, y1=100, fillcolor="yellow", opacity=0.1, layer="below", line_width=0)
        fig_quad.add_shape(type="rect", x0=0, y0=0, x1=VAL_THRESHOLD, y1=50, fillcolor="yellow", opacity=0.1, layer="below", line_width=0)
        fig_quad.add_shape(type="rect", x0=VAL_THRESHOLD, y0=0, x1=MAX_X, y1=50, fillcolor="red", opacity=0.1, layer="below", line_width=0)

        fig_quad.update_layout(title=f"{mode_select} Analysis", xaxis_title=x_label, yaxis_title="RSI (Momentum)", height=600)
        st.plotly_chart(fig_quad, use_container_width=True)

        # --- ALLOCATION TABLE ---
        st.divider()
        st.subheader("3. Optimal Allocation")
        
        with st.expander("📊 Methodology"):
            st.write(f"Uses Linear Programming. Aggregates calculated via Weighted Arithmetic Mean (RSI) and Weighted Harmonic Mean ({metric_col}).")
        
        disp_df = df_opt[["Ticker", "Sector", "Weight", metric_col, "RSI"]].copy()
        
        # Totals
        w_rsi = (disp_df['Weight'] * disp_df['RSI']).sum()
        w_val_inv = (disp_df['Weight'] / disp_df[metric_col]).sum()
        w_val = 1 / w_val_inv if w_val_inv > 0 else 0
        
        summary = pd.DataFrame([{"Ticker": "TOTAL", "Sector": "-", "Weight": disp_df['Weight'].sum(), metric_col: w_val, "RSI": w_rsi}])
        final = pd.concat([disp_df, summary], ignore_index=True)
        
        final["Weight"] = final["Weight"].apply(lambda x: f"{x:.1%}")
        final[metric_col] = final[metric_col].apply(lambda x: f"{x:.2f}")
        final["RSI"] = final["RSI"].apply(lambda x: f"{x:.1f}")
        
        st.dataframe(final, use_container_width=True, height=(len(final)+1)*35)
        
        # --- CHART ---
        st.divider()
        st.subheader("4. Technical Analysis")
        sel_t = st.selectbox("View Asset:", list(hist_map.keys()))
        df_c = hist_map.get(sel_t)
        
        if df_c is not None:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df_c.index, open=df_c['Open'], high=df_c['High'], low=df_c['Low'], close=df_c['Close'], name='OHLC'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c.index, y=df_c['SMA_50'], line=dict(color='orange'), name='SMA 50'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c.index, y=df_c['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", row=2, col=1, line_color="red")
            fig.add_hline(y=30, line_dash="dot", row=2, col=1, line_color="green")
            fig.update_layout(height=500, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

# --- MARKET ANALYSIS ---
    st.divider()
    st.subheader("2. Market Data Analysis")
    st.dataframe(
        df_market[["Ticker", "Sector", metric_col, "RSI", "Return", "Volatility"]].style.format({
            metric_col: "{:.2f}", "RSI": "{:.2f}", "Return": "{:.2%}", "Volatility": "{:.2%}"
        }),
        use_container_width=True, height=(len(df_market)+1)*35
    )

# --- LOGIC SUMMARY ---
st.divider()
with st.expander("ℹ️ How the Optimization Logic Works"):
    st.markdown(r"""
    ### 1. The Scoring Formula
    The optimizer assigns a **"Growth-Momentum Score"** to every asset:
    * **Value (PEG):** Inverse of PEG. Lower PEG = Higher Score.
    * **Momentum (RSI):** Normalized RSI.
    
    $$
    \text{Score} = \underbrace{\left( \frac{\text{RSI}}{100} \right)}_{\text{Momentum}} + \underbrace{\left( \frac{1}{\text{PEG Ratio}} \right)}_{\text{Value}}
    $$

    ### 2. This app uses an open source Linear Solver (Google - OR Tools).  
    * ** We maximize factor exposure ($RSI + Value$) using Linear Programming ($O(n)$) rather than minimizing variance via Quadratic Programming ($O(n^2)$). Risk is managed via concentration constraints rather than historical covariance, avoiding estimation errors common in small samples.
    * **Quadratic Programming (MPT):** Modern Portfolio Theory typically employs Quadratic Programming to minimize portfolio variance ($\sigma^2$). This requires calculating the full covariance matrix $\Sigma$ to account for pairwise asset correlations ($O(n^2)$ complexity). It optimizes for the lowest risk at a given return level.
    * **Linear Programming (Factor Exposure):** This tool utilizes Linear Programming (GLOP solver) to maximize direct factor exposure. Instead of minimizing variance through correlation, we mitigate risk via **concentration constraints** (hard limits on max allocation). This allows for a computationally efficient ($O(n)$) maximization of the 'Growth + Momentum' alpha score without the instability often introduced by covariance estimation errors in small samples.
    """)