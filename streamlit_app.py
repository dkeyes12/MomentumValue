import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

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
    keys_to_drop = ["market_data", "opt_results", "historical_data"]
    for k in keys_to_drop:
        if k in st.session_state:
            del st.session_state[k]

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
    mode="Stock" -> Fetches PEG (Falls back to P/E if missing)
    """
    import time # Import time to handle rate limiting
    
    ticker_list = [t.upper().strip() for t in tickers if t.strip()]
    if not ticker_list: return None, None

    # Bulk download history (Fast)
    bulk_data = yf.download(ticker_list, period=period, group_by='ticker', auto_adjust=False)
    
    snapshot_data = []
    hist_data = {}

    # Create a placeholder for progress
    progress_text = "Fetching valuation metrics..."
    my_bar = st.progress(0, text=progress_text)
    
    total_tickers = len(ticker_list)

    for idx, t in enumerate(ticker_list):
        # Update progress bar
        my_bar.progress((idx + 1) / total_tickers, text=f"Fetching data for {t}...")
        
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
            
            # --- ROBUST VALUATION FETCHING ---
            val_metric = None
            try:
                stock = yf.Ticker(t)
                stock_info = stock.info
                
                if mode == "Popular and widely followed stocks (P/E/G)":
                    # Try PEG first
                    val_metric = stock_info.get('pegRatio')
                    
                    # FALLBACK: If PEG is missing/None, use Trailing P/E
                    if val_metric is None:
                        val_metric = stock_info.get('trailingPE')
                        # If Trailing is missing, use Forward P/E
                        if val_metric is None:
                            val_metric = stock_info.get('forwardPE')
                
                else: # ETF (P/E)
                    val_metric = stock_info.get('trailingPE')
                    if val_metric is None: val_metric = stock_info.get('forwardPE')
                
                # Small sleep to prevent Yahoo from blocking us (Rate Limiting)
                time.sleep(0.1)
                
            except Exception as e:
                val_metric = None
            
            # Final Validity Check
            if val_metric is not None and val_metric > 0:
                row = {
                    "Ticker": t,
                    "Sector": sector_map.get(t, "Unknown"),
                    "Price": current_price,
                    "RSI": rsi,
                    "Volatility": volatility,
                    "Return": pct_return
                }
                
                if mode == "Popular and widely followed stocks (P/E/G)":
                    row["PEG"] = val_metric
                    row["PE"] = np.nan 
                else:
                    row["PE"] = val_metric
                    row["PEG"] = np.nan 
                    
                snapshot_data.append(row)
            else:
                print(f"Skipping {t}: No valid valuation metric found.")

        except Exception as e:
            continue
    
    # Clear progress bar
    my_bar.empty()
            
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
    if mode == "Popular and widely followed stocks (P/E/G)":
        # PEG Score: 1/PEG + RSI/100 
        scores = (df['RSI'] / 100) + (1 / df['PEG']) 
    else:
        # PE Score: 1/PE * 50 + RSI/100 
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
                if mode == "Popular and widely followed stocks (P/E/G)":
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
    st.header("2. Time Horizon")
    # NEW: Time Period Selector
    period_select = st.selectbox(
        "Historical Data Lookback:",
        options=["1y", "2y", "3y", "5y"],
        index=3, # Default to 5y
        help="Determines the window for RSI, Volatility, and Return calculations."
    )

    st.divider()
    st.header("3. Optimization Settings")
    obj_choice = st.radio("Goal", ["Maximize Gain (MomentumValue)", "Minimize Loss (Volatility)"])
    max_concentration = st.slider("Max Allocation per Asset", 0.05, 1.0, 0.25, 0.05)
    
    # Explicit Cache Button
    st.divider()
    if st.button("⚠️ Force Clear Cache"):
        clear_cache_callback()
        st.rerun()

# 2. EDITABLE STOCK INPUT
st.subheader(f"1. Define Universe: {mode_select}")

# --- FIX: LOGIC TO SWAP DEFAULTS ---
if "last_mode" not in st.session_state or st.session_state["last_mode"] != mode_select:
    new_defaults = STOCK_TICKERS if mode_select == "Popular and widely followed stocks (P/E/G)" else ETF_TICKERS
    st.session_state["user_tickers"] = pd.DataFrame(new_defaults)
    st.session_state["last_mode"] = mode_select

col_input, col_action = st.columns([3, 1])

with col_input:
    # --- FIX: DYNAMIC WIDGET KEY ---
    edited_df = st.data_editor(
        st.session_state["user_tickers"], 
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="small", help="Symbol"),
            "Sector": st.column_config.TextColumn("Sector", width="medium"),
        },
        num_rows="dynamic", 
        use_container_width=True,
        key=f"editor_{mode_select}"  # <--- THIS IS THE KEY FIX
    )

with col_action:
    st.write("### ") 
    if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
        
        tickers = edited_df["Ticker"].tolist()
        sector_mapping = dict(zip(edited_df['Ticker'], edited_df['Sector']))

        if len(tickers) < 2:
            st.error("Please enter at least 2 tickers.")
        else:
            with st.spinner(f"Fetching {period_select} of data..."):
                df_mkt, hist_data = process_bulk_data(tickers, sector_mapping, mode_select, period=period_select)
                
                if df_mkt is not None and not df_mkt.empty:
                    df_res = optimize_portfolio(df_mkt, obj_choice, max_concentration, mode_select)
                    st.session_state["market_data"] = df_mkt
                    st.session_state["historical_data"] = hist_data
                    st.session_state["opt_results"] = df_res
                else:
                    st.error("Could not fetch valid data. Try individual stocks instead of ETFs.")

# CSS centering
st.markdown("""<style>[data-testid="stDataFrame"] th, [data-testid="stDataFrame"] td { text-align: center !important; }</style>""", unsafe_allow_html=True)

# 3. EXECUTION
if st.session_state["market_data"] is not None:
    df_market = st.session_state["market_data"]
    df_opt = st.session_state["opt_results"]
    hist_map = st.session_state["historical_data"]
    
    metric_col = "PEG" if mode_select == "Popular and widely followed stocks (P/E/G)" else "PE"

    if df_opt is not None and not df_opt.empty:
        # --- PLOT ---
        st.subheader("2. Portfolio Analysis")
        
        # --- DYNAMIC SCALING LOGIC (Fix for P/E Fallback) ---
        RSI_THRESHOLD = 50
        data_median = df_market[metric_col].median()
        
        if data_median > 5.0:
            # SCENARIO: P/E Data (ETF mode OR Stock mode fallback)
            VAL_THRESHOLD = 25
            MAX_X = max(60, df_market[metric_col].max() * 1.1) 
            x_label = "P/E Ratio (Value)"
            if mode_select == "Popular and widely followed stocks (P/E/G)":
                x_label = "P/E Ratio (PEG Missing - Fallback Applied)"
        else:
            # SCENARIO: PEG Data
            VAL_THRESHOLD = 1.5
            MAX_X = 4.0
            x_label = "PEG Ratio (Growth Value)"
            
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

        # Labels & Lines
        fig_quad.add_vline(x=VAL_THRESHOLD, line_width=1, line_dash="dash", line_color="gray")
        fig_quad.add_hline(y=RSI_THRESHOLD, line_width=1, line_dash="dash", line_color="gray")
        
        fig_quad.add_annotation(x=VAL_THRESHOLD/2, y=90, text="VALUE + MOMENTUM", showarrow=False, font=dict(color="green", size=14, weight="bold"))
        fig_quad.add_annotation(x=VAL_THRESHOLD + (MAX_X-VAL_THRESHOLD)/2, y=90, text="EXPENSIVE MOMENTUM", showarrow=False, font=dict(color="orange", size=10))
        fig_quad.add_annotation(x=VAL_THRESHOLD/2, y=10, text="WEAK / VALUE TRAP", showarrow=False, font=dict(color="orange", size=10))
        fig_quad.add_annotation(x=VAL_THRESHOLD + (MAX_X-VAL_THRESHOLD)/2, y=10, text="EXPENSIVE & WEAK", showarrow=False, font=dict(color="red", size=14, weight="bold"))

        fig_quad.update_xaxes(title_text=x_label, range=[0, MAX_X])
        fig_quad.update_yaxes(title_text="RSI (Momentum)", range=[0, 100])
        fig_quad.update_layout(title=f"{mode_select} Analysis", height=600)
        st.plotly_chart(fig_quad, use_container_width=True)

        # --- EXPLANATION (Methodology Only) ---
        st.divider()
        st.subheader("3. Optimal Portfolio Allocation")
        
        with st.expander("📊 Strategy Breakdown: Allocation Methodology"):
            st.markdown("""
            This model employs a multi-factor approach, optimizing for **Earnings Yield** (Value) and **Relative Strength** (Momentum) under strict variance constraints.
            
            * **Weighting:** The optimal capital allocation coefficient derived from the linear optimization solver.
            * **RSI (Momentum Factor):** The portfolio RSI is the **Weighted Arithmetic Mean** of individual ETFs/stocks and is the metric establishing an ETF's uptrend (>50).
            
            **Note on P/E Calculation (Harmonic Mean):**
            For the Portfolio P/E, we utilize the **Weighted Harmonic Mean** rather than a simple arithmetic average. 
            * *Rationale:* P/E is a ratio of Price to Earnings. Averaging ratios directly can be mathematically misleading due to outliers. The Harmonic Mean correctly averages the underlying "Earnings Yields" (E/P) and inverts the result, providing a true reflection of the portfolio's aggregate valuation.
            """)

        # --- ALLOCATION TABLE (NOW OUTSIDE EXPANDER) ---
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
        
        # --- CHART (NOW OUTSIDE EXPANDER) ---
        st.divider()
        st.subheader("4. Technical Analysis")
        sel_t = st.selectbox("View Asset:", list(hist_map.keys()))
        df_c = hist_map.get(sel_t)
        
        if df_c is not None:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df_c.index, open=df_c['Open'], high=df_c['High'], low=df_c['Low'], close=df_c['Close'], name='OHLC'), row=1, col=1)
            # --- ADDED: 50 & 200 SMA ---
            fig.add_trace(go.Scatter(x=df_c.index, y=df_c['SMA_50'], line=dict(color='orange'), name='SMA 50'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c.index, y=df_c['SMA_200'], line=dict(color='blue', width=2), name='SMA 200'), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df_c.index, y=df_c['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", row=2, col=1, line_color="red")
            fig.add_hline(y=30, line_dash="dot", row=2, col=1, line_color="green")
            fig.update_layout(height=500, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

    # --- MARKET ANALYSIS TABLE ---
    st.divider()
    st.subheader("5. Market Data Analysis")
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

    ### 2. Linear vs. Quadratic Optimization 
    
    

    * **Linear Programming (Factor Exposure):** This tool utilizes Linear Programming (GLOP solver) to maximize direct factor exposure. Instead of minimizing variance through correlation, we mitigate risk via **concentration constraints** (hard limits on max allocation). This allows for a computationally efficient ($O(n)$) maximization of the 'Growth + Momentum' alpha score without the instability often introduced by covariance estimation errors in small samples.
    * **Quadratic Programming (MPT):** Modern Portfolio Theory typically employs Quadratic Programming to minimize portfolio variance ($\sigma^2$). This requires calculating the full covariance matrix $\Sigma$ to account for pairwise asset correlations ($O(n^2)$ complexity). It optimizes for the lowest risk at a given return level.
    """)