import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CONFIGURATION ---
st.set_page_config(page_title="Sector Portfolio Optimizer", layout="wide")

# --- S&P 500 SECTOR UNIVERSE ---
DEFAULT_TICKERS = [
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

# --- HELPER FUNCTIONS ---
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@st.cache_data
def fetch_market_data(tickers, period="1y"):
    """Fetches snapshot data for the Optimizer"""
    data = []
    ticker_list = [t["Ticker"] for t in tickers] if isinstance(tickers[0], dict) else tickers
    
    with st.spinner(f"Fetching data for {len(ticker_list)} assets..."):
        for t in ticker_list:
            try:
                t = t.upper().strip()
                if not t: continue
                
                stock = yf.Ticker(t)
                hist = stock.history(period=period)
                
                if hist.empty: continue

                current_price = hist['Close'].iloc[-1]
                rsi = calculate_rsi(hist['Close']).iloc[-1]
                volatility = hist['Close'].pct_change().std() * np.sqrt(252)
                
                pe = stock.info.get('trailingPE')
                if pe is None: pe = stock.info.get('forwardPE')
                
                if pe is not None and pe > 0:
                    data.append({
                        "Ticker": t,
                        "Price": current_price,
                        "PE": pe,
                        "RSI": rsi,
                        "Volatility": volatility
                    })
            except Exception:
                continue
                
    return pd.DataFrame(data)

def get_chart_data(ticker, period="2y"):
    """Fetches historical data specifically for the Chart"""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty: return None
    
    # Calculate Indicators for the chart
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    return df

# --- OPTIMIZATION ENGINE ---
def optimize_portfolio(df, objective_type, max_weight_per_asset):
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver: return None

    weights = []
    for i in range(len(df)):
        weights.append(solver.NumVar(0.0, max_weight_per_asset, f'w_{i}'))

    constraint_sum = solver.Constraint(1.0, 1.0)
    for w in weights:
        constraint_sum.SetCoefficient(w, 1)

    objective = solver.Objective()
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
                results.append({
                    "Ticker": df['Ticker'].iloc[i],
                    "Weight": w.solution_value(),
                    "RSI": df['RSI'].iloc[i],
                    "PE": df['PE'].iloc[i],
                    "Volatility": df['Volatility'].iloc[i]
                })
        return pd.DataFrame(results)
    else:
        return pd.DataFrame()

# --- DASHBOARD UI ---
st.title("⚖️ Sector Rotation Optimizer")

# 1. SIDEBAR
with st.sidebar:
    st.header("Settings")
    obj_choice = st.radio("Goal", ["Maximize Gain (Score)", "Minimize Loss (Volatility)"])
    max_concentration = st.slider("Max Allocation per Sector", 0.05, 1.0, 0.25, 0.05)
    st.info("Edit the table to add individual stocks or other ETFs.")

# 2. EDITABLE STOCK INPUT
st.subheader("1. Define Universe")
col_input, col_action = st.columns([3, 1])

with col_input:
    if "user_tickers" not in st.session_state:
        st.session_state["user_tickers"] = pd.DataFrame(DEFAULT_TICKERS)

    edited_df = st.data_editor(
        st.session_state["user_tickers"], 
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="small", help="Symbol"),
            "Sector": st.column_config.TextColumn("Sector", width="medium"),
        },
        num_rows="dynamic", 
        use_container_width=True,
        key="ticker_editor"
    )

with col_action:
    st.write("### ") 
    run_optimization = st.button("🚀 Run Optimization", type="primary", use_container_width=True)

# 3. EXECUTION
if run_optimization:
    # CSS for centering
    st.markdown("""
    <style>
        [data-testid="stDataFrame"] th { text-align: center !important; }
        [data-testid="stDataFrame"] td { text-align: center !important; }
    </style>
    """, unsafe_allow_html=True)

    ticker_list = edited_df["Ticker"].tolist()
    
    if len(ticker_list) < 2:
        st.error("Please enter at least 2 tickers.")
    else:
        # Fetch Data
        df_market = fetch_market_data(ticker_list)

        if not df_market.empty:
            st.divider()
            
            # Run Optimization
            with st.spinner("Optimizing..."):
                df_opt = optimize_portfolio(df_market, obj_choice, max_concentration)

            if not df_opt.empty:
                
                # --- VISUALIZATION: SYMMETRIC 2x2 PLOT ---
                st.subheader("2. Portfolio Analysis (Value vs Momentum)")
                
                PE_THRESHOLD = 25  
                RSI_THRESHOLD = 50
                FIXED_MAX_X = PE_THRESHOLD * 2  # 50

                selected_tickers = df_opt['Ticker'].tolist()
                df_remaining = df_market[~df_market['Ticker'].isin(selected_tickers)]

                fig_quad = go.Figure()

                # 1. Plot Unselected (Universe)
                fig_quad.add_trace(go.Scatter(
                    x=df_remaining['PE'].clip(upper=FIXED_MAX_X), 
                    y=df_remaining['RSI'],
                    mode='markers+text',
                    text=df_remaining['Ticker'],
                    textposition="top center",
                    textfont=dict(family="Arial", size=11, color="black"),
                    marker=dict(size=12, color='rgba(128, 128, 128, 0.5)', line=dict(width=1, color='dimgray')),
                    name='Universe'
                ))

                # 2. Plot Selected (Portfolio)
                fig_quad.add_trace(go.Scatter(
                    x=df_opt['PE'].clip(upper=FIXED_MAX_X), 
                    y=df_opt['RSI'],
                    mode='markers+text',
                    text=df_opt['Ticker'],
                    textposition="top center",
                    textfont=dict(family="Arial Black", size=12, color="black"), 
                    marker=dict(size=18, color='blue', line=dict(width=2, color='black')),
                    name='Selected Portfolio'
                ))

                # 3. Backgrounds
                fig_quad.add_shape(type="rect", x0=0, y0=RSI_THRESHOLD, x1=PE_THRESHOLD, y1=100, fillcolor="green", opacity=0.1, layer="below", line_width=0)
                fig_quad.add_shape(type="rect", x0=PE_THRESHOLD, y0=RSI_THRESHOLD, x1=FIXED_MAX_X, y1=100, fillcolor="yellow", opacity=0.1, layer="below", line_width=0)
                fig_quad.add_shape(type="rect", x0=0, y0=0, x1=PE_THRESHOLD, y1=RSI_THRESHOLD, fillcolor="yellow", opacity=0.1, layer="below", line_width=0)
                fig_quad.add_shape(type="rect", x0=PE_THRESHOLD, y0=0, x1=FIXED_MAX_X, y1=RSI_THRESHOLD, fillcolor="red", opacity=0.1, layer="below", line_width=0)

                # Lines & Labels
                fig_quad.add_vline(x=PE_THRESHOLD, line_width=1, line_dash="dash", line_color="gray")
                fig_quad.add_hline(y=RSI_THRESHOLD, line_width=1, line_dash="dash", line_color="gray")
                fig_quad.add_annotation(x=PE_THRESHOLD/2, y=90, text="VALUE + MOMENTUM", showarrow=False, font=dict(color="green", size=14, weight="bold"))
                fig_quad.add_annotation(x=PE_THRESHOLD * 1.5, y=90, text="EXPENSIVE MOMENTUM", showarrow=False, font=dict(color="orange", size=10))
                fig_quad.add_annotation(x=PE_THRESHOLD/2, y=10, text="WEAK / VALUE TRAP", showarrow=False, font=dict(color="orange", size=10))
                fig_quad.add_annotation(x=PE_THRESHOLD * 1.5, y=10, text="EXPENSIVE & WEAK", showarrow=False, font=dict(color="red", size=14, weight="bold"))

                fig_quad.update_xaxes(title_text="P/E Ratio (Value)", range=[0, FIXED_MAX_X])
                fig_quad.update_yaxes(title_text="RSI (Momentum)", range=[0, 100])
                fig_quad.update_layout(height=600, title="Market Universe & Selection")

                st.plotly_chart(fig_quad, use_container_width=True)

                # --- EXPLANATION & ALLOCATION TABLE ---
                st.divider()
                st.subheader("3. Optimal Portfolio Allocation")
                
                with st.expander("📊 Strategy Breakdown: Allocation Methodology"):
                    st.markdown(r"""
                    This model employs a multi-factor approach, optimizing for **Earnings Yield** (Value) and **Relative Strength** (Momentum) under strict variance constraints.
                    
                    * **Weighting ($w$):** The optimal capital allocation coefficient derived from the linear optimization solver.
                    * **RSI (Weighted Arithmetic Mean):** The portfolio RSI is the linear weighted average of individual constituents. This represents the momentum 'center of mass' for the portfolio.
                        $$ \text{Portfolio RSI} = \sum (w_i \cdot RSI_i) $$
                    * **P/E Ratio (Weighted Harmonic Mean):** For the Portfolio P/E, we utilize the **Weighted Harmonic Mean** rather than a simple arithmetic average. 
                        $$ \text{Portfolio P/E} = \frac{1}{\sum (w_i \cdot \frac{1}{PE_i})} $$
                        *Rationale:* P/E is a ratio of Price to Earnings. Averaging ratios directly is mathematically incorrect. The Harmonic Mean correctly averages the underlying "Earnings Yields" (E/P) and inverts the result, providing a true reflection of the portfolio's aggregate valuation.
                    """)

                # --- CALCULATION OF TOTALS ---
                display_df = df_opt[["Ticker", "Weight", "PE", "RSI"]].copy()

                # 1. Weighted RSI (Arithmetic Mean)
                weighted_rsi = (display_df['Weight'] * display_df['RSI']).sum()
                
                # 2. Weighted PE (Harmonic Mean)
                weighted_earnings_yield = (display_df['Weight'] / display_df['PE']).sum()
                portfolio_pe = 1 / weighted_earnings_yield if weighted_earnings_yield > 0 else 0

                # 3. Create Summary Row
                summary_row = pd.DataFrame([{
                    "Ticker": "PORTFOLIO TOTAL",
                    "Weight": display_df['Weight'].sum(),
                    "PE": portfolio_pe,
                    "RSI": weighted_rsi
                }])

                # 4. Append and Format
                final_table = pd.concat([display_df, summary_row], ignore_index=True)
                
                # Formatting
                final_table["Weight"] = final_table["Weight"].apply(lambda x: f"{x:.1%}")
                final_table["PE"] = final_table["PE"].apply(lambda x: f"{x:.1f}")
                final_table["RSI"] = final_table["RSI"].apply(lambda x: f"{x:.1f}")
                
                # Dynamic Height Calculation (35px per row + header buffer)
                height_px = (len(final_table) + 1) * 35

                st.dataframe(
                    final_table, 
                    use_container_width=False,
                    height=height_px  # <--- FIX: Forces table to expand so Summary Row is visible
                )
                
                # --- TECHNICAL ANALYSIS ---
                st.divider()
                st.subheader("4. Technical Analysis (Deep Dive)")
                
                col_sel, _ = st.columns([1, 3])
                with col_sel:
                    chart_ticker = st.selectbox("Select Asset to View:", ticker_list)
                
                rsi_source = "Close" 
                
                with st.spinner(f"Loading chart for {chart_ticker}..."):
                    df_chart = get_chart_data(chart_ticker)
                
                if df_chart is not None:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.7, 0.3])

                    # Price Chart
                    fig.add_trace(go.Candlestick(x=df_chart.index,
                                        open=df_chart['Open'], high=df_chart['High'],
                                        low=df_chart['Low'], close=df_chart['Close'], name='OHLC'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA_50'], 
                                    line=dict(color='orange', width=2), name='50 Day MA'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA_200'], 
                                    line=dict(color='blue', width=2), name='200 Day MA'), row=1, col=1)

                    # RSI Chart
                    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI'], 
                                    line=dict(color='purple', width=2), name=f'RSI ({rsi_source})'), row=2, col=1)
                    fig.add_hline(y=70, line_dash="dot", row=2, col=1, line_color="red")
                    fig.add_hline(y=30, line_dash="dot", row=2, col=1, line_color="green")
                    fig.add_hline(y=50, line_dash="solid", row=2, col=1, line_color="gray", opacity=0.5)

                    fig.update_layout(xaxis_rangeslider_visible=False, height=600, margin=dict(l=20, r=20, t=30, b=20))
                    
                    st.plotly_chart(fig, use_container_width=True)

            else:
                st.error("No optimal solution found. Try relaxing the constraints (increase Max Weight).")
        else:
            st.error("Could not fetch data.")

# --- LOGIC SUMMARY SECTION ---
st.divider()
with st.expander("ℹ️ How the Optimization Logic Works"):
    st.markdown(r"""
    ### 1. The Scoring Formula
    The optimizer assigns a **"Value-Momentum Score"** to every asset:
    * **Value (50% weight):** Measured by Earnings Yield ($1/PE$). 
    * **Momentum (50% weight):** Measured by RSI. 
    
    $$
    \text{Score} = \underbrace{\left( \frac{\text{RSI}}{100} \right)}_{\text{Momentum}} + \underbrace{\left( \frac{1}{\text{PE Ratio}} \times 50 \right)}_{\text{Value}}
    $$
    """)