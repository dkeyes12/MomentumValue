import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(page_title="Sector Portfolio Optimizer", layout="wide")

# --- UPDATED: S&P 500 SECTOR UNIVERSE ---
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
    data = []
    # Handle list of dicts or list of strings
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
                
                # Note: Some ETFs might not have P/E data in yfinance. 
                # We include them if PE > 0.
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
st.subheader("1. Define Universe (S&P 500 Sectors)")
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
    # --- CSS HACK: Force Center Alignment in Dataframes ---
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
        df_market = fetch_market_data(ticker_list)

        if not df_market.empty:
            st.divider()
            
            # --- MARKET ANALYSIS ---
            st.subheader("2. Market Data Analysis")
            
            st.dataframe(
                df_market.style.format({
                    "PE": "{:.2f}", 
                    "RSI": "{:.2f}", 
                    "Return": "{:.2%}", 
                    "Volatility": "{:.2%}"
                }),
                use_container_width=False 
            )

            with st.spinner("Optimizing..."):
                df_opt = optimize_portfolio(df_market, obj_choice, max_concentration)

            if not df_opt.empty:
                st.subheader(f"2. Optimal Allocation ({obj_choice})")
                
                # --- RESULTS TABLE ---
                display_df = df_opt[["Ticker", "Weight", "PE", "RSI"]].copy()
                display_df["Weight"] = display_df["Weight"].apply(lambda x: f"{x:.1%}")
                display_df["PE"] = display_df["PE"].apply(lambda x: f"{x:.1f}")
                display_df["RSI"] = display_df["RSI"].apply(lambda x: f"{x:.1f}")
                
                st.dataframe(display_df, use_container_width=False)

                # --- VISUALIZATION: SYMMETRIC 2x2 PLOT ---
                st.subheader("3. Portfolio Analysis (Value vs Momentum)")
                
                PE_THRESHOLD = 25  
                RSI_THRESHOLD = 50
                FIXED_MAX_X = PE_THRESHOLD * 2  # 50

                selected_tickers = df_opt['Ticker'].tolist()
                df_remaining = df_market[~df_market['Ticker'].isin(selected_tickers)]

                fig_quad = go.Figure()

                # 1. Plot ONLY the Unselected Stocks (Universe)
                fig_quad.add_trace(go.Scatter(
                    x=df_remaining['PE'].clip(upper=FIXED_MAX_X), 
                    y=df_remaining['RSI'],
                    mode='markers+text',
                    text=df_remaining['Ticker'],
                    textposition="top center",
                    textfont=dict(family="Arial", size=11, color="black"),
                    marker=dict(
                        size=12, 
                        color='rgba(128, 128, 128, 0.5)', 
                        line=dict(width=1, color='dimgray')
                    ),
                    name='Universe'
                ))

                # 2. Plot ONLY the Selected Portfolio Stocks
                fig_quad.add_trace(go.Scatter(
                    x=df_opt['PE'].clip(upper=FIXED_MAX_X), 
                    y=df_opt['RSI'],
                    mode='markers+text',
                    text=df_opt['Ticker'],
                    textposition="top center",
                    textfont=dict(family="Arial Black", size=12, color="black"), 
                    marker=dict(
                        size=18, 
                        color='blue', 
                        line=dict(width=2, color='black')
                    ),
                    name='Selected Portfolio'
                ))

                # 3. Add Colored Quadrant Backgrounds
                fig_quad.add_shape(type="rect", x0=0, y0=RSI_THRESHOLD, x1=PE_THRESHOLD, y1=100,
                                   fillcolor="green", opacity=0.1, layer="below", line_width=0)
                
                fig_quad.add_shape(type="rect", x0=PE_THRESHOLD, y0=RSI_THRESHOLD, x1=FIXED_MAX_X, y1=100,
                                   fillcolor="yellow", opacity=0.1, layer="below", line_width=0)

                fig_quad.add_shape(type="rect", x0=0, y0=0, x1=PE_THRESHOLD, y1=RSI_THRESHOLD,
                                   fillcolor="yellow", opacity=0.1, layer="below", line_width=0)
                
                fig_quad.add_shape(type="rect", x0=PE_THRESHOLD, y0=0, x1=FIXED_MAX_X, y1=RSI_THRESHOLD,
                                   fillcolor="red", opacity=0.1, layer="below", line_width=0)

                # Add Crosshair Lines
                fig_quad.add_vline(x=PE_THRESHOLD, line_width=1, line_dash="dash", line_color="gray")
                fig_quad.add_hline(y=RSI_THRESHOLD, line_width=1, line_dash="dash", line_color="gray")

                # Quadrant Labels
                fig_quad.add_annotation(x=PE_THRESHOLD/2, y=90, text="VALUE + MOMENTUM", showarrow=False, font=dict(color="green", size=14, weight="bold"))
                fig_quad.add_annotation(x=PE_THRESHOLD * 1.5, y=90, text="EXPENSIVE MOMENTUM", showarrow=False, font=dict(color="orange", size=10))
                fig_quad.add_annotation(x=PE_THRESHOLD/2, y=10, text="WEAK / VALUE TRAP", showarrow=False, font=dict(color="orange", size=10))
                fig_quad.add_annotation(x=PE_THRESHOLD * 1.5, y=10, text="EXPENSIVE & WEAK", showarrow=False, font=dict(color="red", size=14, weight="bold"))

                # Strict Axis Range
                fig_quad.update_xaxes(title_text="P/E Ratio (Value)", range=[0, FIXED_MAX_X])
                fig_quad.update_yaxes(title_text="RSI (Momentum)", range=[0, 100])
                fig_quad.update_layout(height=600, title="Market Universe & Selection")

                st.plotly_chart(fig_quad, use_container_width=True)

            else:
                st.error("No optimal solution found. Try relaxing the constraints (increase Max Weight).")
        else:
            st.error("Could not fetch data.")

# --- LOGIC SUMMARY SECTION ---
st.divider()
with st.expander("ℹ️ How the Optimization Logic Works"):
    st.markdown("""
    ### 1. The Scoring Formula
    The optimizer assigns a **"Value-Momentum Score"** to every asset:
    * **Value (50% weight):** Measured by Earnings Yield ($1/PE$). 
    * **Momentum (50% weight):** Measured by RSI. 
    
    $$
    \\text{Score} = \\underbrace{\\left( \\frac{\\text{RSI}}{100} \\right)}_{\\text{Momentum}} + \\underbrace{\\left( \\frac{1}{\\text{PE Ratio}} \\times 50 \\right)}_{\\text{Value}}
    $$

    ### 2. Optimization Modes
    * **📈 Maximize Gain:** The solver finds the exact mix of sectors that maximizes the **Total Portfolio Score**, subject to your concentration limit.
    * **🛡️ Minimize Loss:** The solver finds the mix with the **lowest historical Volatility**. 
        * *Constraint:* The portfolio's average Score must still be at least equal to the market average to ensure quality.
    """)