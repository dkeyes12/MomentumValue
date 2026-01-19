import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- SKFOLIO IMPORTS ---
try:
    from skfolio import Portfolio, Population
    from skfolio.preprocessing import prices_to_returns
    SKFOLIO_AVAILABLE = True
except ImportError:
    SKFOLIO_AVAILABLE = False

# --- CONFIGURATION ---
st.set_page_config(page_title="MomentumValue Unified Dashboard", layout="wide")

# --- SHARED DATASETS (Standardized Sector Names) ---
BENCHMARK_SECTOR_DATA = {
    "Information Technology": 0.315, "Financials": 0.132, "Health Care": 0.124,
    "Consumer Discretionary": 0.103, "Communication Services": 0.088, "Industrials": 0.085,
    "Consumer Staples": 0.061, "Energy": 0.038, "Utilities": 0.024,
    "Real Estate": 0.023, "Materials": 0.022
}

# Updated to match BENCHMARK keys exactly
STOCK_TICKERS = [
    {"Ticker": "NVDA", "Sector": "Information Technology"}, 
    {"Ticker": "AAPL", "Sector": "Information Technology"},
    {"Ticker": "MSFT", "Sector": "Information Technology"}, 
    {"Ticker": "AMZN", "Sector": "Consumer Discretionary"},
    {"Ticker": "GOOGL", "Sector": "Communication Services"}, 
    {"Ticker": "META", "Sector": "Communication Services"},
    {"Ticker": "TSLA", "Sector": "Consumer Discretionary"}, 
    {"Ticker": "JPM", "Sector": "Financials"},
    {"Ticker": "V", "Sector": "Financials"}, 
    {"Ticker": "JNJ", "Sector": "Health Care"},
    {"Ticker": "LLY", "Sector": "Health Care"}, 
    {"Ticker": "XOM", "Sector": "Energy"}
]

# --- SHARED HELPER FUNCTIONS ---

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def plot_quadrant_chart(df, metric_col, rsi_col, weight_col=None, title="Asset Selection Matrix"):
    if df.empty: return go.Figure()

    data_median = df[metric_col].median()
    if pd.isna(data_median) or data_median > 5.0: # Likely P/E
        max_val = df[metric_col].max()
        safe_max = 60 if pd.isna(max_val) else max(60, max_val * 1.1)
        VAL_THRESHOLD = 25; MAX_X = safe_max
        x_label = "P/E Ratio (Lower is Better)"
    else: # Likely PEG
        VAL_THRESHOLD = 1.5; MAX_X = 4.0
        x_label = "PEG Ratio (Lower is Better)"
    
    fig = go.Figure()
    sizes = df[weight_col] * 200 if weight_col and weight_col in df.columns else 15

    fig.add_trace(go.Scatter(
        x=df[metric_col].clip(upper=MAX_X), y=df[rsi_col],
        mode='markers+text', text=df['Ticker'], textposition="top center",
        marker=dict(size=sizes, color=df[rsi_col], colorscale='RdYlGn', showscale=True, colorbar=dict(title="RSI Strength")),
        hovertemplate="<b>%{text}</b><br>RSI: %{y:.1f}<br>Valuation: %{x:.2f}<extra></extra>"
    ))

    fig.add_shape(type="rect", x0=0, y0=50, x1=VAL_THRESHOLD, y1=100, fillcolor="green", opacity=0.1, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=VAL_THRESHOLD, y0=50, x1=MAX_X, y1=100, fillcolor="yellow", opacity=0.1, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=0, y0=0, x1=VAL_THRESHOLD, y1=50, fillcolor="yellow", opacity=0.1, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=VAL_THRESHOLD, y0=0, x1=MAX_X, y1=50, fillcolor="red", opacity=0.1, layer="below", line_width=0)
    
    fig.add_annotation(x=VAL_THRESHOLD/2, y=95, text="VALUE + MOMENTUM", showarrow=False, font=dict(color="green", weight="bold"))
    fig.add_annotation(x=VAL_THRESHOLD + (MAX_X-VAL_THRESHOLD)/2, y=95, text="EXPENSIVE MOMENTUM", showarrow=False, font=dict(color="orange", size=10))
    fig.add_annotation(x=VAL_THRESHOLD/2, y=5, text="WEAK / VALUE TRAP", showarrow=False, font=dict(color="orange", size=10))
    fig.add_annotation(x=VAL_THRESHOLD + (MAX_X-VAL_THRESHOLD)/2, y=5, text="EXPENSIVE & WEAK", showarrow=False, font=dict(color="red", weight="bold"))

    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title="Momentum (RSI)", height=550)
    return fig

@st.cache_data
def process_bulk_data(tickers, sector_map, mode, period="5y"):
    ticker_list = [t.upper().strip() for t in tickers if t.strip()]
    if not ticker_list: return None, None
    
    try:
        bulk_data = yf.download(ticker_list, period=period, group_by='ticker', auto_adjust=False)
    except Exception:
        return None, None
    
    snapshot_data = []
    hist_data = {}

    for t in ticker_list:
        try:
            if len(ticker_list) == 1: df = bulk_data.copy()
            else: df = bulk_data[t].copy()
            
            if df.empty: continue
            df = df.dropna(how='all')
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            df['RSI'] = calculate_rsi(df['Close'])
            hist_data[t] = df

            rsi = df['RSI'].iloc[-1]
            vol = df['Close'].pct_change().std() * np.sqrt(252)
            ret = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
            
            try:
                info = yf.Ticker(t).info
                if "P/E/G" in mode: val = info.get('pegRatio') or info.get('trailingPE')
                else: val = info.get('trailingPE') or info.get('forwardPE')
            except: val = np.nan
            
            if val and val > 0:
                snapshot_data.append({
                    "Ticker": t, "Sector": sector_map.get(t, "Unknown"),
                    "Price": df['Close'].iloc[-1], "RSI": rsi,
                    "Volatility": vol, "Return": ret,
                    "PEG" if "P/E/G" in mode else "PE": val
                })
        except: continue
            
    return pd.DataFrame(snapshot_data), hist_data

def optimize_portfolio(df, objective_type, max_weight_per_asset, mode, sector_limits=None):
    """
    Optimizes portfolio with optional Sector Constraints fed from Mode 1.
    """
    if df is None or df.empty: return pd.DataFrame()
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver: return None

    # Define Variables
    weights = [solver.NumVar(0.0, max_weight_per_asset, f'w_{i}') for i in range(len(df))]
    
    # Constraint 1: Sum of weights = 100%
    constraint_sum = solver.Constraint(1.0, 1.0)
    for w in weights: constraint_sum.SetCoefficient(w, 1)

    # Constraint 2: Sector Limits (From Mode 1)
    if sector_limits:
        # Group indices by sector
        sector_groups = {}
        for i, row in df.iterrows():
            sec = row['Sector']
            if sec not in sector_groups: sector_groups[sec] = []
            sector_groups[sec].append(i)
        
        # Apply constraint for each sector present in the dataframe
        for sec, indices in sector_groups.items():
            if sec in sector_limits:
                limit = sector_limits[sec]
                # Constraint: Sum(weights of sector) <= Limit
                # We use <= to allow cash or flexibility, though typically it matches exactly if optimization is efficient
                c_sec = solver.Constraint(0.0, limit)
                for idx in indices:
                    c_sec.SetCoefficient(weights[idx], 1)

    # Objective Function
    metric_col = "PEG" if "P/E/G" in mode else "PE"
    if metric_col == "PEG": scores = (df['RSI'] / 100) + (1 / df['PEG']) 
    else: scores = (df['RSI'] / 100) + ((1 / df['PE']) * 50)
    
    objective = solver.Objective()
    if "Gain" in objective_type:
        for i, w in enumerate(weights): objective.SetCoefficient(w, scores.iloc[i])
        objective.SetMaximization()
    else:
        avg_score = scores.mean()
        c_qual = solver.Constraint(avg_score, solver.infinity())
        for i, w in enumerate(weights): c_qual.SetCoefficient(w, scores.iloc[i])
        for i, w in enumerate(weights): objective.SetCoefficient(w, df['Volatility'].iloc[i])
        objective.SetMinimization()

    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        results = []
        for i, w in enumerate(weights):
            if w.solution_value() > 0.001:
                row = df.iloc[i].to_dict()
                row["Weight"] = w.solution_value()
                results.append(row)
        return pd.DataFrame(results)
    return pd.DataFrame()

# --- MODE 1: SECTOR REBALANCER LOGIC ---
def run_sector_rebalancer():
    st.header("Mode 1: Top-Down S&P 500 Rebalancer")
    st.markdown("Adjust broad market sector weights. **These targets will be saved for Mode 2.**")
    
    current_tech_pct = BENCHMARK_SECTOR_DATA["Information Technology"] * 100
    
    col_slide, col_metrics = st.columns([1, 2])
    with col_slide:
        tech_cap = st.slider("Max Tech Exposure (%)", 0.0, 60.0, float(current_tech_pct), 0.5)
        
    target_tech = tech_cap / 100.0
    rem_weight_new = 1.0 - target_tech
    rem_weight_old = 1.0 - BENCHMARK_SECTOR_DATA["Information Technology"]
    scale = rem_weight_new / rem_weight_old if rem_weight_old > 0 else 0
    
    new_alloc = {}
    for sec, w in BENCHMARK_SECTOR_DATA.items():
        if sec == "Information Technology": new_alloc[sec] = target_tech
        else: new_alloc[sec] = w * scale
    
    # --- SAVE TO SESSION STATE FOR MODE 2 ---
    st.session_state["sector_targets"] = new_alloc
    st.success(f"✅ Sector targets saved! Tech constrained to {tech_cap}%. Go to 'Portfolio Optimizer' to apply them.")

    df_alloc = pd.DataFrame([{"Sector": k, "Benchmark": v, "Custom": new_alloc[k]} for k, v in BENCHMARK_SECTOR_DATA.items()])
    df_alloc["Delta"] = df_alloc["Custom"] - df_alloc["Benchmark"]
    
    with col_metrics:
        m1, m2 = st.columns(2)
        m1.metric("Tech Weight", f"{tech_cap:.1f}%", f"{tech_cap - current_tech_pct:.1f}%", delta_color="inverse")
        active_share = np.sum(np.abs(df_alloc["Custom"] - df_alloc["Benchmark"])) / 2
        m2.metric("Active Share", f"{active_share:.1%}")

    # Charts
    tab1, tab2 = st.tabs(["📊 Sector Weights", "📝 Data"])
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_alloc["Sector"], y=df_alloc["Benchmark"], name="Benchmark", marker_color="lightgray"))
        fig.add_trace(go.Bar(x=df_alloc["Sector"], y=df_alloc["Custom"], name="Rebalanced", marker_color="#4F8BF9"))
        fig.update_layout(barmode='group', height=400, yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        st.dataframe(df_alloc.style.format({"Benchmark": "{:.1%}", "Custom": "{:.1%}", "Delta": "{:+.1%}"}), use_container_width=True)

# --- MODE 2: STOCK OPTIMIZER LOGIC ---
def run_stock_optimizer():
    st.header("Mode 2: Bottom-Up Stock Optimizer")
    
    # Inputs
    col_conf, col_univ = st.columns([1, 2])
    with col_conf:
        mode_sel = st.radio("Metrics:", ["Popular (P/E/G)", "Standard (P/E)"])
        obj = st.radio("Objective:", ["Maximize Gain", "Minimize Volatility"])
        max_w = st.slider("Max Weight (Individual)", 0.05, 1.0, 0.25, 0.05)
        
        # --- SECTOR CONSTRAINT TOGGLE ---
        use_sector_limits = False
        if "sector_targets" in st.session_state:
            st.divider()
            st.markdown("🔗 **Macro Link Active**")
            use_sector_limits = st.checkbox("Apply Sector Limits from Mode 1?", value=True)
            if use_sector_limits:
                tech_limit = st.session_state['sector_targets'].get('Information Technology', 0)
                st.caption(f"Constraints active: Tech ≤ {tech_limit:.1%}, etc.")
        
    with col_univ:
        if "user_tickers" not in st.session_state: 
            st.session_state["user_tickers"] = pd.DataFrame(STOCK_TICKERS)
        edited = st.data_editor(st.session_state["user_tickers"], num_rows="dynamic", use_container_width=True)
        
        if st.button("🚀 Optimize Portfolio", type="primary"):
            with st.spinner("Fetching data..."):
                t_list = edited["Ticker"].tolist()
                s_map = dict(zip(edited["Ticker"], edited["Sector"]))
                df_mkt, hist = process_bulk_data(t_list, s_map, mode_sel)
                
                if df_mkt is not None and not df_mkt.empty:
                    # PASS SECTOR LIMITS IF TOGGLED
                    limits = st.session_state["sector_targets"] if use_sector_limits else None
                    
                    df_res = optimize_portfolio(df_mkt, obj, max_w, mode_sel, sector_limits=limits)
                    
                    st.session_state["opt_res"] = df_res
                    st.session_state["mkt_data"] = df_mkt
                    st.session_state["hist_data"] = hist
                else:
                    st.error("Data fetch failed or returned empty universe.")

    # Results
    if "opt_res" in st.session_state and st.session_state["opt_res"] is not None and not st.session_state["opt_res"].empty:
        df_res = st.session_state["opt_res"]
        df_mkt = st.session_state["mkt_data"]
        hist = st.session_state["hist_data"]
        metric_col = "PEG" if "P/E/G" in mode_sel else "PE"

        st.divider()
        st.subheader("1. Asset Selection Matrix")
        fig_quad = plot_quadrant_chart(df_res, metric_col, "RSI", weight_col="Weight")
        st.plotly_chart(fig_quad, use_container_width=True)

        st.subheader("2. Optimal Allocation")
        col_tbl, col_tv = st.columns([2, 1])
        with col_tbl:
            disp = df_res.sort_values("Weight", ascending=False).copy()
            tot_w = disp["Weight"].sum()
            w_rsi = (disp["Weight"]*disp["RSI"]).sum()/tot_w if tot_w > 0 else 0
            w_vol = (disp["Weight"]*disp["Volatility"]).sum()/tot_w if tot_w > 0 else 0
            w_val_inv = (disp["Weight"]/disp[metric_col]).sum()
            w_val = tot_w/w_val_inv if w_val_inv > 0 else 0
            
            sum_row = pd.DataFrame([{"Ticker": "TOTAL", "Weight": tot_w, "RSI": w_rsi, "Volatility": w_vol, metric_col: w_val}])
            final = pd.concat([disp, sum_row], ignore_index=True)
            
            st.dataframe(final.style.format({"Weight": "{:.1%}", "RSI": "{:.1f}", "Volatility": "{:.1%}", metric_col: "{:.2f}"}), use_container_width=True)
            
        with col_tv:
            with st.expander("📤 TradingView Export"):
                st.write("Copy/Paste to Pine Editor:")
                lines = ["//@version=5", "indicator('Portfolio', overlay=true)", f"var table t = table.new(position.top_right, 2, {len(df_res)+1})"]
                for idx, (i, r) in enumerate(disp.iterrows()):
                    lines.append(f"table.cell(t, 0, {idx}, '{r['Ticker']}')")
                    lines.append(f"table.cell(t, 1, {idx}, '{r['Weight']*100:.1f}%')")
                st.code("\n".join(lines))

        st.divider()
        st.subheader("3. Backtest Analysis")
        if SKFOLIO_AVAILABLE and hist:
            try:
                p_df = pd.DataFrame({t: d['Close'] for t, d in hist.items()}).dropna()
                X = prices_to_returns(p_df)
                sel_ticks = df_res["Ticker"].tolist()
                sel_ws = df_res["Weight"].tolist()
                X_strat = X[sel_ticks]
                
                strat = Portfolio(X=X_strat, weights=sel_ws, name="Optimized")
                bench = Portfolio(X=X, weights=[1/len(X.columns)]*len(X.columns), name="Equal Weight Universe")
                pop = Population([strat, bench])
                
                c1, c2 = st.columns(2)
                with c1: st.plotly_chart(pop.plot_cumulative_returns(), use_container_width=True)
                with c2: st.dataframe(pop.summary().astype(str), use_container_width=True)
            except Exception as e:
                st.error(f"Backtest Error: {e}")

# --- MAIN CONTROLLER ---
def run_app():
    st.sidebar.title("MomentumValue")
    app_mode = st.sidebar.radio("Select Mode:", ["Sector Rebalancer (Macro)", "Portfolio Optimizer (Micro)"])
    st.sidebar.divider()
    
    if app_mode == "Sector Rebalancer (Macro)":
        run_sector_rebalancer()
    else:
        run_stock_optimizer()

if __name__ == "__main__":
    try:
        run_app()
    except Exception as e:
        st.error(f"Application Error: {e}")