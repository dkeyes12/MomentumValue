import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

# --- SKFOLIO IMPORTS ---
try:
    from skfolio import Portfolio, Population
    from skfolio.preprocessing import prices_to_returns
    SKFOLIO_AVAILABLE = True
except ImportError:
    SKFOLIO_AVAILABLE = False

# --- CONFIGURATION ---
st.set_page_config(page_title="MomentumValue Unified Dashboard", layout="wide")

# --- SHARED DATASETS ---
BENCHMARK_SECTOR_DATA = {
    "Information Technology": 0.350, 
    "Financials": 0.130, 
    "Health Care": 0.120,
    "Consumer Discretionary": 0.100, 
    "Communication Services": 0.090, 
    "Industrials": 0.080,
    "Consumer Staples": 0.060, 
    "Energy": 0.035, 
    "Utilities": 0.020,
    "Real Estate": 0.020, 
    "Materials": 0.020
}

SECTOR_TICKER_MAP = {
    "Information Technology": "XLK", "Health Care": "XLV", "Financials": "XLF",
    "Real Estate": "XLRE", "Energy": "XLE", "Materials": "XLB",
    "Consumer Discretionary": "XLY", "Consumer Staples": "XLP",
    "Industrials": "XLI", "Utilities": "XLU", "Communication Services": "XLC"
}

DEFAULT_TICKERS = [
    {"Ticker": "XLK", "Sector": "Information Technology"},
    {"Ticker": "XLV", "Sector": "Health Care"},
    {"Ticker": "XLF", "Sector": "Financials"},
    {"Ticker": "XLRE", "Sector": "Real Estate"},
    {"Ticker": "XLE", "Sector": "Energy"},
    {"Ticker": "XLB", "Sector": "Materials"},
    {"Ticker": "XLY", "Sector": "Consumer Discretionary"},
    {"Ticker": "XLP", "Sector": "Consumer Staples"},
    {"Ticker": "XLI", "Sector": "Industrials"},
    {"Ticker": "XLU", "Sector": "Utilities"},
    {"Ticker": "XLC", "Sector": "Communication Services"}
]

# --- SHARED HELPER FUNCTIONS ---

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=3600) 
def get_live_tech_weight(base_weight=0.350):
    try:
        tickers = yf.tickers.Tickers("XLK SPY")
        hist = tickers.history(period="1d")
        if not hist.empty and "Close" in hist.columns:
            xlk_ret = (hist["Close"]["XLK"].iloc[-1] - hist["Open"]["XLK"].iloc[0]) / hist["Open"]["XLK"].iloc[0]
            spy_ret = (hist["Close"]["SPY"].iloc[-1] - hist["Open"]["SPY"].iloc[0]) / hist["Open"]["SPY"].iloc[0]
            relative_perf = (1 + xlk_ret) / (1 + spy_ret)
            estimated_weight = base_weight * relative_perf
            return estimated_weight * 100 
    except Exception:
        pass
    return base_weight * 100

def plot_quadrant_chart(df_in, metric_col, rsi_col, weight_col=None, title="Asset Selection Matrix"):
    if df_in.empty: return go.Figure()
    
    df = df_in.copy()
    data_median = df[metric_col].median()
    if pd.isna(data_median) or data_median > 5.0: 
        max_val = df[metric_col].max()
        safe_max = 60 if pd.isna(max_val) else max(60, max_val * 1.1)
        VAL_THRESHOLD = 25; MAX_X = safe_max
        x_label = "P/E Ratio (Lower is Better)"
    else: 
        VAL_THRESHOLD = 1.5; MAX_X = 4.0
        x_label = "PEG Ratio (Lower is Better)"
    
    fig = go.Figure()

    if weight_col and weight_col in df.columns:
        df['Is_Selected'] = df[weight_col] > 0.0001
    else:
        df['Is_Selected'] = True

    df_unselected = df[~df['Is_Selected']]
    if not df_unselected.empty:
        fig.add_trace(go.Scatter(
            x=df_unselected[metric_col].clip(upper=MAX_X), y=df_unselected[rsi_col],
            mode='markers+text', text=df_unselected['Ticker'], textposition="top center",
            marker=dict(size=12, color='#E0E0E0', line=dict(width=1, color='#A0A0A0')),
            hovertemplate="<b>%{text}</b><br>RSI: %{y:.1f}<br>Valuation: %{x:.2f}<br>Status: Not Selected<extra></extra>",
            name="Unselected"
        ))

    df_selected = df[df['Is_Selected']]
    if not df_selected.empty:
        fig.add_trace(go.Scatter(
            x=df_selected[metric_col].clip(upper=MAX_X), y=df_selected[rsi_col],
            mode='markers+text', text=df_selected['Ticker'], textposition="top center",
            marker=dict(
                size=12, 
                color=df_selected[rsi_col], colorscale='RdYlGn_r', showscale=True, 
                colorbar=dict(title="RSI (Red=High)"), line=dict(width=1, color='DarkSlateGrey')
            ),
            hovertemplate="<b>%{text}</b><br>RSI: %{y:.1f}<br>Valuation: %{x:.2f}<br>Weight: %{customdata[0]:.2%}<extra></extra>",
            customdata=df_selected[[weight_col]] if weight_col in df_selected.columns else np.zeros((len(df_selected), 1)),
            name="Selected"
        ))

    fig.add_shape(type="rect", x0=0, y0=50, x1=VAL_THRESHOLD, y1=100, fillcolor="green", opacity=0.1, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=VAL_THRESHOLD, y0=50, x1=MAX_X, y1=100, fillcolor="orange", opacity=0.1, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=0, y0=0, x1=VAL_THRESHOLD, y1=50, fillcolor="yellow", opacity=0.1, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=VAL_THRESHOLD, y0=0, x1=MAX_X, y1=50, fillcolor="red", opacity=0.1, layer="below", line_width=0)
    
    fig.add_annotation(x=VAL_THRESHOLD/2, y=95, text="VALUE + MOMENTUM", showarrow=False, font=dict(color="green", weight="bold"))
    fig.add_annotation(x=VAL_THRESHOLD + (MAX_X-VAL_THRESHOLD)/2, y=95, text="EXPENSIVE MOMENTUM", showarrow=False, font=dict(color="orange", size=10))
    fig.add_annotation(x=VAL_THRESHOLD/2, y=5, text="WEAK / VALUE TRAP", showarrow=False, font=dict(color="orange", size=10))
    fig.add_annotation(x=VAL_THRESHOLD + (MAX_X-VAL_THRESHOLD)/2, y=5, text="EXPENSIVE & WEAK", showarrow=False, font=dict(color="red", weight="bold"))

    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title="Momentum (RSI)", height=600, showlegend=False)
    return fig

@st.cache_data
def process_bulk_data(tickers, sector_map, mode, period="5y"):
    ticker_list = [t.upper().strip() for t in tickers if t.strip()]
    if not ticker_list: return None, None, None
    
    try:
        bulk_data = yf.download(ticker_list, period=period, group_by='ticker', auto_adjust=False)
    except Exception:
        return None, None, None
    
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
    
    final_df = pd.DataFrame(snapshot_data)
    
    cov_matrix = None
    if not final_df.empty and hist_data:
        valid_tickers = final_df['Ticker'].tolist()
        try:
            price_df = pd.DataFrame({t: hist_data[t]['Close'] for t in valid_tickers}).dropna()
            if not price_df.empty:
                cov_matrix = price_df.pct_change().cov()
        except:
            cov_matrix = None
            
    return final_df, hist_data, cov_matrix

def optimize_portfolio(df, objective_type, max_weight_per_asset, mode, sector_limits=None, cov_matrix=None):
    if df is None or df.empty: return pd.DataFrame()

    metric_col = "PEG" if "P/E/G" in mode else "PE"
    if metric_col == "PEG": scores = (df['RSI'] / 100) + (1 / df['PEG']) 
    else: scores = (df['RSI'] / 100) + ((1 / df['PE']) * 50)
    avg_score = scores.mean()

    # --- INTELLIGENT BOUNDS CALCULATION ---
    bounds_list = []
    sector_counts = df['Sector'].value_counts().to_dict()
    
    for i, row in df.iterrows():
        sec = row['Sector']
        upper = max_weight_per_asset 
        
        if sector_limits and sec in sector_limits:
            limit = sector_limits[sec]
            count = sector_counts.get(sec, 1)
            capacity = count * max_weight_per_asset
            
            if capacity < limit:
                upper = limit 
        
        bounds_list.append((0.0, upper))

    # ==========================================
    # CASE 1: MINIMIZE VOLATILITY (QUADRATIC)
    # ==========================================
    if "Volatility" in objective_type and cov_matrix is not None:
        num_assets = len(df)
        
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix.values, weights))
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        constraints.append({'type': 'ineq', 'fun': lambda x: np.dot(x, scores.values) - avg_score})
        
        if sector_limits:
            sector_map_indices = {}
            for i, row in df.iterrows():
                sec = row['Sector']
                if sec not in sector_map_indices: sector_map_indices[sec] = []
                sector_map_indices[sec].append(i)
            
            for sec, limit in sector_limits.items():
                if sec in sector_map_indices:
                    indices = sector_map_indices[sec]
                    def make_constraint(idx_list, limit_val):
                        if sec == "Information Technology":
                            return lambda x: np.sum(x[idx_list]) - limit_val 
                        else:
                            return lambda x: limit_val - np.sum(x[idx_list]) 
                    
                    c_type = 'eq' if sec == "Information Technology" else 'ineq'
                    constraints.append({'type': c_type, 'fun': make_constraint(indices, limit)})
        
        init_guess = np.array(num_assets * [1. / num_assets,])
        try:
            result = minimize(portfolio_variance, init_guess, method='SLSQP', bounds=tuple(bounds_list), constraints=constraints, tol=1e-6)
            if result.success:
                results = []
                for i, w in enumerate(result.x):
                    row = df.iloc[i].to_dict()
                    row["Weight"] = w
                    results.append(row)
                return pd.DataFrame(results)
        except:
            pass 

    # ==========================================
    # CASE 2: MAXIMIZE GAIN (LINEAR)
    # ==========================================
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver: return None

    weights = []
    for i, (lower, upper) in enumerate(bounds_list):
        weights.append(solver.NumVar(lower, upper, f'w_{i}'))
    
    constraint_sum = solver.Constraint(1.0, 1.0)
    for w in weights: constraint_sum.SetCoefficient(w, 1)

    if sector_limits:
        sector_groups = {}
        for i, row in df.iterrows():
            sec = row['Sector']
            if sec not in sector_groups: sector_groups[sec] = []
            sector_groups[sec].append(i)
        
        for sec, indices in sector_groups.items():
            if sec in sector_limits:
                limit = sector_limits[sec]
                if sec == "Information Technology":
                    c_sec = solver.Constraint(limit, limit) 
                else:
                    c_sec = solver.Constraint(0.0, limit)
                
                for idx in indices:
                    c_sec.SetCoefficient(weights[idx], 1)

    objective = solver.Objective()
    if "Gain" in objective_type:
        for i, w in enumerate(weights): objective.SetCoefficient(w, scores.iloc[i])
        objective.SetMaximization()
    else:
        c_qual = solver.Constraint(avg_score, solver.infinity())
        for i, w in enumerate(weights): c_qual.SetCoefficient(w, scores.iloc[i])
        for i, w in enumerate(weights): objective.SetCoefficient(w, df['Volatility'].iloc[i])
        objective.SetMinimization()

    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        results = []
        for i, w in enumerate(weights):
            val = w.solution_value()
            row = df.iloc[i].to_dict()
            row["Weight"] = val
            results.append(row)
        return pd.DataFrame(results)
    return pd.DataFrame()

# --- STEP 1: REBALANCE TECHNOLOGY ---
def run_sector_rebalancer():
    st.header("Step 1: Rebalance Technology")
    
    live_weight = get_live_tech_weight()
    today_str = datetime.today().strftime('%m/%d/%Y')
    st.info(f"📅 Today, {today_str}, technology makes up {live_weight:.1f}% of the S&P500. Historically technology has been 15%.")
    
    st.markdown("Adjust broad market sector weights. **These targets will be saved for Step 2.**")
    
    col_slide, col_metrics = st.columns([1, 2])
    with col_slide:
        tech_cap = st.slider("Max Tech Exposure (%)", 0.0, 60.0, 15.0, 0.5)
        
    target_tech = tech_cap / 100.0
    rem_weight_new = 1.0 - target_tech
    rem_weight_old = 1.0 - BENCHMARK_SECTOR_DATA["Information Technology"]
    scale = rem_weight_new / rem_weight_old if rem_weight_old > 0 else 0
    
    new_alloc = {}
    for sec, w in BENCHMARK_SECTOR_DATA.items():
        if sec == "Information Technology": new_alloc[sec] = target_tech
        else: new_alloc[sec] = w * scale
    
    st.session_state["sector_targets"] = new_alloc
    st.success(f"✅ Sector targets saved! Tech fixed at {tech_cap}%. Go to 'Step 2' to apply them.")

    df_alloc = pd.DataFrame([
        {
            "Sector": k, 
            "Ticker": SECTOR_TICKER_MAP.get(k, "-"), 
            "Benchmark": v, 
            "Custom": new_alloc[k]
        } 
        for k, v in BENCHMARK_SECTOR_DATA.items()
    ])
    df_alloc["Delta"] = df_alloc["Custom"] - df_alloc["Benchmark"]
    
    with col_metrics:
        m1, m2 = st.columns(2)
        m1.metric("Tech Weight", f"{tech_cap:.1f}%", f"{tech_cap - live_weight:.1f}%", delta_color="inverse")
        active_share = np.sum(np.abs(df_alloc["Custom"] - df_alloc["Benchmark"])) / 2
        m2.metric("Active Share", f"{active_share:.1%}")

    tab1, tab2 = st.tabs(["📊 Sector Weights", "📝 Data"])
    with tab1:
        # --- FIXED: Max Height Buffer Calculation ---
        max_y = max(df_alloc["Benchmark"].max(), df_alloc["Custom"].max())
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_alloc["Sector"], y=df_alloc["Benchmark"], 
            name="Benchmark", marker_color="lightgray",
            text=df_alloc["Benchmark"], texttemplate='%{text:.1%}', textposition='outside'
        ))
        fig.add_trace(go.Bar(
            x=df_alloc["Sector"], y=df_alloc["Custom"], 
            name="Rebalanced", marker_color="#4F8BF9",
            text=df_alloc["Custom"], texttemplate='%{text:.1%}', textposition='outside'
        ))
        # Applied Buffer logic here
        fig.update_layout(
            barmode='group', 
            height=400, 
            yaxis=dict(
                tickformat='.0%', 
                range=[0, max_y * 1.1] # 10% Buffer
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        st.dataframe(
            df_alloc.style.format({
                "Benchmark": "{:.1%}", 
                "Custom": "{:.1%}", 
                "Delta": "{:+.1%}"
            }), 
            column_order=["Sector", "Ticker", "Benchmark", "Custom", "Delta"],
            use_container_width=True
        )

# --- STEP 2: OPTIMIZE ---
def run_stock_optimizer():
    st.header("Step 2: Optimize")
    
    col_conf, col_univ = st.columns([1, 2])
    with col_conf:
        mode_sel = st.radio("Metrics:", ["Standard (P/E)", "Popular (P/E/G)"])
        obj = st.radio("Objective:", ["Maximize Gain", "Minimize Volatility"])
        
        use_sector_limits = False
        if "sector_targets" in st.session_state:
            st.divider()
            st.markdown("🔗 **Macro Link Active**")
            use_sector_limits = st.checkbox("Apply Sector Limits from Step 1?", value=True)
            if use_sector_limits:
                tech_lim = st.session_state['sector_targets'].get('Information Technology',0)
                st.caption(f"Constraints active (Tech = {tech_lim:.1%})")
            st.divider()

        max_w = st.slider("Max Weight (Individual)", 0.05, 1.0, 0.25, 0.05)
        
    with col_univ:
        if "user_tickers" not in st.session_state: 
            st.session_state["user_tickers"] = pd.DataFrame(DEFAULT_TICKERS)
        
        display_df = st.session_state["user_tickers"].copy()
        
        if use_sector_limits and "sector_targets" in st.session_state:
            targets = st.session_state["sector_targets"]
            display_df["Macro Cap"] = display_df["Sector"].map(targets)
        
        column_cfg = {
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "Sector": st.column_config.TextColumn("Sector", width="medium"),
            "Macro Cap": st.column_config.NumberColumn(
                "Macro Cap", 
                format="%.2f%%", 
                disabled=True
            )
        }
        
        edited = st.data_editor(
            display_df, 
            column_config=column_cfg,
            num_rows="dynamic", 
            use_container_width=True
        )
        
        if st.button("🚀 Optimize Portfolio", type="primary"):
            with st.spinner("Fetching data..."):
                t_list = edited["Ticker"].tolist()
                s_map = dict(zip(edited["Ticker"], edited["Sector"]))
                df_mkt, hist, cov = process_bulk_data(t_list, s_map, mode_sel)
                
                if df_mkt is not None and not df_mkt.empty:
                    limits = st.session_state["sector_targets"] if use_sector_limits else None
                    df_res = optimize_portfolio(df_mkt, obj, max_w, mode_sel, sector_limits=limits, cov_matrix=cov)
                    
                    st.session_state["opt_res"] = df_res
                    st.session_state["mkt_data"] = df_mkt
                    st.session_state["hist_data"] = hist
                else:
                    st.error("Data fetch failed or returned empty universe.")

    if "opt_res" in st.session_state and st.session_state["opt_res"] is not None and not st.session_state["opt_res"].empty:
        df_res = st.session_state["opt_res"]
        df_mkt = st.session_state["mkt_data"]
        hist = st.session_state["hist_data"]
        metric_col = "PEG" if "P/E/G" in mode_sel else "PE"

        st.divider()
        st.subheader("1. Asset Selection Matrix")
        # Passing original dataframe to avoid contamination
        fig_quad = plot_quadrant_chart(df_res, metric_col, "RSI", weight_col="Weight")
        st.plotly_chart(fig_quad, use_container_width=True)

        st.subheader("2. Optimal Allocation")
        col_tbl, col_tv = st.columns([2, 1])
        with col_tbl:
            disp = df_res[df_res["Weight"] > 0.001].sort_values("Weight", ascending=False).copy()
            
            tot_w = disp["Weight"].sum()
            w_rsi = (disp["Weight"]*disp["RSI"]).sum()/tot_w if tot_w > 0 else 0
            w_vol = (disp["Weight"]*disp["Volatility"]).sum()/tot_w if tot_w > 0 else 0
            w_val_inv = (disp["Weight"]/disp[metric_col]).sum()
            w_val = tot_w/w_val_inv if w_val_inv > 0 else 0
            
            sum_row = pd.DataFrame([{"Ticker": "TOTAL", "Weight": tot_w, "RSI": w_rsi, "Volatility": w_vol, metric_col: w_val}])
            final = pd.concat([disp, sum_row], ignore_index=True)
            
            st.dataframe(
                final.style.format({
                    "Weight": "{:.2%}", 
                    "RSI": "{:.1f}", 
                    "Volatility": "{:.2%}", 
                    metric_col: "{:.2f}"
                }), 
                use_container_width=True
            )
            
            with st.expander("📊 Strategy Breakdown: Allocation Methodology"):
                st.markdown("""
                This model employs a multi-factor approach, optimizing for **Earnings Yield** (Value) and **Relative Strength** (Momentum).
                
                * **Weighting:** The optimal capital allocation coefficient derived from the linear optimization solver.
                * **RSI (Momentum Factor):** The portfolio RSI is the **Weighted Arithmetic Mean** of individual constituents.
                * **P/E (Value Factor):** We utilize the **Weighted Harmonic Mean** rather than a simple arithmetic average.
                
                **Why Harmonic Mean?** Averaging valuation ratios (like P/E) directly using an arithmetic mean creates a mathematical bias that overstates the "expensiveness" of the portfolio. The Harmonic Mean correctly averages the underlying "Earnings Yields" (E/P), providing a true reflection of the portfolio's aggregate valuation.
                """)

        with col_tv:
            with st.expander("📤 TradingView Export"):
                st.write("Copy/Paste to Pine Editor:")
                lines = ["//@version=5", "indicator('Portfolio', overlay=true)", f"var table t = table.new(position.top_right, 2, {len(disp)+1})"]
                for idx, (i, r) in enumerate(disp.iterrows()):
                    lines.append(f"table.cell(t, 0, {idx}, '{r['Ticker']}')")
                    lines.append(f"table.cell(t, 1, {idx}, '{r['Weight']*100:.2f}%')")
                st.code("\n".join(lines))

        st.divider()
        st.subheader("3. Backtest Analysis")
        if SKFOLIO_AVAILABLE and hist:
            try:
                p_df = pd.DataFrame({t: d['Close'] for t, d in hist.items()}).dropna()
                X = prices_to_returns(p_df)
                
                sel_ticks = disp["Ticker"].tolist()
                sel_ws = disp["Weight"].tolist()
                X_strat = X[sel_ticks]
                
                strat = Portfolio(X=X_strat, weights=sel_ws, name="Optimized")
                bench = Portfolio(X=X, weights=[1/len(X.columns)]*len(X.columns), name="Equal Weight Universe")
                pop = Population([strat, bench])
                
                c1, c2 = st.columns(2)
                with c1: st.plotly_chart(pop.plot_cumulative_returns(), use_container_width=True)
                with c2: st.dataframe(pop.summary().astype(str), use_container_width=True)
            except Exception as e:
                st.error(f"Backtest Error: {e}")

        st.divider()
        with st.expander("ℹ️ How the Optimization Logic Works"):
            st.markdown(r"""
            ### 1. The Scoring Formula
            The optimizer assigns a **"Growth-Momentum Score"** to every asset:
            
            $$
            \text{Score} = \underbrace{\left( \frac{\text{RSI}}{100} \right)}_{\text{Momentum}} + \underbrace{\left( \frac{1}{\text{PEG Ratio}} \right)}_{\text{Value}}
            $$

            ### 2. Linear vs. Quadratic Optimization 
            
            * **Linear Programming (Factor Exposure):** This tool utilizes Linear Programming (GLOP solver) to maximize direct factor exposure. Instead of minimizing variance through correlation, we mitigate risk via **concentration constraints** (hard limits on max allocation). This allows for a computationally efficient ($O(n)$) maximization of the 'Growth + Momentum' alpha score without the instability often introduced by covariance estimation errors in small samples.
            * **Quadratic Programming (MPT):** Modern Portfolio Theory typically employs Quadratic Programming to minimize portfolio variance ($\sigma^2$). This requires calculating the full covariance matrix $\Sigma$ to account for pairwise asset correlations ($O(n^2)$ complexity). It optimizes for the lowest risk at a given return level.
            """)

# --- MAIN CONTROLLER ---
def run_app():
    st.sidebar.title("MomentumValue")
    app_mode = st.sidebar.radio("Select Step:", ["Step 1: Rebalance Technology", "Step 2: Optimize"])
    st.sidebar.divider()
    
    if app_mode == "Step 1: Rebalance Technology":
        run_sector_rebalancer()
    else:
        run_stock_optimizer()

if __name__ == "__main__":
    try:
        run_app()
    except Exception as e:
        st.error(f"Application Error: {e}")