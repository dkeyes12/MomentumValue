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
def main():
    # Only runs when called directly
    st.set_page_config(page_title="Portfolio Optimizer [MomentumValue + Backtest]", layout="wide")
    try:
        run_app()
    except Exception as e:
        st.error(f"An error occurred: {e}")

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
    st.cache_data.clear()
    keys_to_drop = ["market_data", "opt_results", "historical_data"]
    for k in keys_to_drop:
        if k in st.session_state:
            del st.session_state[k]

# --- HELPER FUNCTIONS ---
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def generate_pinescript(df_results):
    """Generates TradingView Pine Script to display allocation table."""
    script = [
        "//@version=5",
        "indicator('MomentumValue Portfolio Allocation', overlay=true)",
        "var table tbl = table.new(position.top_right, 2, " + str(len(df_results) + 2) + ", border_width=1)",
        "if barstate.islast",
        "    table.cell(tbl, 0, 0, 'Asset', bgcolor=color.new(color.blue, 30), text_color=color.white)",
        "    table.cell(tbl, 1, 0, 'Weight', bgcolor=color.new(color.blue, 30), text_color=color.white)"
    ]
    
    # Sort for Pine Script display as well
    df_sorted = df_results.sort_values(by="Weight", ascending=False)
    
    for i, row in df_sorted.iterrows():
        ticker = row['Ticker']
        weight = f"{row['Weight']*100:.1f}%"
        # row_idx logic for table positioning
        script.append(f"    table.cell(tbl, 0, {i+1}, '{ticker}', bgcolor=color.new(color.gray, 90), text_color=color.black)")
        script.append(f"    table.cell(tbl, 1, {i+1}, '{weight}', bgcolor=color.new(color.gray, 90), text_color=color.black)")
        
    return "\n".join(script)

@st.cache_data
def process_bulk_data(tickers, sector_map, mode, period="5y"):
    ticker_list = [t.upper().strip() for t in tickers if t.strip()]
    if not ticker_list: return None, None

    # Bulk download
    bulk_data = yf.download(ticker_list, period=period, group_by='ticker', auto_adjust=False)
    
    snapshot_data = []
    hist_data = {}

    try:
        my_bar = st.progress(0, text="Fetching valuation metrics...")
    except:
        my_bar = None

    total_tickers = len(ticker_list)

    for idx, t in enumerate(ticker_list):
        if my_bar:
            my_bar.progress((idx + 1) / total_tickers, text=f"Fetching data for {t}...")
        
        try:
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

            current_price = df['Close'].iloc[-1]
            start_price = df['Close'].iloc[0]
            pct_return = (current_price - start_price) / start_price if start_price > 0 else 0
            
            rsi = df['RSI'].iloc[-1]
            volatility = df['Close'].pct_change().std() * np.sqrt(252)
            
            # --- VALUATION FETCHING ---
            val_metric = None
            try:
                stock = yf.Ticker(t)
                stock_info = stock.info
                
                if mode == "Popular and widely followed stocks (P/E/G)":
                    val_metric = stock_info.get('pegRatio')
                    if val_metric is None:
                        val_metric = stock_info.get('trailingPE')
                        if val_metric is None:
                            val_metric = stock_info.get('forwardPE')
                else: 
                    val_metric = stock_info.get('trailingPE')
                    if val_metric is None: val_metric = stock_info.get('forwardPE')
                
                time.sleep(0.1)
            except Exception:
                val_metric = None
            
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

        except Exception:
            continue
    
    if my_bar:
        my_bar.empty()
            
    return pd.DataFrame(snapshot_data), hist_data

def optimize_portfolio(df, objective_type, max_weight_per_asset, mode):
    if df is None or df.empty: return pd.DataFrame()
    
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver: return None

    weights = []
    for i in range(len(df)):
        weights.append(solver.NumVar(0.0, max_weight_per_asset, f'w_{i}'))

    constraint_sum = solver.Constraint(1.0, 1.0)
    for w in weights:
        constraint_sum.SetCoefficient(w, 1)

    objective = solver.Objective()
    
    if mode == "Popular and widely followed stocks (P/E/G)":
        scores = (df['RSI'] / 100) + (1 / df['PEG']) 
    else:
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

def run_app():
    if "market_data" not in st.session_state: st.session_state["market_data"] = None
    if "opt_results" not in st.session_state: st.session_state["opt_results"] = None
    if "historical_data" not in st.session_state: st.session_state["historical_data"] = {}

    st.title("⚖️ Portfolio Optimizer & Backtester")
    
    if not SKFOLIO_AVAILABLE:
        st.error("⚠️ `skfolio` library not found. Please install it using `pip install skfolio` to enable backtesting features.")

    with st.sidebar:
        st.header("1. Settings")
        mode_select = st.radio(
            "Universe:", 
            ["S&P 500 Sectors (P/E)", "Popular and widely followed stocks (P/E/G)"], 
            on_change=clear_cache_callback
        )
        period_select = st.selectbox("Lookback Period:", ["1y", "2y", "3y", "5y"], index=3)
        st.divider()
        st.header("2. Optimization")
        obj_choice = st.radio("Goal", ["Maximize Gain (Score)", "Minimize Loss (Volatility)"])
        max_concentration = st.slider("Max Allocation", 0.05, 1.0, 0.25, 0.05)
        st.divider()
        if st.button("⚠️ Force Clear Cache"):
            clear_cache_callback()
            st.rerun()

    tab_opt, tab_backtest = st.tabs(["⚙️ Optimization & Analysis", "📈 Backtesting (skfolio)"])

    # === TAB 1: OPTIMIZATION ===
    with tab_opt:
        st.subheader(f"1. Define Universe: {mode_select}")
        
        if "last_mode" not in st.session_state or st.session_state["last_mode"] != mode_select:
            new_defaults = STOCK_TICKERS if mode_select == "Popular and widely followed stocks (P/E/G)" else ETF_TICKERS
            st.session_state["user_tickers"] = pd.DataFrame(new_defaults)
            st.session_state["last_mode"] = mode_select

        col_input, col_action = st.columns([3, 1])
        with col_input:
            edited_df = st.data_editor(
                st.session_state["user_tickers"], 
                column_config={"Ticker": st.column_config.TextColumn("Ticker", width="small")},
                num_rows="dynamic", use_container_width=True, key=f"editor_{mode_select}"
            )
        with col_action:
            st.write("### ")
            if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
                tickers = edited_df["Ticker"].tolist()
                sector_map = dict(zip(edited_df['Ticker'], edited_df['Sector']))
                if len(tickers) < 2:
                    st.error("Need 2+ tickers.")
                else:
                    with st.spinner(f"Fetching {period_select} data..."):
                        df_mkt, hist_data = process_bulk_data(tickers, sector_map, mode_select, period=period_select)
                        if df_mkt is not None and not df_mkt.empty:
                            df_res = optimize_portfolio(df_mkt, obj_choice, max_concentration, mode_select)
                            st.session_state["market_data"] = df_mkt
                            st.session_state["historical_data"] = hist_data
                            st.session_state["opt_results"] = df_res
                        else:
                            st.error("Fetch failed.")

        if st.session_state["market_data"] is not None:
            df_market = st.session_state["market_data"]
            metric_col = "PEG" if mode_select == "Popular and widely followed stocks (P/E/G)" else "PE"

            if st.session_state["opt_results"] is not None and not st.session_state["opt_results"].empty:
                df_opt = st.session_state["opt_results"]

                # 2. Portfolio Visualization
                st.subheader("2. Portfolio Visualization")
                RSI_THRESHOLD = 50
                data_median = df_market[metric_col].median()
                
                if data_median > 5.0:
                    VAL_THRESHOLD = 25; MAX_X = max(60, df_market[metric_col].max() * 1.1)
                    x_label = "P/E Ratio"
                else:
                    VAL_THRESHOLD = 1.5; MAX_X = 4.0
                    x_label = "PEG Ratio"

                fig_quad = go.Figure()
                rem = df_market[~df_market['Ticker'].isin(df_opt['Ticker'])]
                fig_quad.add_trace(go.Scatter(x=rem[metric_col].clip(upper=MAX_X), y=rem['RSI'], mode='markers+text', text=rem['Ticker'], name='Universe', marker=dict(color='gray', size=10)))
                fig_quad.add_trace(go.Scatter(x=df_opt[metric_col].clip(upper=MAX_X), y=df_opt['RSI'], mode='markers+text', text=df_opt['Ticker'], name='Selected', marker=dict(color='blue', size=15)))
                
                fig_quad.add_shape(type="rect", x0=0, y0=50, x1=VAL_THRESHOLD, y1=100, fillcolor="green", opacity=0.1, layer="below", line_width=0)
                fig_quad.add_shape(type="rect", x0=VAL_THRESHOLD, y0=50, x1=MAX_X, y1=100, fillcolor="yellow", opacity=0.1, layer="below", line_width=0)
                fig_quad.add_shape(type="rect", x0=0, y0=0, x1=VAL_THRESHOLD, y1=50, fillcolor="yellow", opacity=0.1, layer="below", line_width=0)
                fig_quad.add_shape(type="rect", x0=VAL_THRESHOLD, y0=0, x1=MAX_X, y1=50, fillcolor="red", opacity=0.1, layer="below", line_width=0)
                
                fig_quad.add_annotation(x=VAL_THRESHOLD/2, y=90, text="VALUE + MOMENTUM", showarrow=False, font=dict(color="green", size=14, weight="bold"))
                fig_quad.add_annotation(x=VAL_THRESHOLD + (MAX_X-VAL_THRESHOLD)/2, y=90, text="EXPENSIVE MOMENTUM", showarrow=False, font=dict(color="orange", size=10))
                fig_quad.add_annotation(x=VAL_THRESHOLD/2, y=10, text="WEAK / VALUE TRAP", showarrow=False, font=dict(color="orange", size=10))
                fig_quad.add_annotation(x=VAL_THRESHOLD + (MAX_X-VAL_THRESHOLD)/2, y=10, text="EXPENSIVE & WEAK", showarrow=False, font=dict(color="red", size=14, weight="bold"))

                fig_quad.update_layout(title="Asset Selection Matrix", xaxis_title=x_label, yaxis_title="RSI (Momentum)", height=500)
                st.plotly_chart(fig_quad, use_container_width=True)

                # 3. Optimal Allocation
                st.divider()
                st.subheader("3. Optimal Allocation")
                
                with st.expander("📊 Strategy Breakdown: Allocation Methodology"):
                    st.markdown("""
                    This model employs a multi-factor approach, optimizing for **Earnings Yield** (Value) and **Relative Strength** (Momentum).
                    
                    * **Weighting:** The optimal capital allocation coefficient derived from the linear optimization solver.
                    * **RSI (Momentum Factor):** The portfolio RSI is the **Weighted Arithmetic Mean** of individual constituents.
                    * **P/E (Value Factor):** We utilize the **Weighted Harmonic Mean** rather than a simple arithmetic average.
                    
                    

                    **Why Harmonic Mean?** Averaging valuation ratios (like P/E) directly using an arithmetic mean creates a mathematical bias that overstates the "expensiveness" of the portfolio. The Harmonic Mean correctly averages the underlying "Earnings Yields" (E/P), providing a true reflection of the portfolio's aggregate valuation.
                    """)

                col_table, col_export = st.columns([2, 1])
                
                with col_table:
                    # Sort results by weight
                    disp_df = df_opt.copy().sort_values(by="Weight", ascending=False)
                    
                    # --- CALCULATE TOTALS ---
                    total_weight = disp_df['Weight'].sum()
                    
                    # Weighted Avg RSI
                    w_rsi = (disp_df['Weight'] * disp_df['RSI']).sum() / total_weight if total_weight > 0 else 0
                    
                    # Harmonic Mean for Valuation (P/E or PEG)
                    # Avoid division by zero
                    if (disp_df[metric_col] == 0).any():
                        w_val = np.nan
                    else:
                        w_val_inv = (disp_df['Weight'] / disp_df[metric_col]).sum()
                        w_val = total_weight / w_val_inv if w_val_inv > 0 else 0
                    
                    # Weighted Avg Volatility
                    w_vol = (disp_df['Weight'] * disp_df['Volatility']).sum() / total_weight if total_weight > 0 else 0
                    
                    summary = pd.DataFrame([{
                        "Ticker": "TOTAL", 
                        "Sector": "-", 
                        "Weight": total_weight, 
                        metric_col: w_val, 
                        "RSI": w_rsi,
                        "Volatility": w_vol
                    }])
                    
                    final_df = pd.concat([disp_df, summary], ignore_index=True)
                    
                    # Formatting
                    final_df["Weight"] = final_df["Weight"].apply(lambda x: f"{x:.1%}")
                    final_df[metric_col] = final_df[metric_col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
                    final_df["RSI"] = final_df["RSI"].apply(lambda x: f"{x:.1f}")
                    final_df["Volatility"] = final_df["Volatility"].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
                    
                    st.dataframe(final_df, use_container_width=True)
                
                # --- EXPORT TO TRADINGVIEW ---
                with col_export:
                    with st.expander("📤 Export to TradingView"):
                        st.markdown("**1. Pine Script (Charts):** Copy and paste into Pine Editor.")
                        pine_code = generate_pinescript(df_opt) 
                        st.code(pine_code, language="pinescript")
                        
                        st.divider()
                        st.markdown("**2. JSON (Bots):** For webhook alerts.")
                        df_json = df_opt.sort_values(by="Weight", ascending=False)
                        json_data = df_json[["Ticker", "Weight"]].to_json(orient="records")
                        st.code(json_data, language="json")

                # 4. Technical Analysis
                st.subheader("4. Technical Analysis")
                if st.session_state["historical_data"]:
                    sel_t = st.selectbox("View Chart:", list(st.session_state["historical_data"].keys()))
                    df_c = st.session_state["historical_data"][sel_t]
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                    fig.add_trace(go.Candlestick(x=df_c.index, open=df_c['Open'], high=df_c['High'], low=df_c['Low'], close=df_c['Close'], name='OHLC'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_c.index, y=df_c['SMA_50'], line=dict(color='orange'), name='SMA 50'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_c.index, y=df_c['SMA_200'], line=dict(color='blue', width=2), name='SMA 200'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_c.index, y=df_c['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
                    fig.add_hline(y=70, row=2, col=1, line_dash="dot", line_color="red")
                    fig.add_hline(y=30, row=2, col=1, line_dash="dot", line_color="green")
                    fig.update_layout(height=500, xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

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
            
            $$
            \text{Score} = \underbrace{\left( \frac{\text{RSI}}{100} \right)}_{\text{Momentum}} + \underbrace{\left( \frac{1}{\text{PEG Ratio}} \right)}_{\text{Value}}
            $$

            ### 2. Linear vs. Quadratic Optimization 
            
            

            * **Linear Programming (Factor Exposure):** This tool utilizes Linear Programming (GLOP solver) to maximize direct factor exposure. Instead of minimizing variance through correlation, we mitigate risk via **concentration constraints** (hard limits on max allocation). This allows for a computationally efficient ($O(n)$) maximization of the 'Growth + Momentum' alpha score without the instability often introduced by covariance estimation errors in small samples.
            * **Quadratic Programming (MPT):** Modern Portfolio Theory typically employs Quadratic Programming to minimize portfolio variance ($\sigma^2$). This requires calculating the full covariance matrix $\Sigma$ to account for pairwise asset correlations ($O(n^2)$ complexity). It optimizes for the lowest risk at a given return level.
            """)

    # === TAB 2: BACKTESTING ===
    with tab_backtest:
        st.header("📈 Historical Backtest (skfolio)")
        if not SKFOLIO_AVAILABLE:
            st.warning("Please install skfolio to see backtesting results.")
        elif st.session_state["opt_results"] is None:
            st.info("Please run the optimization in the first tab to generate a portfolio to backtest.")
        else:
            df_opt = st.session_state["opt_results"]
            hist_data = st.session_state["historical_data"]
            price_dict = {t: data['Close'] for t, data in hist_data.items()}
            price_df = pd.DataFrame(price_dict).dropna()
            X = prices_to_returns(price_df)
            
            # --- ISOLATE SELECTED ASSETS FOR STRATEGY ---
            selected_tickers = df_opt['Ticker'].tolist()
            selected_weights = df_opt['Weight'].tolist()
            X_strategy = X[selected_tickers]
            
            strategy_portfolio = Portfolio(X=X_strategy, weights=selected_weights, name="Optimized Strategy")
            
            # Benchmark uses FULL universe
            n_assets = len(X.columns)
            benchmark_portfolio = Portfolio(X=X, weights=[1.0/n_assets]*n_assets, name="Equal Weighted Benchmark")
            
            pop = Population([strategy_portfolio, benchmark_portfolio])
            st.markdown("Comparing your optimized portfolio against an equal-weighted benchmark.")
            
            st.subheader("Cumulative Returns")
            try:
                fig_cum = pop.plot_cumulative_returns()
                st.plotly_chart(fig_cum, use_container_width=True)
            except Exception as e:
                st.error(f"Could not plot cumulative returns: {e}")
            
            st.subheader("Risk & Return Metrics")
            try:
                summary_df = pop.summary()
                st.dataframe(summary_df.astype(str), use_container_width=True)
            except Exception as e:
                st.error(f"Could not generate summary: {e}")
            
            st.subheader("Portfolio Composition")
            try:
                fig_comp = strategy_portfolio.plot_composition()
                st.plotly_chart(fig_comp, use_container_width=True)
            except Exception as e:
                st.error(f"Could not plot composition: {e}")

if __name__ == "__main__":
    main()