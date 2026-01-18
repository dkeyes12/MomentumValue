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
st.set_page_config(page_title="Portfolio Optimizer [MomentumValue + Backtest]", layout="wide")

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

# --- MAIN APP ---
def main():
    if "market_data" not in st.session_state: st.session_state["market_data"] = None
    if "opt_results" not in st.session_state: st.session_state["opt_results"] = None
    if "historical_data" not in st.session_state: st.session_state["historical_data"] = {}

    st.title("⚖️ Portfolio Optimizer & Backtester")
    
    if not SKFOLIO_AVAILABLE:
        st.error("⚠️ `skfolio` library not found. Please install it using `pip install skfolio` to enable backtesting features.")

    # 1. SIDEBAR
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

    # --- TABS FOR UI ---
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
                        # Fixed variable name bug: using sector_map now
                        df_mkt, hist_data = process_bulk_data(tickers, sector_map, mode_select, period=period_select)
                        if df_mkt is not None and not df_mkt.empty:
                            df_res = optimize_portfolio(df_mkt, obj_choice, max_concentration, mode_select)
                            st.session_state["market_data"] = df_mkt
                            st.session_state["historical_data"] = hist_data
                            st.session_state["opt_results"] = df_res
                        else:
                            st.error("Fetch failed.")

        # Display Results if Available
        if st.session_state["market_data"] is not None and st.session_state["opt_results"] is not None:
            df_market = st.session_state["market_data"]
            df_opt = st.session_state["opt_results"]
            metric_col = "PEG" if mode_select == "Popular and widely followed stocks (P/E/G)" else "PE"

            # Plots
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
            
            # Quadrants (Broken into multiple lines to avoid SyntaxError)
            fig_quad.add_shape(
                type="rect", x0=0, y0=50, x1=VAL_THRESHOLD, y1=100, 
                fillcolor="green", opacity=0.1, layer="below", line_width=0
            )
            fig_quad.add_shape(
                type="rect", x0=VAL_THRESHOLD, y0=50, x1=MAX_X, y1=100, 
                fillcolor="yellow", opacity=0.1, layer="below", line_width=0
            )
            fig_quad.add_shape(
                type="rect", x0=0, y0=0, x1=VAL_THRESHOLD, y1=50, 
                fillcolor="yellow", opacity=0.1, layer="below", line_width=0
            )
            fig_quad.add_shape(
                type="rect", x0=VAL_THRESHOLD, y0=0, x1=MAX_X, y1=50, 
                fillcolor="red", opacity=0.1, layer="below", line_width=0
            )
            
            fig_quad.update_layout(title="Asset Selection Matrix", xaxis_title=x_label, yaxis_title="RSI (Momentum)", height=500)
            st.plotly_chart(fig_quad, use_container_width=True)

            # Allocation
            st.subheader("3. Optimal Allocation")
            st.dataframe(df_opt, use_container_width=True)

            # Technicals
            st.subheader("4. Technical Analysis")
            if st.session_state["historical_data"]:
                sel_t = st.selectbox("View Chart:", list(st.session_state["historical_data"].keys()))
                df_c = st.session_state["historical_data"][sel_t]
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(x=df_c.index, open=df_c['Open'], high=df_c['High'], low=df_c['Low'], close=df_c['Close'], name='OHLC'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_c.index, y=df_c['SMA_50'], line=dict(color='orange'), name='SMA 50'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_c.index, y=df_c['SMA_200'], line=dict(color='blue'), name='SMA 200'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_c.index, y=df_c['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
                fig.add_hline(y=70, row=2, col=1, line_dash="dot", line_color="red")
                fig.add_hline(y=30, row=2, col=1, line_dash="dot", line_color="green")
                fig.update_layout(height=500, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

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
            
            # 1. Prepare Data for skfolio
            # Combine all closing prices into a single DataFrame
            price_dict = {t: data['Close'] for t, data in hist_data.items()}
            price_df = pd.DataFrame(price_dict).dropna()
            
            # Convert to Returns
            X = prices_to_returns(price_df)
            
            # 2. Define Weights
            # Map optimized weights to the columns in price_df (order matters!)
            # Create a dictionary of {Ticker: Weight} from optimization results
            weight_map = dict(zip(df_opt['Ticker'], df_opt['Weight']))
            
            # Generate weight list for the 'X' DataFrame columns
            # If a ticker is in X but wasn't selected (0 weight), we give it 0.
            strategy_weights = [weight_map.get(t, 0.0) for t in X.columns]
            
            # 3. Create Portfolios
            # Strategy Portfolio
            strategy_portfolio = Portfolio(
                X=X, 
                weights=strategy_weights, 
                name="Optimized Strategy (Static)"
            )
            
            # Benchmark (Equal Weighted)
            n_assets = len(X.columns)
            benchmark_portfolio = Portfolio(
                X=X,
                weights=[1.0/n_assets] * n_assets,
                name="Equal Weighted Benchmark"
            )
            
            # 4. Analyze Population
            pop = Population([strategy_portfolio, benchmark_portfolio])
            
            # 5. Display Results
            st.markdown("""
            **Note:** This backtest assumes a **Buy & Hold** strategy using the weights optimized today applied to historical data. 
            It compares your selected portfolio against an Equal-Weighted benchmark of the entire universe.
            """)
            
            # Plot Cumulative Returns
            st.subheader("Cumulative Returns")
            # skfolio returns a plotly figure
            fig_cum = pop.plot_cumulative_returns()
            st.plotly_chart(fig_cum, use_container_width=True)
            
            # Summary Statistics
            st.subheader("Risk & Return Metrics")
            summary_df = pop.summary() 
            st.dataframe(summary_df.style.format("{:.2%}"), use_container_width=True)
            
            # Composition Pie Chart 
            st.subheader("Portfolio Composition")
            

[Image of Pie Chart]

            fig_comp = strategy_portfolio.plot_composition()
            st.plotly_chart(fig_comp, use_container_width=True)

if __name__ == "__main__":
    main()