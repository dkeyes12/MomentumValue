# Copilot / AI Agent Instructions for MomentumValue ✅

Short summary
- Lightweight Streamlit app that builds an interactive stock "momentum" dashboard (P/E vs RSI + price charts).
- Single main source file: `app.py`. Dependencies in `requirements.txt` (streamlit, yfinance, pandas, plotly).

Big-picture architecture & data flow 🔧
- UI layer: Streamlit (`st.sidebar`, `st.metric`, `st.plotly_chart`) provides inputs and live reloading.
- Data layer: `yfinance.Ticker` history and `.info` are used to fetch OHLCV + metadata (P/E). Code expects standard OHLCV columns (`Open, High, Low, Close, Volume`).
- Compute layer: In-memory Pandas DataFrame; indicators are added in `get_stock_data()` (SMA_50, SMA_200, `RSI`).
- Presentation layer: Plotly `make_subplots` / `go.Figure` for price & RSI charts + a quadrant scatter for Value vs Momentum.

Critical functions & conventions (search these names) 🔎
- `calculate_rsi(data_series, window=14)` — uses `diff()` + `ewm()` smoothing; RSI column name is `RSI`.
- `get_stock_data(ticker, period, source_column)` — fetches data, computes `SMA_50`, `SMA_200`, and `RSI` (from selectable source: `Close`, `Open`, `High`, `Low`). Returns `(df, info)` or `(None, None)` on failure.
- `determine_signal(price, sma50, sma200, rsi)` — returns `("BUY"|"SELL / SHORT"|"WAIT", color)`; decision logic relies on SMA50 > SMA200 and RSI > 50.
- Column naming: `SMA_50`, `SMA_200`, `RSI` — keep names consistent when extending indicators.

Developer workflows (commands you will use frequently) ▶️
- Install deps: `pip install -r requirements.txt`
- Run locally: `streamlit run app.py`  (note: repository comment references `dashboard.py` — the correct entry is `app.py`)
- Typical debug flow: change code, save, and Streamlit auto-reloads. Use `st.warning`, `st.error`, and `st.spinner` used by the app for user feedback.

Project-specific patterns & gotchas ⚠️
- Data availability: `yfinance` may return empty DataFrames or missing fields in `.info`. The app uses fallbacks (e.g., `trailingPE` then `forwardPE`) and shows warnings when data not present.
- RSI source is user-selected via sidebar (`rsi_source`). If adding a new source (e.g., VWAP), also update the sidebar options and compute VWAP in `get_stock_data()` before calculating RSI.
- VWAP note in code: the app does not compute VWAP by default; comments mention it requires per-timeframe calculation.
- Error handling: wrap network calls and heavy computations so Streamlit can surface errors (`st.error`) instead of crashing the session.

How to add a new indicator (example) 💡
- Add calculation inside `get_stock_data()` and assign a descriptive column (e.g., `df['VWAP'] = ...`).
- Update UI label and any metrics or plot traces to use the new column.
- Minimal example (within `get_stock_data()`):

```python
# simple per-row typical price VWAP-ish (note: for production compute per-interval correctly)
df['typical'] = (df['High'] + df['Low'] + df['Close']) / 3
df['VWAP'] = (df['typical'] * df['Volume']).cumsum() / df['Volume'].cumsum()
```

Testing & PR guidance ✅
- There are no unit tests in the repo currently. For changes that affect calculations, add tests around `calculate_rsi()` and `determine_signal()`.
- Keep UI changes self-contained; ensure metric labels reflect any change (e.g., `RSI (14) on {rsi_source}`).
- For data-dependent features, include a mock yfinance object or a saved CSV snapshot for reproducible tests.

Integration points & external behavior 📡
- yfinance: network calls can be slow and flaky — consider caching or retrying for heavy loads.
- Plotly + Streamlit: charts are synchronous and re-render on data updates; page config uses `layout='wide'` for dashboard-style UI.

What the agent should NOT do automatically 🚫
- Do not change the main file name or entrypoint without confirming (the app is `app.py`).
- Avoid broad refactors that alter column names or UI text without updating all dependent code and tests.

Where to look first when editing
- `app.py` — the whole app is in one file; start here to understand the flows and add small features.
- `requirements.txt` — confirm required package versions when adding libraries.

If anything is unclear or you'd like more detail (tests, sample fixtures, CI steps), tell me which area to expand and I will iterate. 🙋‍♀️
