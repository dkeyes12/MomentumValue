# MomentumValue

A multi-page Streamlit application for stock analysis and portfolio optimization, combining value investing (P/E ratios) with momentum indicators (RSI) for individual stocks and sector-based portfolios.

## Quick start ✅

1. Create and activate a Python virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the app:

```powershell
streamlit run streamlit_app.py
# or
python -m streamlit run streamlit_app.py
```

Open the URL Streamlit prints in your browser (usually http://localhost:8501).

---

## Project overview 🔧

This is a multi-page Streamlit app with the following structure:

- **Main Dashboard** (`streamlit_app.py`): Individual stock momentum analysis with P/E vs RSI quadrant plots and technical charts.
- **Portfolio Optimizer** (`pages/PortfolioOptimizer.py`): Sector-based portfolio optimization using linear programming to balance value and momentum across ETFs.

**Dependencies**: `streamlit`, `yfinance`, `pandas`, `numpy`, `ortools`, `plotly`, `matplotlib` (see `requirements.txt`).

**Data flow**: Streamlit UI → `yfinance.Ticker.history()` & `.info` → Pandas DataFrame → indicator calculations (`SMA_50`, `SMA_200`, `RSI`) → Plotly visualizations and optimization.

---

## Pages Overview 📄

### 1. Stock Momentum Dashboard
- Analyze individual stocks with interactive P/E vs RSI quadrant analysis
- Technical charts with candlesticks, moving averages, and RSI indicators
- Buy/Sell/Wait signals based on trend and momentum conditions
- Configurable RSI source (Close, Open, High, Low) and time periods

### 2. Portfolio Optimizer
- Optimize sector ETF portfolios using multi-factor scoring (Value + Momentum)
- Linear programming solver to maximize gains or minimize volatility
- Editable universe of stocks/ETFs with sector mapping
- Portfolio allocation tables with weighted P/E and RSI calculations
- Deep-dive technical analysis for individual holdings

---

## Important notes & gotchas ⚠️

- yfinance can return empty DataFrames or omit fields in `.info` (e.g., `trailingPE`). The app falls back to `forwardPE` when `trailingPE` is missing.
- `RSI` is computed using the user-selected source (`Close`, `Open`, `High`, `Low`) from the sidebar. If you add new sources (e.g., VWAP), compute and add the column inside `get_stock_data()` before calculating RSI.
- Portfolio optimization uses OR-Tools linear solver; ensure sufficient data points for reliable results.
- The main entry point is `streamlit_app.py` — Streamlit automatically discovers pages in the `pages/` folder.

---

## Development tips ✍️

- Streamlit auto-reloads on save — make small iterative edits to `streamlit_app.py` or files in `pages/` and watch the UI refresh.
- Use `st.warning`, `st.error`, and `st.spinner` when adding new network-dependent features so failures surface gracefully.
- Session state is used in the Portfolio Optimizer to persist optimization results across interactions.

### Adding indicators

Add calculations inside `get_stock_data()` (in both files) and assign descriptive column names (e.g., `VWAP`, `SMA_20`). Example (not production-accurate VWAP):

```python
# inside get_stock_data(df):
df['typical'] = (df['High'] + df['Low'] + df['Close']) / 3
df['VWAP'] = (df['typical'] * df['Volume']).cumsum() / df['Volume'].cumsum()
```

---

## Testing & CI suggestions (no tests currently) 🧪

- There are no unit tests yet. Suggested first targets: `calculate_rsi()` and `determine_signal()` in both apps.
- Add `pytest` as a dev dependency and place tests under `tests/`.
- Example test command after adding pytest:

```powershell
pip install pytest
pytest -q
```

---

## Contributing

1. Fork the repo and create a feature branch
2. Add tests for new features that touch computation logic
3. Open a PR with a clear description and screenshots if UI changes

---

If you'd like, I can add example unit tests for `calculate_rsi()` and `determine_signal()` and/or a GH Actions workflow to run tests on PRs — tell me which and I'll prepare them. 👩‍💻
