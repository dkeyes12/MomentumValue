# MomentumValue

A multi-page Streamlit application for comprehensive stock analysis and portfolio optimization, combining value investing metrics (P/E and PEG ratios) with momentum indicators (RSI) for both individual stock selection and portfolio construction.

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

- **Main Dashboard** (`streamlit_app.py`): Dual-mode portfolio optimizer supporting both S&P 500 sector ETFs (P/E-based) and individual stock portfolios (PEG-based) using linear programming.
- **Stock Selection Assist** (`pages/Stock Selection Assist.py`): Individual stock momentum analysis with P/E vs RSI quadrant plots and technical charts.

**Dependencies**: `streamlit`, `yfinance`, `pandas`, `numpy`, `ortools`, `plotly` (see `requirements.txt`).

**Data flow**: Streamlit UI → `yfinance.Ticker.history()` & `.info` → Pandas DataFrame → indicator calculations (`SMA_50`, `SMA_200`, `RSI`) → Linear optimization (main app) or visualization (stock assist).

---

## Pages Overview 📄

### 1. Portfolio Optimizer (Main App)
- **Dual Mode Operation**: Switch between "S&P 500 Sectors (P/E)" for sector ETF allocation and "Popular and widely followed stocks (P/E/G)" for individual stock portfolios
- **ETF Mode**: Optimizes S&P 500 sector ETF portfolios using P/E ratios and RSI momentum
- **Stock Mode**: Advanced optimization using PEG (Price/Earnings to Growth) ratios for growth-adjusted valuation
- **Linear Programming**: Uses OR-Tools GLOP solver to maximize gains or minimize volatility
- **Interactive Universe**: Editable table of tickers with sector mapping
- **Portfolio Analytics**: Weighted P/E/PEG and RSI calculations, allocation tables, and technical deep-dives

### 2. Stock Selection Assist
- Analyze individual stocks with interactive P/E vs RSI quadrant analysis
- Technical charts with candlesticks, moving averages, and RSI indicators
- Buy/Sell/Wait signals based on trend and momentum conditions
- Configurable RSI source (Close, Open, High, Low) and time periods

---

## Important notes & gotchas ⚠️

- yfinance can return empty DataFrames or omit fields in `.info` (e.g., `trailingPE`, `pegRatio`). The app handles fallbacks and filters out invalid data.
- `RSI` is computed using the user-selected source in the stock assist page. The portfolio optimizer uses Close price for consistency.
- Portfolio optimization uses OR-Tools linear solver; ensure sufficient data points for reliable results.
- The main entry point is `streamlit_app.py` — Streamlit automatically discovers pages in the `pages/` folder.
- PEG data is typically only available for individual stocks, which is why stock mode uses individual companies rather than ETFs.
- Cache management is implemented to handle mode switching without data conflicts.

---

## Development tips ✍️

- Streamlit auto-reloads on save — make small iterative edits to `streamlit_app.py` or files in `pages/` and watch the UI refresh.
- Use `st.warning`, `st.error`, and `st.spinner` when adding new network-dependent features so failures surface gracefully.
- Session state is used extensively for caching optimization results and historical data.
- The dual-mode architecture requires careful cache management when switching between ETF and stock modes.

### Adding indicators

Add calculations inside `process_bulk_data()` (main app) or `get_stock_data()` (stock assist) and assign descriptive column names (e.g., `VWAP`, `SMA_20`). Example (not production-accurate VWAP):

```python
# inside process_bulk_data() or get_stock_data():
df['typical'] = (df['High'] + df['Low'] + df['Close']) / 3
df['VWAP'] = (df['typical'] * df['Volume']).cumsum() / df['Volume'].cumsum()
```

---

## Testing & CI suggestions (no tests currently) 🧪

- There are no unit tests yet. Suggested first targets: `calculate_rsi()`, `optimize_portfolio()`, and `process_bulk_data()` functions.
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

If you'd like, I can add example unit tests for `calculate_rsi()` and `optimize_portfolio()` and/or a GH Actions workflow to run tests on PRs — tell me which and I'll prepare them. 👩‍💻
