# MomentumValue

A lightweight Streamlit dashboard that visualizes stock value vs momentum (P/E vs RSI) and price/RSI charts.

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
streamlit run app.py
# or
python -m streamlit run app.py
```

Open the URL Streamlit prints in your browser (usually http://localhost:8501).

---

## Project overview 🔧

- Entry point: `app.py` — the entire UI and logic live in a single file.
- Dependencies: `streamlit`, `yfinance`, `pandas`, `plotly` (see `requirements.txt`).
- Data flow: Streamlit UI → `yfinance.Ticker.history()` & `.info` → Pandas DataFrame → indicator calculations (`SMA_50`, `SMA_200`, `RSI`) → Plotly visualizations.

---

## Important notes & gotchas ⚠️

- yfinance can return empty DataFrames or omit fields in `.info` (e.g., `trailingPE`). The app falls back to `forwardPE` when `trailingPE` is missing.
- `RSI` is computed using the user-selected source (`Close`, `Open`, `High`, `Low`) from the sidebar. If you add new sources (e.g., VWAP), compute and add the column inside `get_stock_data()` before calculating RSI.
- The repo comment mentions `dashboard.py` but the actual file is `app.py` — do not rename the entrypoint without consensus.

---

## Development tips ✍️

- Streamlit auto-reloads on save — make small iterative edits to `app.py` and watch the UI refresh.
- Use `st.warning`, `st.error`, and `st.spinner` when adding new network-dependent features so failures surface gracefully.

### Adding indicators

Add calculations inside `get_stock_data()` and assign descriptive column names (e.g., `VWAP`, `SMA_20`). Example (not production-accurate VWAP):

```python
# inside get_stock_data(df):
df['typical'] = (df['High'] + df['Low'] + df['Close']) / 3
df['VWAP'] = (df['typical'] * df['Volume']).cumsum() / df['Volume'].cumsum()
```

---

## Testing & CI suggestions (no tests currently) 🧪

- There are no unit tests yet. Suggested first targets: `calculate_rsi()` and `determine_signal()`.
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
