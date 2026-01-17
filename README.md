# MomentumValue 📈

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)

A sophisticated multi-page Streamlit application that combines **value investing** principles with **momentum analysis** for intelligent stock selection and portfolio optimization.

## ✨ Features

- **Dual-Mode Portfolio Optimization**: Optimize S&P 500 sector ETFs using P/E ratios or individual stocks using PEG ratios
- **Interactive Stock Analysis**: P/E vs RSI quadrant plots with technical charts and buy/sell signals
- **Linear Programming**: Uses OR-Tools for mathematically optimal portfolio allocation
- **Real-time Data**: Live stock data via Yahoo Finance API
- **Responsive UI**: Clean, intuitive interface built with Streamlit

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dkeyes12/MomentumValue.git
   cd MomentumValue
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Open your browser** to `http://localhost:8501`

## 📊 Application Structure

### 🏠 Main Dashboard (`streamlit_app.py`)
- **ETF Mode**: Optimize S&P 500 sector ETF portfolios using P/E ratios and RSI momentum
- **Stock Mode**: Advanced optimization using PEG (Price/Earnings to Growth) ratios for growth-adjusted valuation
- **Linear Programming**: Maximizes returns while respecting constraints using OR-Tools GLOP solver
- **Interactive Universe**: Editable ticker tables with sector mapping
- **Portfolio Analytics**: Weighted metrics, allocation tables, and technical analysis

### 📈 Stock Selection Assist (`pages/Stock Selection Assist.py`)
- **Individual Stock Analysis**: Deep-dive analysis for any publicly traded stock
- **P/E vs RSI Quadrant**: Visual momentum vs value positioning
- **Technical Charts**: Candlestick charts with moving averages and RSI indicators
- **Trading Signals**: Buy/Sell/Wait recommendations based on trend and momentum
- **Configurable Parameters**: Adjustable RSI source (Close/Open/High/Low) and time periods

## 🛠️ Technical Details

### Dependencies
- `streamlit` - Web app framework
- `yfinance` - Yahoo Finance data API
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `ortools` - Linear programming optimization
- `plotly` - Interactive visualizations

### Data Flow
```
Streamlit UI → Yahoo Finance API → Pandas DataFrame
    ↓
Indicator Calculations (SMA, RSI)
    ↓
Linear Optimization / Visualization
```

### Key Functions
- `calculate_rsi()` - Relative Strength Index computation
- `optimize_portfolio()` - Linear programming portfolio optimization
- `get_stock_data()` - Data fetching and indicator calculation
- `determine_signal()` - Buy/sell/hold signal generation

## ⚠️ Important Notes

- **Data Reliability**: Yahoo Finance data may occasionally be incomplete; the app handles fallbacks gracefully
- **PEG Availability**: PEG ratios are typically only available for individual stocks, not ETFs
- **Cache Management**: Session state caching prevents unnecessary API calls
- **Optimization Constraints**: Ensure sufficient historical data for reliable optimization results

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

## 🧪 Testing

Run the existing test suite:
```bash
pip install -r requirements-dev.txt
pytest test_portfolio_optimizer.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋 Support

If you find this project helpful, please give it a ⭐️!

For questions or issues, please open a GitHub issue.
