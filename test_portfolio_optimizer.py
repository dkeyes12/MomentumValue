import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Import functions from your main script
# Ensure your main script is named 'streamlit_app.py' or adjust this import
from streamlit_app import calculate_rsi, optimize_portfolio, process_bulk_data

# --- FIXTURES (Sample Data) ---

@pytest.fixture
def sample_price_data():
    """Generates 50 days of dummy price data."""
    dates = pd.date_range(start="2023-01-01", periods=50)
    # create a trend: prices going up from 100 to 149
    prices = np.arange(100, 150)
    df = pd.DataFrame({"Close": prices, "Open": prices, "High": prices+1, "Low": prices-1}, index=dates)
    return df

@pytest.fixture
def mock_yf_download(sample_price_data):
    """Mocks yf.download to return our sample data."""
    with patch("yfinance.download") as mock_download:
        # Simulate the structure returned by yfinance bulk download
        # For single ticker, it returns a DataFrame directly.
        # For multiple, it returns a MultiIndex DF. We'll simulate single ticker behavior for loop.
        mock_download.return_value = sample_price_data
        yield mock_download

@pytest.fixture
def mock_ticker_info():
    """Mocks yf.Ticker to return specific valuation metrics."""
    with patch("yfinance.Ticker") as mock_ticker:
        # Create a mock instance
        instance = mock_ticker.return_value
        # Default info
        instance.info = {"pegRatio": 1.5, "trailingPE": 20, "forwardPE": 18}
        yield mock_ticker

# --- TESTS ---

# 1. Unit Test: RSI Calculation
def test_calculate_rsi(sample_price_data):
    # In a pure uptrend, RSI should be high
    rsi = calculate_rsi(sample_price_data['Close'])
    
    # Check shape
    assert len(rsi) == 50
    # Check logic: Last value should be near 100 for perfect uptrend
    assert rsi.iloc[-1] > 90
    # Check NaN handling: First 14 values should be NaN
    assert pd.isna(rsi.iloc[0])

# 2. Integration Test: Data Processing & Fallback Logic
def test_process_bulk_data_peg_mode(mock_yf_download, mock_ticker_info):
    tickers = ["AAPL"]
    sector_map = {"AAPL": "Tech"}
    
    # Call the function (mocking yf.download via fixture)
    # We mock streamlit progress bar to avoid UI errors during test
    with patch("streamlit.progress") as mock_prog:
        df_snapshot, _ = process_bulk_data(tickers, sector_map, mode="Popular and widely followed stocks (P/E/G)", period="1y")
    
    assert not df_snapshot.empty
    assert "PEG" in df_snapshot.columns
    # Our mock fixture set pegRatio to 1.5
    assert df_snapshot["PEG"].iloc[0] == 1.5
    # PE should be NaN in Stock Mode
    assert np.isnan(df_snapshot["PE"].iloc[0])

def test_process_bulk_data_pe_fallback(mock_yf_download):
    # This test specifically mocks a missing PEG ratio to test fallback
    tickers = ["MSFT"]
    sector_map = {"MSFT": "Tech"}
    
    with patch("yfinance.Ticker") as mock_ticker:
        instance = mock_ticker.return_value
        # PEG is None, Trailing PE is 25
        instance.info = {"pegRatio": None, "trailingPE": 25}
        
        with patch("streamlit.progress"):
            df_snapshot, _ = process_bulk_data(tickers, sector_map, mode="Popular and widely followed stocks (P/E/G)")
            
    # Should use the fallback value (25) as PEG
    assert df_snapshot["PEG"].iloc[0] == 25

# 3. Logic Test: Optimization Engine
def test_optimization_logic():
    # Create a dummy DataFrame representing the Market Snapshot
    # Asset A: Good RSI (70), Good PEG (1.0) -> High Score
    # Asset B: Bad RSI (30), Bad PEG (3.0) -> Low Score
    data = {
        "Ticker": ["A", "B"],
        "Sector": ["Tech", "Energy"],
        "RSI": [70, 30],
        "PEG": [1.0, 3.0],
        "Volatility": [0.2, 0.2]
    }
    df = pd.DataFrame(data)
    
    # Run optimizer targeting "Maximize Gain"
    # Mode="Stock (PEG)" implies formula: Score = RSI/100 + 1/PEG
    result = optimize_portfolio(df, "Maximize Gain