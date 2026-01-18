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
        mock_download.return_value = sample_price_data
        yield mock_download

@pytest.fixture
def mock_ticker_info():
    """Mocks yf.Ticker to return specific valuation metrics."""
    with patch("yfinance.Ticker") as mock_ticker:
        instance = mock_ticker.return_value
        instance.info = {"pegRatio": 1.5, "trailingPE": 20, "forwardPE": 18}
        yield mock_ticker

# --- TESTS ---

# 1. Unit Test: RSI Calculation
def test_calculate_rsi(sample_price_data):
    rsi = calculate_rsi(sample_price_data['Close'])
    assert len(rsi) == 50
    assert rsi.iloc[-1] > 90
    assert pd.isna(rsi.iloc[0])

# 2. Integration Test: Data Processing & Fallback Logic
def test_process_bulk_data_peg_mode(mock_yf_download, mock_ticker_info):
    tickers = ["AAPL"]
    sector_map = {"AAPL": "Tech"}
    
    # Mock streamlit progress to avoid UI errors
    with patch("streamlit.progress") as mock_prog:
        df_snapshot, _ = process_bulk_data(tickers, sector_map, mode="Popular and widely followed stocks (P/E/G)", period="1y")
    
    assert not df_snapshot.empty
    assert "PEG" in df_snapshot.columns
    assert df_snapshot["PEG"].iloc[0] == 1.5
    assert np.isnan(df_snapshot["PE"].iloc[0])

def test_process_bulk_data_pe_fallback(mock_yf_download):
    tickers = ["MSFT"]
    sector_map = {"MSFT": "Tech"}
    
    with patch("yfinance.Ticker") as mock_ticker:
        instance = mock_ticker.return_value
        # PEG is None, Trailing PE is 25
        instance.info = {"pegRatio": None, "trailingPE": 25}
        
        with patch("streamlit.progress"):
            df_snapshot, _ = process_bulk_data(tickers, sector_map, mode="Popular and widely followed stocks (P/E/G)")
            
    assert df_snapshot["PEG"].iloc[0] == 25

# 3. Logic Test: Optimization Engine
def test_optimization_logic():
    data = {
        "Ticker": ["A", "B"],
        "Sector": ["Tech", "Energy"],
        "RSI": [70, 30],
        "PEG": [1.0, 3.0],
        "Volatility": [0.2, 0.2]
    }
    df = pd.DataFrame(data)
    
    # Use max_weight=0.6 to ensure diverse portfolio so both A and B are selected
    result = optimize_portfolio(df, "Maximize Gain (Score)", 0.6, "Popular and widely followed stocks (P/E/G)")
    
    assert not result.empty
    weight_A = result.loc[result['Ticker'] == "A", "Weight"].values[0]
    weight_B = result.loc[result['Ticker'] == "B", "Weight"].values[0]
    
    assert weight_A > weight_B
    assert abs(result['Weight'].sum() - 1.0) < 0.001

# 4. Edge Case: Empty Data
def test_optimization_empty_input():
    df_empty = pd.DataFrame()
    result = optimize_portfolio(df_empty, "Maximize Gain (Score)", 0.5, "Popular and widely followed stocks (P/E/G)")
    assert result.empty