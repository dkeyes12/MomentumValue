import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Import functions from your main script
from streamlit_app import calculate_rsi, optimize_portfolio, process_bulk_data

# --- FIXTURES (Sample Data) ---

@pytest.fixture
def sample_price_data():
    """Generates 50 days of dummy price data."""
    dates = pd.date_range(start="2023-01-01", periods=50)
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

def test_calculate_rsi(sample_price_data):
    rsi = calculate_rsi(sample_price_data['Close'])
    assert len(rsi) == 50
    assert rsi.iloc[-1] > 90
    assert pd.isna(rsi.iloc[0])

def test_process_bulk_data_peg_mode(mock_yf_download, mock_ticker_info):
    tickers = ["AAPL"]
    sector_map = {"AAPL": "Tech"}
    
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

def test_optimization_logic():
    data = {
        "Ticker": ["A", "B"],
        "Sector": ["Tech", "Energy"],
        "RSI": [70, 30],
        "PEG": [1.0, 3.0],
        "Volatility": [0.2, 0.2]
    }
    df = pd.DataFrame(data)
    
    # --- FIX: Max Weight Constraint ---
    # We set max_weight to 0.6. 
    # Asset A is better, so it gets 0.6 (60%).
    # Asset B gets the remainder 0.4 (40%).
    # This ensures BOTH assets appear in the result, preventing IndexError.
    result = optimize_portfolio(df, "Maximize Gain (Score)", 0.6, "Popular and widely followed stocks (P/E/G)")
    
    assert not result.empty
    # Now both A and B should exist in the dataframe
    weight_A = result.loc[result['Ticker'] == "A", "Weight"].values[0]
    weight_B = result.loc[result['Ticker'] == "B", "Weight"].values[0]
    
    # Check logic: Better asset gets more weight
    assert weight_A > weight_B
    # Check Math: Sums to 1.0
    assert abs(result['Weight'].sum() - 1.0) < 0.001

def test_optimization_empty_input():
    # --- FIX: Empty Input Test ---
    # We pass an empty DF. The updated function should return empty DF immediately
    # without crashing on column access.
    df_empty = pd.DataFrame()
    result = optimize_portfolio(df_empty, "Maximize Gain (Score)", 0.5, "Popular and widely followed stocks (P/E/G)")
    assert result.empty