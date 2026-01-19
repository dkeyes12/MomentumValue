import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Import functions from your main script
# Ensure your main script is named 'streamlit_app.py' or adjust this import
from streamlit_app import calculate_rsi, optimize_portfolio, process_bulk_data

# --- FIXTURES (Sample Data Setup) ---

@pytest.fixture
def sample_price_data():
    """Generates 50 days of dummy price data (Uptrend)."""
    dates = pd.date_range(start="2023-01-01", periods=50)
    prices = np.arange(100, 150) # Prices going from 100 to 149
    df = pd.DataFrame({
        "Close": prices, 
        "Open": prices, 
        "High": prices+1, 
        "Low": prices-1,
        "Volume": 1000
    }, index=dates)
    return df

@pytest.fixture
def mock_yf_download(sample_price_data):
    """Mocks yf.download to return our sample data instantly."""
    with patch("yfinance.download") as mock_download:
        mock_download.return_value = sample_price_data
        yield mock_download

@pytest.fixture
def mock_ticker_info():
    """Mocks yf.Ticker to return specific valuation metrics."""
    with patch("yfinance.Ticker") as mock_ticker:
        instance = mock_ticker.return_value
        # Default info: PEG is good (1.5)
        instance.info = {"pegRatio": 1.5, "trailingPE": 20, "forwardPE": 18}
        yield mock_ticker

# --- UNIT TESTS ---

def test_calculate_rsi(sample_price_data):
    """Test RSI calculation logic."""
    rsi = calculate_rsi(sample_price_data['Close'])
    
    assert len(rsi) == 50
    # First 14 values should be NaN (lookback period)
    assert pd.isna(rsi.iloc[0])
    # Last value should be > 90 because sample data is a straight uptrend
    assert rsi.iloc[-1] > 90

def test_process_bulk_data_peg_mode(mock_yf_download, mock_ticker_info):
    """Test data fetching in Stock Mode (expects PEG)."""
    tickers = ["AAPL"]
    sector_map = {"AAPL": "Tech"}
    
    # We patch streamlit.progress to avoid UI errors during test
    with patch("streamlit.progress"):
        df_snapshot, hist_data = process_bulk_data(
            tickers, 
            sector_map, 
            mode="Popular and widely followed stocks (P/E/G)", 
            period="1y"
        )
    
    # Validations
    assert not df_snapshot.empty
    assert "PEG" in df_snapshot.columns
    # Check if it grabbed the mocked PEG (1.5)
    assert df_snapshot.iloc[0]["PEG"] == 1.5
    # PE column should be NaN in PEG mode (logic separation)
    assert np.isnan(df_snapshot.iloc[0]["PE"])
    # Check if history was stored
    assert "AAPL" in hist_data

def test_process_bulk_data_pe_fallback(mock_yf_download):
    """Test Fallback Logic: If PEG is missing, use Trailing PE."""
    tickers = ["MSFT"]
    sector_map = {"MSFT": "Tech"}
    
    with patch("yfinance.Ticker") as mock_ticker:
        instance = mock_ticker.return_value
        # Mock missing PEG, but valid P/E
        instance.info = {"pegRatio": None, "trailingPE": 25}
        
        with patch("streamlit.progress"):
            df_snapshot, _ = process_bulk_data(
                tickers, 
                sector_map, 
                mode="Popular and widely followed stocks (P/E/G)"
            )
            
    # Logic check: Should take 25 (P/E) and put it into the PEG column as fallback
    assert df_snapshot.iloc[0]["PEG"] == 25

def test_optimization_logic():
    """Test Linear Programming Logic."""
    # Setup Data: Asset A is "Better" (High RSI, Low Valuation)
    data = {
        "Ticker": ["A", "B"],
        "Sector": ["Tech", "Energy"],
        "RSI": [80, 30],       # A has better momentum
        "PEG": [1.0, 3.0],     # A has better value (lower is better)
        "PE": [np.nan, np.nan],
        "Volatility": [0.1, 0.1]
    }
    df = pd.DataFrame(data)
    
    # Run Optimizer
    # We use max_weight=0.6. Since A is better, it should get 60%.
    # B should get the remaining 40%.
    result = optimize_portfolio(
        df, 
        objective_type="Maximize Gain (Score)", 
        max_weight_per_asset=0.6, 
        mode="Popular and widely followed stocks (P/E/G)"
    )
    
    assert not result.empty
    assert len(result) == 2 # Ensure both assets are selected
    
    weight_A = result.loc[result['Ticker'] == "A", "Weight"].values[0]
    weight_B = result.loc[result['Ticker'] == "B", "Weight"].values[0]
    
    # Assertions
    assert weight_A > weight_B
    assert abs(weight_A - 0.6) < 0.001
    assert abs(weight_B - 0.4) < 0.001
    assert abs(result['Weight'].sum() - 1.0) < 0.001

def test_optimization_empty_input():
    """Test edge case: Empty DataFrame input."""
    df_empty = pd.DataFrame()
    result = optimize_portfolio(
        df_empty, 
        "Maximize Gain (Score)", 
        0.5, 
        "Popular and widely followed stocks (P/E/G)"
    )
    assert result.empty