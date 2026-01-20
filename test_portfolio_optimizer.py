import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Import functions from main app
from streamlit_app import (
    calculate_rsi,
    process_bulk_data,
    optimize_portfolio,
    run_linear_optimization
)

# --- FIXTURES ---

@pytest.fixture
def sample_price_data():
    """Generates synthetic price data for testing RSI."""
    dates = pd.date_range(start="2023-01-01", periods=100)
    prices = [100 + i + (i % 5) for i in range(100)]
    return pd.Series(prices, index=dates)

@pytest.fixture
def mock_market_data():
    """Generates a mock DataFrame representing processed market data."""
    data = {
        "Ticker": ["AAPL", "MSFT", "JPM", "XOM", "PFE"],
        "Sector": [
            "Information Technology", 
            "Information Technology", 
            "Financials", 
            "Energy", 
            "Health Care"
        ],
        "Price": [150.0, 300.0, 140.0, 110.0, 40.0],
        "RSI": [30.0, 70.0, 50.0, 60.0, 40.0],       
        "Volatility": [0.2, 0.25, 0.15, 0.3, 0.1],
        "Return": [0.1, 0.2, 0.05, 0.15, 0.02],
        "PE": [25.0, 35.0, 12.0, 10.0, 15.0],        
        "PEG": [2.5, 3.0, 1.2, 1.0, 1.5]
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_cov_matrix():
    """Generates a simple diagonal covariance matrix for 5 assets."""
    tickers = ["AAPL", "MSFT", "JPM", "XOM", "PFE"]
    return pd.DataFrame(np.eye(5) * 0.04, columns=tickers, index=tickers)

# --- TESTS: HELPER FUNCTIONS ---

def test_calculate_rsi(sample_price_data):
    """Test RSI calculation logic."""
    rsi = calculate_rsi(sample_price_data)
    assert not rsi.empty
    assert rsi.iloc[-1] > 50
    assert 0 <= rsi.iloc[-1] <= 100

# --- TESTS: OPTIMIZATION LOGIC ---

def test_linear_optimization_max_gain(mock_market_data):
    """Test Maximize Gain (Linear) with standard constraints."""
    max_weight = 0.5
    mode = "Standard (P/E)"
    
    # Run optimization directly via helper to bypass complex switching logic if needed
    result = run_linear_optimization(
        mock_market_data, 
        max_weight, 
        mode, 
        sector_limits=None, 
        bounds_list=[(0, 0.5)]*5, 
        use_strict_equality=False
    )
    
    assert not result.empty
    assert np.isclose(result["Weight"].sum(), 1.0, atol=0.01)
    assert result["Weight"].max() <= max_weight + 0.001

def test_sector_constraints_strict(mock_market_data):
    """Test that sector limits are respected (e.g., Tech = 40%)."""
    max_weight = 1.0
    mode = "Standard (P/E)"
    sector_limits = {"Information Technology": 0.40}
    
    result = optimize_portfolio(mock_market_data, "Maximize Gain", max_weight, mode, sector_limits=sector_limits)
    
    assert not result.empty
    tech_weight = result[result["Sector"] == "Information Technology"]["Weight"].sum()
    assert np.isclose(tech_weight, 0.40, atol=0.01)

def test_dynamic_bound_relaxation(mock_market_data):
    """
    Test 'Feasibility Override'.
    Scenario:
    - Tech Target: 50%
    - Tech Stocks: 2 (AAPL, MSFT)
    - Max Weight: 20%
    
    Standard Math: 2 stocks * 20% = 40% Capacity. 
    Problem: 40% < 50% Target.
    
    Expected Behavior: Optimizer relaxes Tech individual max weights to allow reaching 50%.
    Global Feasibility: 3 Non-Tech stocks * 20% = 60%. Total Capacity (40+60=100) is tight but valid.
    """
    max_weight_individual = 0.20  # UPDATED from 0.10 to ensure global feasibility
    target_tech_sector = 0.50     # Needs 0.50 total
    
    sector_limits = {"Information Technology": target_tech_sector}
    
    result = optimize_portfolio(
        mock_market_data, 
        "Maximize Gain", 
        max_weight_individual, 
        "Standard (P/E)", 
        sector_limits=sector_limits
    )
    
    assert not result.empty, "Optimization failed to find a solution"
    
    tech_rows = result[result["Sector"] == "Information Technology"]
    tech_sum = tech_rows["Weight"].sum()
    
    # 1. Tech Sector should reach the high target (0.50)
    assert np.isclose(tech_sum, 0.50, atol=0.01)
    
    # 2. To reach 50% with only 2 stocks, they MUST exceed the 20% individual cap
    # (Average weight would be 25%)
    assert tech_rows["Weight"].max() > 0.20

def test_minimize_volatility_quadratic(mock_market_data, mock_cov_matrix):
    """Test Minimize Volatility using the quadratic solver."""
    max_weight = 0.3
    
    result = optimize_portfolio(
        mock_market_data, 
        "Minimize Volatility", 
        max_weight, 
        "Standard (P/E)", 
        sector_limits=None, 
        cov_matrix=mock_cov_matrix
    )
    
    assert not result.empty
    assert np.isclose(result["Weight"].sum(), 1.0, atol=0.01)
    assert result["Weight"].max() <= max_weight + 0.001

def test_optimization_failure_handling(mock_market_data):
    """Test graceful failure for mathematically impossible constraints."""
    # Impossible: Max weight 0.05 for 5 assets (Total Cap 0.25 < 1.0 required)
    max_weight = 0.05 
    
    result = optimize_portfolio(mock_market_data, "Maximize Gain", max_weight, "Standard (P/E)")
    
    # Should return empty DataFrame, not crash
    assert result.empty

# --- TESTS: DATA INTEGRATION (MOCKED) ---

@patch('yfinance.download')
def test_process_bulk_data_integration(mock_yf_download):
    """Test data processing pipeline with mocked yfinance."""
    dates = pd.date_range("2023-01-01", periods=10)
    
    # Mock return objects
    mock_df_aapl = pd.DataFrame({"Close": [150.0]*10, "Open": [145.0]*10}, index=dates)
    mock_df_msft = pd.DataFrame({"Close": [300.0]*10, "Open": [295.0]*10}, index=dates)
    
    class MockYFData:
        def __getitem__(self, key):
            if key == "AAPL": return mock_df_aapl
            if key == "MSFT": return mock_df_msft
            return pd.DataFrame()
        def copy(self): return self

    mock_yf_download.return_value = MockYFData()
    
    with patch('yfinance.Ticker') as mock_ticker:
        mock_instance = mock_ticker.return_value
        mock_instance.info = {'trailingPE': 20.0, 'pegRatio': 1.5}
        
        tickers_input = ["AAPL", "MSFT"]
        sector_map = {"AAPL": "Tech", "MSFT": "Tech"}
        
        df_result, hist_data, cov_matrix = process_bulk_data(tickers_input, sector_map, "Standard (P/E)")
        
        assert not df_result.empty
        assert len(df_result) == 2
        assert "RSI" in df_result.columns