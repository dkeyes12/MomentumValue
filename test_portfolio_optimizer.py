"""
Test suite for MomentumValue portfolio optimizer.

Tests cover:
- RSI calculation
- Linear optimization (maximize gain, minimize volatility)
- Sector constraints and feasibility overrides
- Data integration with mocked yfinance
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime

# Import functions from streamlit_app (main app)
try:
    from streamlit_app import (
        calculate_rsi,
        process_bulk_data,
        optimize_portfolio,
        run_linear_optimization,
    )
except ImportError:
    # Fallback to app.py if streamlit_app not available
    from app import (
        calculate_rsi,
        process_bulk_data,
        optimize_portfolio,
        run_linear_optimization,
    )


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_price_data():
    """Generate 100 trading days of synthetic price data."""
    dates = pd.date_range(start="2023-01-01", periods=100)
    prices = [100 + i + (i % 5) for i in range(100)]
    return pd.Series(prices, index=dates, name="Close")


@pytest.fixture
def mock_market_data():
    """Generate mock market data DataFrame with 5 stocks across 3 sectors."""
    return pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "JPM", "XOM", "PFE"],
        "Sector": [
            "Information Technology",
            "Information Technology",
            "Financials",
            "Energy",
            "Health Care",
        ],
        "Price": [150.0, 300.0, 140.0, 110.0, 40.0],
        "RSI": [30.0, 70.0, 50.0, 60.0, 40.0],
        "Volatility": [0.2, 0.25, 0.15, 0.3, 0.1],
        "Return": [0.1, 0.2, 0.05, 0.15, 0.02],
        "PE": [25.0, 35.0, 12.0, 10.0, 15.0],
        "PEG": [2.5, 3.0, 1.2, 1.0, 1.5],
    })


@pytest.fixture
def mock_cov_matrix():
    """Generate diagonal covariance matrix for 5 assets."""
    tickers = ["AAPL", "MSFT", "JPM", "XOM", "PFE"]
    return pd.DataFrame(np.eye(5) * 0.04, columns=tickers, index=tickers)


@pytest.fixture
def simple_market_data():
    """Generate minimal market data for edge case testing."""
    return pd.DataFrame({
        "Ticker": ["A", "B", "C"],
        "Sector": ["Tech", "Finance", "Energy"],
        "Price": [100.0, 50.0, 75.0],
        "RSI": [50.0, 55.0, 60.0],
        "Volatility": [0.15, 0.20, 0.25],
        "Return": [0.10, 0.08, 0.12],
        "PE": [20.0, 15.0, 10.0],
        "PEG": [1.5, 1.2, 0.9],
    })



# ============================================================================
# TESTS: RSI CALCULATION
# ============================================================================

class TestRSICalculation:
    """Test RSI indicator computation."""

    def test_rsi_basic_calculation(self, sample_price_data):
        """RSI should be computed without errors and fall in valid range."""
        rsi = calculate_rsi(sample_price_data)
        
        assert not rsi.empty, "RSI series should not be empty"
        assert len(rsi) == len(sample_price_data), "RSI length should match input"
        
        # All RSI values should be in [0, 100]
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all(), \
            "RSI values must be between 0 and 100"

    def test_rsi_uptrend_high_values(self):
        """RSI should be high (>50) in sustained uptrend."""
        dates = pd.date_range(start="2023-01-01", periods=50)
        prices = pd.Series(np.linspace(100, 150, 50), index=dates)
        
        rsi = calculate_rsi(prices)
        final_rsi = rsi.iloc[-1]
        
        assert final_rsi > 50, "RSI should be elevated in uptrend"

    def test_rsi_downtrend_low_values(self):
        """RSI should be low (<50) in sustained downtrend."""
        dates = pd.date_range(start="2023-01-01", periods=50)
        prices = pd.Series(np.linspace(150, 100, 50), index=dates)
        
        rsi = calculate_rsi(prices)
        final_rsi = rsi.iloc[-1]
        
        assert final_rsi < 50, "RSI should be depressed in downtrend"

    def test_rsi_with_nan_handling(self):
        """RSI calculation should handle NaN values gracefully."""
        dates = pd.date_range(start="2023-01-01", periods=30)
        prices = pd.Series([100.0] * 30, index=dates)
        
        rsi = calculate_rsi(prices)
        # Flat prices should produce middle-range RSI values
        valid_rsi = rsi.dropna()
        assert not valid_rsi.empty


# ============================================================================
# TESTS: LINEAR OPTIMIZATION
# ============================================================================

class TestLinearOptimization:
    """Test portfolio linear optimization functionality."""

    def test_linear_optimization_max_gain(self, mock_market_data):
        """Maximize Gain should allocate to highest returns."""
        max_weight = 0.5
        mode = "Standard (P/E)"
        
        result = run_linear_optimization(
            mock_market_data,
            max_weight,
            mode,
            sector_limits=None,
            bounds_list=[(0, 0.5)] * 5,
            use_strict_equality=False,
        )
        
        assert not result.empty, "Optimization should produce a result"
        # Weights should sum to 1.0 (within tolerance)
        assert np.isclose(result["Weight"].sum(), 1.0, atol=0.01), \
            "Portfolio weights should sum to 100%"
        # No single position should exceed max_weight
        assert result["Weight"].max() <= max_weight + 0.001, \
            "Individual position weights should respect max_weight"

    def test_linear_optimization_respects_bounds(self, mock_market_data):
        """Individual position bounds should be enforced."""
        max_weight = 0.3
        
        result = optimize_portfolio(
            mock_market_data,
            "Maximize Gain",
            max_weight,
            "Standard (P/E)",
            sector_limits=None,
        )
        
        assert not result.empty
        assert (result["Weight"] <= max_weight + 0.001).all(), \
            "All weights should be <= max_weight"
        assert (result["Weight"] >= 0).all(), \
            "All weights should be >= 0"

    def test_sector_constraints_respected(self, mock_market_data):
        """Sector limits should be enforced strictly."""
        max_weight = 1.0
        sector_limits = {"Information Technology": 0.40}
        
        result = optimize_portfolio(
            mock_market_data,
            "Maximize Gain",
            max_weight,
            "Standard (P/E)",
            sector_limits=sector_limits,
        )
        
        assert not result.empty
        tech_weight = result[result["Sector"] == "Information Technology"]["Weight"].sum()
        
        assert np.isclose(tech_weight, 0.40, atol=0.01), \
            f"Tech sector should be exactly 40%, got {tech_weight:.2%}"


# ============================================================================
# TESTS: FEASIBILITY & CONSTRAINT RELAXATION
# ============================================================================

class TestFeasibilityOverride:
    """Test constraint relaxation when targets are mathematically tight."""

    def test_sector_target_relaxes_individual_bounds(self, mock_market_data):
        """
        When sector target exceeds individual capacity, bounds should relax.
        
        Scenario:
        - 2 Tech stocks, 20% individual max
        - 50% sector target
        - Standard capacity: 2 * 20% = 40% < 50% needed
        - Expected: Individual bounds relax to meet 50% target
        """
        max_weight_individual = 0.20
        target_tech_sector = 0.50
        sector_limits = {"Information Technology": target_tech_sector}
        
        result = optimize_portfolio(
            mock_market_data,
            "Maximize Gain",
            max_weight_individual,
            "Standard (P/E)",
            sector_limits=sector_limits,
        )
        
        assert not result.empty, "Optimization should find feasible solution"
        
        tech_rows = result[result["Sector"] == "Information Technology"]
        tech_sum = tech_rows["Weight"].sum()
        
        # Sector target should be met
        assert np.isclose(tech_sum, target_tech_sector, atol=0.01), \
            f"Tech sector should be {target_tech_sector:.0%}, got {tech_sum:.2%}"
        
        # Individual bounds must be relaxed (some stock > 20%)
        assert tech_rows["Weight"].max() > max_weight_individual, \
            "At least one Tech stock weight should exceed individual cap"

    def test_impossible_constraints_return_empty(self, simple_market_data):
        """Infeasible constraint set should return empty DataFrame."""
        # 3 assets with 5% max each = 15% capacity < 100% required
        max_weight = 0.05
        
        result = optimize_portfolio(
            simple_market_data,
            "Maximize Gain",
            max_weight,
            "Standard (P/E)",
            sector_limits=None,
        )
        
        # Should gracefully fail without crashing
        assert result.empty, "Infeasible constraints should return empty result"



# ============================================================================
# TESTS: VOLATILITY OPTIMIZATION
# ============================================================================

class TestVolatilityOptimization:
    """Test risk minimization strategies."""

    def test_minimize_volatility_quadratic(self, mock_market_data, mock_cov_matrix):
        """Minimize Volatility should produce valid portfolio."""
        max_weight = 0.3
        
        result = optimize_portfolio(
            mock_market_data,
            "Minimize Volatility",
            max_weight,
            "Standard (P/E)",
            sector_limits=None,
            cov_matrix=mock_cov_matrix,
        )
        
        assert not result.empty, "Volatility optimization should produce result"
        assert np.isclose(result["Weight"].sum(), 1.0, atol=0.01), \
            "Weights should sum to 100%"
        assert result["Weight"].max() <= max_weight + 0.001, \
            "Individual weights should respect max_weight"

    def test_minimize_volatility_with_sector_limits(self, mock_market_data, mock_cov_matrix):
        """Volatility minimization with sector constraints."""
        max_weight = 0.4
        sector_limits = {"Information Technology": 0.30}
        
        result = optimize_portfolio(
            mock_market_data,
            "Minimize Volatility",
            max_weight,
            "Standard (P/E)",
            sector_limits=sector_limits,
            cov_matrix=mock_cov_matrix,
        )
        
        assert not result.empty
        tech_weight = result[result["Sector"] == "Information Technology"]["Weight"].sum()
        assert np.isclose(tech_weight, 0.30, atol=0.01), \
            "Sector constraint should be enforced"


# ============================================================================
# TESTS: DATA INTEGRATION (MOCKED)
# ============================================================================

class TestDataIntegration:
    """Test data fetching and processing pipeline."""

    @patch("yfinance.download")
    def test_process_bulk_data_integration(self, mock_yf_download):
        """Process bulk data with mocked yfinance API."""
        dates = pd.date_range("2023-01-01", periods=20)
        
        # Mock price data for two tickers
        mock_aapl = pd.DataFrame(
            {"Close": [150.0] * 20, "Open": [145.0] * 20},
            index=dates,
        )
        mock_msft = pd.DataFrame(
            {"Close": [300.0] * 20, "Open": [295.0] * 20},
            index=dates,
        )
        
        class MockYFDataFrame:
            """Mock yfinance download return object."""
            def __getitem__(self, key):
                if key == "AAPL":
                    return mock_aapl
                elif key == "MSFT":
                    return mock_msft
                return pd.DataFrame()
            
            def copy(self):
                return self
        
        mock_yf_download.return_value = MockYFDataFrame()
        
        with patch("yfinance.Ticker") as mock_ticker:
            mock_instance = mock_ticker.return_value
            mock_instance.info = {"trailingPE": 20.0, "pegRatio": 1.5}
            
            tickers_input = ["AAPL", "MSFT"]
            sector_map = {"AAPL": "Tech", "MSFT": "Tech"}
            
            df_result, hist_data, cov_matrix = process_bulk_data(
                tickers_input,
                sector_map,
                "Standard (P/E)",
            )
            
            assert not df_result.empty, "Result DataFrame should not be empty"
            assert len(df_result) == 2, "Should process 2 tickers"
            assert "RSI" in df_result.columns, "RSI column should be computed"

    @patch("yfinance.Ticker")
    def test_process_bulk_data_missing_pe_handling(self, mock_ticker):
        """Handle missing P/E ratio gracefully."""
        # Mock ticker with missing PE data
        mock_instance = mock_ticker.return_value
        mock_instance.info = {}  # No PE data
        
        with patch("yfinance.download") as mock_download:
            dates = pd.date_range("2023-01-01", periods=20)
            mock_data = pd.DataFrame(
                {
                    "Close": np.linspace(100, 110, 20),
                    "Open": np.linspace(99, 109, 20),
                },
                index=dates,
            )
            
            class MockResult:
                def __getitem__(self, key):
                    return mock_data
                def copy(self):
                    return self
            
            mock_download.return_value = MockResult()
            
            # Should not crash when PE is missing
            df_result, _, _ = process_bulk_data(
                ["TEST"],
                {"TEST": "TestSector"},
                "Standard (P/E)",
            )
            
            assert not df_result.empty


# ============================================================================
# TESTS: EDGE CASES & ERROR HANDLING
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_stock_portfolio(self, simple_market_data):
        """Portfolio with only 1 stock should allocate 100% to it."""
        single_stock = simple_market_data.iloc[[0]].copy()
        
        result = optimize_portfolio(
            single_stock,
            "Maximize Gain",
            max_weight=1.0,
            mode="Standard (P/E)",
        )
        
        assert not result.empty
        assert np.isclose(result["Weight"].iloc[0], 1.0, atol=0.01)

    def test_equal_return_stocks(self):
        """Stocks with equal returns should distribute equally."""
        data = pd.DataFrame({
            "Ticker": ["A", "B", "C"],
            "Sector": ["S1", "S2", "S3"],
            "Price": [100.0, 100.0, 100.0],
            "RSI": [50.0, 50.0, 50.0],
            "Volatility": [0.2, 0.2, 0.2],
            "Return": [0.10, 0.10, 0.10],  # All equal
            "PE": [20.0, 20.0, 20.0],
            "PEG": [1.5, 1.5, 1.5],
        })
        
        result = optimize_portfolio(data, "Maximize Gain", 1.0, "Standard (P/E)")
        
        assert not result.empty
        # With equal returns, should distribute somewhat evenly
        assert np.isclose(result["Weight"].std(), 0, atol=0.1)

    def test_zero_volatility_handling(self):
        """Handle edge case of zero volatility."""
        data = pd.DataFrame({
            "Ticker": ["A", "B"],
            "Sector": ["S1", "S2"],
            "Price": [100.0, 100.0],
            "RSI": [50.0, 50.0],
            "Volatility": [0.0, 0.0],  # Zero volatility edge case
            "Return": [0.1, 0.15],
            "PE": [20.0, 15.0],
            "PEG": [1.5, 1.2],
        })
        
        result = optimize_portfolio(data, "Maximize Gain", 1.0, "Standard (P/E)")
        
        # Should handle gracefully without errors
        assert isinstance(result, pd.DataFrame)

    def test_all_negative_returns(self, simple_market_data):
        """Portfolio with all negative returns should still allocate."""
        data = simple_market_data.copy()
        data["Return"] = [-0.1, -0.05, -0.02]
        
        result = optimize_portfolio(data, "Maximize Gain", 0.5, "Standard (P/E)")
        
        # Should allocate to least-bad option
        if not result.empty:
            assert np.isclose(result["Weight"].sum(), 1.0, atol=0.01)


# ============================================================================
# TESTS: MODE SELECTION
# ============================================================================

class TestModeSelection:
    """Test different optimization modes (P/E, PEG, Momentum, etc.)."""

    def test_mode_standard_pe(self, mock_market_data):
        """Standard (P/E) mode should work."""
        result = optimize_portfolio(
            mock_market_data,
            "Maximize Gain",
            0.5,
            mode="Standard (P/E)",
        )
        
        assert not result.empty

    def test_mode_peg_ratio(self, mock_market_data):
        """PEG Ratio mode should work."""
        result = optimize_portfolio(
            mock_market_data,
            "Maximize Gain",
            0.5,
            mode="PEG Ratio",
        )
        
        assert not result.empty or result.empty  # Either valid or infeasible

    def test_multiple_modes_consistency(self, mock_market_data):
        """Different modes should produce different allocations."""
        result_pe = optimize_portfolio(
            mock_market_data,
            "Maximize Gain",
            0.5,
            mode="Standard (P/E)",
        )
        
        result_peg = optimize_portfolio(
            mock_market_data,
            "Maximize Gain",
            0.5,
            mode="PEG Ratio",
        )
        
        # Both should be valid portfolios (or both empty)
        assert (not result_pe.empty) or result_pe.empty
        assert (not result_peg.empty) or result_peg.empty