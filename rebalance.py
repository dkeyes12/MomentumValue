import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="S&P 500 Rebalance & Analysis", layout="wide")

# --- DATA: S&P 500 SECTOR WEIGHTS (APPROX JAN 2026) ---
BENCHMARK_DATA = {
    "Information Technology": 0.315,
    "Financials": 0.132,
    "Health Care": 0.124,
    "Consumer Discretionary": 0.103,
    "Communication Services": 0.088,
    "Industrials": 0.085,
    "Consumer Staples": 0.061,
    "Energy": 0.038,
    "Utilities": 0.024,
    "Real Estate": 0.023,
    "Materials": 0.022
}

# --- SAMPLE STOCK DATA FOR QUADRANT (Mock Data for Demo) ---
# In a real app, this would come from yfinance
SAMPLE_STOCKS = [
    {"Ticker": "NVDA", "Sector": "Information Technology", "RSI": 75, "PEG": 1.2, "Weight": 7.2},
    {"Ticker": "AAPL", "Sector": "Information Technology", "RSI": 55, "PEG": 2.8, "Weight": 6.0},
    {"Ticker": "MSFT", "Sector": "Information Technology", "RSI": 60, "PEG": 2.4, "Weight": 5.5},
    {"Ticker": "JPM", "Sector": "Financials", "RSI": 65, "PEG": 1.1, "Weight": 1.8},
    {"Ticker": "V", "Sector": "Financials", "RSI": 45, "PEG": 1.9, "Weight": 1.0},
    {"Ticker": "LLY", "Sector": "Health Care", "RSI": 80, "PEG": 1.5, "Weight": 1.5},
    {"Ticker": "JNJ", "Sector": "Health Care", "RSI": 40, "PEG": 3.2, "Weight": 1.2},
    {"Ticker": "XOM", "Sector": "Energy", "RSI": 55, "PEG": 0.9, "Weight": 1.1},
    {"Ticker": "AMZN", "Sector": "Consumer Discretionary", "RSI": 58, "PEG": 1.8, "Weight": 4.1},
    {"Ticker": "GOOGL", "Sector": "Communication Services", "RSI": 52, "PEG": 1.4, "Weight": 3.8},
    {"Ticker": "META", "Sector": "Communication Services", "RSI": 70, "PEG": 1.1, "Weight": 2.5},
    {"Ticker": "TSLA", "Sector": "Consumer Discretionary", "RSI": 35, "PEG": 4.5, "Weight": 2.3}
]
df_stocks = pd.DataFrame(SAMPLE_STOCKS)

def redistribute_weights(benchmark_dict, target_tech_weight):
    """
    Redistributes portfolio weights based on a target technology weight.
    Maintains relative proportionality of all non-tech sectors.
    """
    tech_key = "Information Technology"
    current_tech = benchmark_dict[tech_key]
    
    remaining_weight_new = 1.0 - target_tech_weight
    remaining_weight_old = 1.0 - current_tech
    
    if remaining_weight_old == 0:
        scale_factor = 0
    else:
        scale_factor = remaining_weight_new / remaining_weight_old
        
    new_weights = {}
    for sector, weight in benchmark_dict.items():
        if sector == tech_key:
            new_weights[sector] = target_tech_weight
        else:
            new_weights[sector] = weight * scale_factor
            
    return new_weights

def main():
    st.title("⚖️ S&P 500 Rebalance & Analysis")
    st.markdown("""
    **Objective:** Adjust broad sector exposure (Top-Down) while analyzing individual asset quality (Bottom-Up).
    """)

    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("1. Rebalance Constraints")
    
    current_tech_pct = BENCHMARK_DATA["Information Technology"] * 100
    
    tech_cap = st.sidebar.slider(
        "Max Tech Exposure (%)",
        min_value=0.0, max_value=60.0, value=float(current_tech_pct), step=0.5
    )
    
    target_tech = tech_cap / 100.0
    
    # --- CALCULATIONS ---
    new_allocation = redistribute_weights(BENCHMARK_DATA, target_tech)
    
    df = pd.DataFrame([
        {"Sector": k, "Benchmark Weight": v, "Custom Weight": new_allocation[k]} 
        for k, v in BENCHMARK_DATA.items()
    ])
    
    df["Delta"] = df["Custom Weight"] - df["Benchmark Weight"]
    active_share = np.sum(np.abs(df["Custom Weight"] - df["Benchmark Weight"])) / 2
    
    # --- METRICS ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Target Tech Weight", f"{tech_cap:.1f}%", f"{tech_cap - current_tech_pct:.1f}%", delta_color="inverse")
    col2.metric("Active Share", f"{active_share:.1%}", help="Deviation from S&P 500")
    beneficiary = df[df["Sector"] != "Information Technology"].sort_values("Delta", ascending=False).iloc[0]
    col3.metric("Top Beneficiary", beneficiary["Sector"], f"+{beneficiary['Delta']:.1%}")

    st.divider()

    # --- TABS ---
    tab_alloc, tab_quad, tab_details = st.tabs(["📊 Sector Allocation", "🎯 Asset Selection Matrix", "📝 Detailed Data"])
    
    # === TAB 1: SECTOR ALLOCATION ===
    with tab_alloc:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["Sector"], y=df["Benchmark Weight"], name="S&P 500", marker_color="lightgray"))
        fig.add_trace(go.Bar(x=df["Sector"], y=df["Custom Weight"], name="Rebalanced", marker_color="#4F8BF9"))
        
        fig.update_layout(title="Sector Weight Comparison", yaxis_tickformat='.0%', barmode='group', height=450)
        st.plotly_chart(fig, use_container_width=True)

    # === TAB 2: QUADRANT ANALYSIS ===
    with tab_quad:
        st.markdown("### Momentum vs. Value Matrix (Top Holdings)")
        st.caption("Analyzing top holdings to see if they justify their weight in the rebalanced portfolio.")
        
        # Quadrant Logic
        RSI_THRESHOLD = 50
        PEG_THRESHOLD = 1.5
        MAX_X = 5.0  # Cap PEG for visualization
        
        fig_quad = go.Figure()

        # Add Scatter Points
        # Color by Sector to link back to rebalancing theme
        fig_quad.add_trace(go.Scatter(
            x=df_stocks['PEG'].clip(upper=MAX_X),
            y=df_stocks['RSI'],
            mode='markers+text',
            text=df_stocks['Ticker'],
            textposition="top center",
            marker=dict(
                size=df_stocks['Weight']*3, # Bubble size = Weight in index
                color=df_stocks['RSI'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="RSI Strength")
            ),
            hovertemplate="<b>%{text}</b><br>RSI: %{y}<br>PEG: %{x}<br>Weight: %{marker.size:.2f}%<extra></extra>"
        ))

        # Add Quadrant Backgrounds
        # 1. Best: Low PEG (Left), High RSI (Top) -> Green
        fig_quad.add_shape(type="rect", x0=0, y0=50, x1=PEG_THRESHOLD, y1=100, fillcolor="green", opacity=0.1, layer="below", line_width=0)
        # 2. Expensive Momentum: High PEG (Right), High RSI (Top) -> Orange
        fig_quad.add_shape(type="rect", x0=PEG_THRESHOLD, y0=50, x1=MAX_X, y1=100, fillcolor="orange", opacity=0.1, layer="below", line_width=0)
        # 3. Value Trap: Low PEG (Left), Low RSI (Bottom) -> Yellow
        fig_quad.add_shape(type="rect", x0=0, y0=0, x1=PEG_THRESHOLD, y1=50, fillcolor="yellow", opacity=0.1, layer="below", line_width=0)
        # 4. Weak: High PEG (Right), Low RSI (Bottom) -> Red
        fig_quad.add_shape(type="rect", x0=PEG_THRESHOLD, y0=0, x1=MAX_X, y1=50, fillcolor="red", opacity=0.1, layer="below", line_width=0)

        # Labels
        fig_quad.add_annotation(x=PEG_THRESHOLD/2, y=95, text="VALUE + MOMENTUM", showarrow=False, font=dict(color="green", weight="bold"))
        fig_quad.add_annotation(x=3.5, y=95, text="EXPENSIVE MOMENTUM", showarrow=False, font=dict(color="orange", weight="bold"))
        fig_quad.add_annotation(x=PEG_THRESHOLD/2, y=5, text="WEAK / VALUE TRAP", showarrow=False, font=dict(color="orange", weight="bold"))
        fig_quad.add_annotation(x=3.5, y=5, text="EXPENSIVE & WEAK", showarrow=False, font=dict(color="red", weight="bold"))

        fig_quad.update_layout(
            xaxis_title="Valuation (PEG Ratio) - Lower is Better",
            yaxis_title="Momentum (RSI) - Higher is Better",
            height=600,
            xaxis=dict(range=[0, MAX_X]),
            yaxis=dict(range=[20, 90])
        )
        
        st.plotly_chart(fig_quad, use_container_width=True)
        
        st.info("💡 **Bubble Size** represents the stock's current weight in the S&P 500.")

    # === TAB 3: DETAILS ===
    with tab_details:
        # Format for display
        display_df = df.copy()
        display_df["Benchmark Weight"] = display_df["Benchmark Weight"].apply(lambda x: f"{x:.1%}")
        display_df["Custom Weight"] = display_df["Custom Weight"].apply(lambda x: f"{x:.1%}")
        display_df["Delta"] = display_df["Delta"].apply(lambda x: f"{x:+.1%}")
        
        st.dataframe(display_df, use_container_width=True, height=500)

if __name__ == "__main__":
    main()