# ... (This is the end of the Quadrant Plot logic inside the if pe_ratio block)
        st.plotly_chart(fig_quad, use_container_width=True) # Changed width="stretch" to use_container_width=True
    else:
        st.warning("Cannot generate Value vs Momentum plot: P/E Ratio data is missing or negative.")
   
    # --- ROW 2: PRICE & RSI CHARTS ---
    st.subheader("Technical Analysis (Price & Trends)")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Price Chart (uses Close price)
    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'], high=df['High'],
                    low=df['Low'], close=df['Close'], name='OHLC'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], 
                            line=dict(color='orange', width=2), name='50 Day MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], 
                            line=dict(color='blue', width=2), name='200 Day MA'), row=1, col=1)

    # RSI Chart (uses the selected source)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], 
                            line=dict(color='purple', width=2), name=f'RSI ({rsi_source})'), row=2, col=1) 
    fig.add_hline(y=70, line_dash="dot", row=2, col=1, line_color="red")
    fig.add_hline(y=30, line_dash="dot", row=2, col=1, line_color="green")
    fig.add_hline(y=50, line_dash="solid", row=2, col=1, line_color="gray", opacity=0.5)

    fig.update_layout(xaxis_rangeslider_visible=False, height=600, margin=dict(l=20, r=20, t=30, b=20))
    
    # 🔴 FIX: Use 'use_container_width=True' instead of width=True
    st.plotly_chart(fig, use_container_width=True) 

else:
    st.warning("No data found. Please check the ticker symbol.")