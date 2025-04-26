# Portfolio Optimization App in Streamlit

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="Portfolio Optimizer", layout="centered")
st.title("📊 Portfolio Optimization App")

# Sidebar inputs
st.sidebar.header("Configuration")
tickers = st.sidebar.text_input("Enter tickers (comma separated)", "AAPL,MSFT,GOOGL,AMZN,TSLA")
tickers = [t.strip().upper() for t in tickers.split(',') if t.strip()]
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2022-01-01'))
end_date = st.sidebar.date_input("End Date", pd.to_datetime('2024-01-01'))

if len(tickers) < 2:
    st.warning("Please enter at least two tickers.")
    st.stop()

# Download data safely
data = yf.download(tickers, start=start_date, end=end_date)

if data.empty:
    st.error("No data retrieved. Please check your ticker symbols and date range.")
    st.stop()

# Try to access 'Adj Close', fall back to 'Close'
try:
    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.levels[0]:
            data = data['Adj Close']
        elif 'Close' in data.columns.levels[0]:
            st.warning("Using 'Close' instead of 'Adj Close' as fallback.")
            data = data['Close']
        else:
            st.error("Neither 'Adj Close' nor 'Close' found in the data.")
            st.stop()
    else:
        if 'Adj Close' in data.columns:
            data = data[['Adj Close']]
        elif 'Close' in data.columns:
            st.warning("Using 'Close' instead of 'Adj Close' as fallback.")
            data = data[['Close']]
        else:
            st.error("'Adj Close' or 'Close' not found in single ticker data.")
            st.stop()
except Exception as e:
    st.error(f"Unexpected error selecting price data: {str(e)}")
    st.stop()

returns = data.pct_change().dropna()

# Mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_assets = len(tickers)

# Portfolio performance metrics
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns) * 252
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = returns / std_dev
    return returns, std_dev, sharpe_ratio

def negative_sharpe(weights, mean_returns, cov_matrix):
    return -portfolio_performance(weights, mean_returns, cov_matrix)[2]

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(num_assets))
initial_weights = num_assets * [1. / num_assets]

# Optimize
opt_result = minimize(negative_sharpe, initial_weights,
                      args=(mean_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
opt_weights = opt_result.x
opt_weights = np.round(opt_weights, 6)
ret, vol, sharpe = portfolio_performance(opt_weights, mean_returns, cov_matrix)

# Results
st.subheader("📈 Optimization Results")
st.markdown(f"**Expected Annual Return:** {ret:.2%}")
st.markdown(f"**Annual Volatility:** {vol:.2%}")
st.markdown(f"**Sharpe Ratio:** {sharpe:.2f}")

# Display weights
st.subheader("🔍 Optimal Portfolio Weights")
weight_df = pd.DataFrame({"Ticker": tickers, "Weight": opt_weights})
weight_df["Weight"] = weight_df["Weight"] * 100
st.dataframe(weight_df[weight_df["Weight"] > 0.01].sort_values("Weight", ascending=False).reset_index(drop=True))

# Pie chart
nonzero_weights = opt_weights > 0.01
fig, ax = plt.subplots()
ax.pie(opt_weights[nonzero_weights], labels=np.array(tickers)[nonzero_weights], autopct='%1.1f%%', startangle=140)
ax.set_title("Optimized Portfolio Allocation")
st.pyplot(fig)
