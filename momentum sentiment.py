# Portfolio Optimization App in Streamlit

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

st.set_page_config(page_title="Portfolio Optimizer", layout="centered")

# Custom styled header
st.markdown("""
<div style='background-color:#0E1117; padding: 1rem; border-radius: 10px;'>
    <h1 style='color:#FAFAFA;'>🧠 Portfolio Optimizer</h1>
    <p style='color:#CCCCCC;'>Use quantitative methods to create optimized portfolios based on Sharpe Ratio or Minimum Variance. Designed for aspiring quants & data-driven investors.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("Configuration")
tickers = st.sidebar.text_input("Enter tickers (comma separated)", "AAPL,MSFT,GOOGL,AMZN,TSLA")
tickers = [t.strip().upper() for t in tickers.split(',') if t.strip()]
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2022-01-01'))
end_date = st.sidebar.date_input("End Date", pd.to_datetime('2024-01-01'))
opt_mode = st.sidebar.selectbox("Optimization Objective", ["Max Sharpe Ratio", "Minimum Variance"])

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

# Objective Functions
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0):
    annual_return = np.dot(weights, mean_returns) * 252
    annual_volatility = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(252)
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    balance_penalty = np.var(weights)
    concentration_penalty = np.sum(weights**3)
    score = sharpe_ratio - 0.2 * balance_penalty - 0.1 * concentration_penalty
    return annual_return, annual_volatility, score

def max_sharpe_objective(weights, mean_returns, cov_matrix):
    return -portfolio_performance(weights, mean_returns, cov_matrix)[2]

def min_variance_objective(weights, mean_returns, cov_matrix):
    return weights.T @ cov_matrix @ weights

penalty_factor = 10

def min_variance_with_penalty(weights, mean_returns, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    diversity_penalty = np.sum(weights**3)
    return variance + penalty_factor * diversity_penalty

initial_weights = np.array([1. / num_assets] * num_assets)
bounds = tuple((0, 1) for _ in range(num_assets))
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

if opt_mode == "Minimum Variance":
    result = minimize(min_variance_with_penalty, initial_weights,
                      args=(mean_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
else:
    result = minimize(max_sharpe_objective, initial_weights,
                      args=(mean_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)

opt_weights = np.round(result.x, 6)
ret, vol, score = portfolio_performance(opt_weights, mean_returns, cov_matrix)

st.subheader("📈 Optimization Results")
st.markdown(f"**Expected Annual Return:** {ret:.2%}")
st.markdown(f"**Annual Volatility:** {vol:.2%}")
if opt_mode == "Minimum Variance":
    st.markdown(f"**Variance + Penalty Objective Value:** {min_variance_with_penalty(opt_weights, mean_returns, cov_matrix):.6f}")
else:
    st.markdown(f"**Sharpe Ratio (adjusted for balance and concentration):** {score:.2f}")

st.subheader("🔍 Optimal Portfolio Weights")
weight_df = pd.DataFrame({"Ticker": tickers, "Weight": opt_weights})
weight_df["Weight"] = weight_df["Weight"] * 100
st.dataframe(weight_df[weight_df["Weight"] > 0.01].sort_values("Weight", ascending=False).reset_index(drop=True))

fig, ax = plt.subplots()
nonzero_weights = opt_weights > 0.01
ax.pie(opt_weights[nonzero_weights], labels=np.array(tickers)[nonzero_weights], autopct='%1.1f%%', startangle=140)
ax.set_title("Optimized Portfolio Allocation")
st.pyplot(fig)

st.subheader("📊 Historical Portfolio Performance")
portfolio_returns = (returns * opt_weights).sum(axis=1)
cumulative_returns = (1 + portfolio_returns).cumprod()
fig2, ax2 = plt.subplots()
ax2.plot(cumulative_returns, label='Optimized Portfolio')
ax2.set_title("Cumulative Returns Over Time")
ax2.set_ylabel("Portfolio Value")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)

var_95 = np.percentile(portfolio_returns, 5)
st.markdown(f"**1-Day 95% Value-at-Risk (VaR):** {var_95:.2%}")

st.subheader("🔗 Asset Correlation Heatmap")
fig3, ax3 = plt.subplots()
sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax3)
ax3.set_title("Correlation Between Assets")
st.pyplot(fig3)

# Monte Carlo Simulation and Efficient Frontier
st.subheader("🌀 Monte Carlo Simulated Portfolios")
n_sim = 5000
all_weights = np.zeros((n_sim, num_assets))
ret_arr = np.zeros(n_sim)
vol_arr = np.zeros(n_sim)
sharpe_arr = np.zeros(n_sim)

for i in range(n_sim):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    all_weights[i, :] = weights
    ret_arr[i] = np.dot(weights, mean_returns) * 252
    vol_arr[i] = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(252)
    sharpe_arr[i] = (ret_arr[i]) / vol_arr[i]

fig4, ax4 = plt.subplots()
sc = ax4.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis', alpha=0.5)
ax4.scatter(vol, ret, c='red', s=60, edgecolors='black', label='Optimized')
ax4.set_xlabel('Volatility')
ax4.set_ylabel('Expected Return')
ax4.set_title('Efficient Frontier & Simulated Portfolios')
ax4.grid(True)
fig4.colorbar(sc, label='Sharpe Ratio')
ax4.legend()
st.pyplot(fig4)

csv = weight_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Portfolio Weights (CSV)",
    data=csv,
    file_name="optimized_portfolio.csv",
    mime="text/csv"
)
