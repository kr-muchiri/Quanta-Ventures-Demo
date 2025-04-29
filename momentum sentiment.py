# Portfolio Optimization App in Streamlit

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from datetime import datetime, timedelta
import time

# Configure the page
st.set_page_config(page_title="Advanced Portfolio Optimizer", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 7px;
        padding: 15px;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px;
        color: #565656;
    }
    .stAlert {
        border-radius: 7px !important;
    }
</style>
""", unsafe_allow_html=True)

# Custom styled header
st.markdown("""
<div style='background-color:#0E1117; padding: 1.5rem; border-radius: 10px; width: 100%; display: flex; flex-direction: column; align-items: center;'>
    <h1 style='color:#FAFAFA;'>🧠 Advanced Portfolio Optimizer</h1>
    <p style='color:#CCCCCC;'>Use quantitative methods to create optimized portfolios with advanced metrics, benchmarks, and risk controls.</p>
</div>
""", unsafe_allow_html=True)

# ----- HELPER FUNCTIONS -----

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_data(tickers, start_date, end_date, benchmark=None):
    """Fetch stock data with error handling and caching"""
    if benchmark and benchmark not in tickers:
        fetch_tickers = tickers + [benchmark]
    else:
        fetch_tickers = tickers
    
    try:
        data = yf.download(fetch_tickers, start=start_date, end=end_date)
        if data.empty or len(data) < 20:  # Make sure we have enough data
            st.error("Insufficient data retrieved. Please check your ticker symbols and date range.")
            return None
            
        # Check which column structure we have
        if isinstance(data.columns, pd.MultiIndex):
            if 'Adj Close' in data.columns.levels[0]:
                price_data = data['Adj Close'].copy()
            elif 'Close' in data.columns.levels[0]:
                st.warning("Using 'Close' instead of 'Adj Close' prices as fallback.")
                price_data = data['Close'].copy()
            else:
                st.error("Price data unavailable for the selected tickers.")
                return None
        else:
            # Single ticker case
            if 'Adj Close' in data.columns:
                price_data = data['Adj Close'].to_frame()
            else:
                st.warning("Using 'Close' instead of 'Adj Close' prices as fallback.")
                price_data = data['Close'].to_frame()
                
        # Handle tickers with missing data
        missing_data = price_data.columns[price_data.isna().sum() > len(price_data) * 0.3]
        if len(missing_data) > 0:
            st.warning(f"Removed {', '.join(missing_data)} due to insufficient data")
            price_data = price_data.drop(columns=missing_data)
            
        return price_data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def calculate_portfolio_metrics(weights, returns, cov_matrix, risk_free_rate=0.0):
    """Calculate comprehensive portfolio metrics"""
    # Annualized metrics
    annual_return = np.dot(weights, returns.mean()) * 252
    annual_volatility = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(252)
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
    
    # Diversification metrics
    diversification_ratio = np.sum(weights * returns.std() * np.sqrt(252)) / annual_volatility
    herfindahl_index = np.sum(weights**2)  # Lower is more diversified
    
    # Downside metrics
    downside_returns = returns[returns < 0]
    downside_deviation = np.sqrt(np.dot(weights, downside_returns.mean()) ** 2) * np.sqrt(252)
    sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
    
    # Portfolio beta (if benchmark provided)
    beta = None
    if 'SPY' in returns.columns:
        portfolio_returns = (returns.drop(columns=['SPY']) * weights[:-1]).sum(axis=1)
        market_returns = returns['SPY']
        beta = np.cov(portfolio_returns, market_returns)[0, 1] / np.var(market_returns)
    
    return {
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'diversification_ratio': diversification_ratio,
        'herfindahl_index': herfindahl_index,
        'beta': beta
    }

# ----- OPTIMIZATION FUNCTIONS -----

def max_sharpe_objective(weights, returns, cov_matrix, risk_free_rate=0.0, balance_penalty=0.2, concentration_penalty=0.1):
    """Maximize sharpe ratio with penalties for over-concentration"""
    metrics = calculate_portfolio_metrics(weights, returns, cov_matrix, risk_free_rate)
    
    # Penalties for concentration
    balance_penalty_value = np.var(weights) * balance_penalty
    concentration_penalty_value = np.sum(weights**3) * concentration_penalty
    
    # We minimize the negative of the objective
    return -(metrics['sharpe_ratio'] - balance_penalty_value - concentration_penalty_value)

def min_variance_objective(weights, returns, cov_matrix, balance_penalty=0.2):
    """Minimize portfolio variance with a penalty for concentration"""
    variance = weights.T @ cov_matrix @ weights
    diversity_penalty = np.sum(weights**3) * balance_penalty
    return variance + diversity_penalty

def optimize_portfolio(returns, cov_matrix, method="Max Sharpe", risk_tolerance=1.0, risk_free_rate=0.0):
    """Portfolio optimization with different objectives"""
    num_assets = returns.shape[1]
    
    # Initial weights, bounds, and constraints
    initial_weights = np.array([1.0 / num_assets] * num_assets)
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Apply different optimization strategies
    if method == "Max Sharpe Ratio":
        # Adjust penalties based on risk tolerance (higher tolerance = lower penalties)
        balance_penalty = 0.2 * (2 - risk_tolerance)
        concentration_penalty = 0.1 * (2 - risk_tolerance)
        
        result = minimize(
            max_sharpe_objective, 
            initial_weights,
            args=(returns, cov_matrix, risk_free_rate, balance_penalty, concentration_penalty),
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )
    elif method == "Minimum Variance":
        # Adjust penalty based on risk tolerance
        balance_penalty = 0.2 * (2 - risk_tolerance)
        
        result = minimize(
            min_variance_objective, 
            initial_weights,
            args=(returns, cov_matrix, balance_penalty),
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )
    elif method == "Maximum Diversification":
        # For maximum diversification, we maximize the ratio of weighted volatilities to portfolio volatility
        def neg_diversification_ratio(weights):
            diversification_ratio = np.sum(weights * returns.std() * np.sqrt(252)) / np.sqrt(weights.T @ cov_matrix @ weights) / np.sqrt(252)
            concentration_penalty = np.sum(weights**3) * 0.1 * (2 - risk_tolerance)
            return -diversification_ratio + concentration_penalty
            
        result = minimize(
            neg_diversification_ratio, 
            initial_weights,
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )
    elif method == "Equal Weight":
        # Simple equal weighting
        return initial_weights
    else:
        st.error(f"Unknown optimization method: {method}")
        return initial_weights
    
    if not result.success:
        st.warning(f"Optimization did not converge. Using fallback method. Message: {result.message}")
        return initial_weights
        
    return result.x

def simulate_portfolios(returns, cov_matrix, n_simulations=3000):
    """Run Monte Carlo simulation to generate random portfolios"""
    num_assets = returns.shape[1]
    
    all_weights = np.zeros((n_simulations, num_assets))
    ret_arr = np.zeros(n_simulations)
    vol_arr = np.zeros(n_simulations)
    sharpe_arr = np.zeros(n_simulations)
    
    for i in range(n_simulations):
        # Generate random weights
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        all_weights[i, :] = weights
        
        # Calculate performance metrics
        ret_arr[i] = np.dot(weights, returns.mean()) * 252
        vol_arr[i] = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(252)
        sharpe_arr[i] = ret_arr[i] / vol_arr[i] if vol_arr[i] > 0 else 0
    
    return {
        'weights': all_weights,
        'returns': ret_arr,
        'volatility': vol_arr,
        'sharpe': sharpe_arr
    }

# ----- ANALYSIS & VISUALIZATION FUNCTIONS -----

def analyze_optimized_portfolio(price_data, opt_weights, benchmark=None):
    """Calculate performance metrics for the optimized portfolio"""
    returns = price_data.pct_change().dropna()
    
    # Calculate daily returns of optimized portfolio
    if benchmark and benchmark in returns.columns:
        portfolio_returns = np.dot(returns.drop(columns=[benchmark]), opt_weights[:-1])
        benchmark_returns = returns[benchmark]
    else:
        portfolio_returns = np.dot(returns, opt_weights)
        benchmark_returns = None
    
    # Convert to series if it's not already
    if not isinstance(portfolio_returns, pd.Series):
        portfolio_returns = pd.Series(portfolio_returns, index=returns.index)
    
    # Calculate cumulative returns
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    if benchmark_returns is not None:
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
    else:
        benchmark_cumulative = None
    
    # Calculate drawdowns - with error handling
    try:
        portfolio_peak = portfolio_cumulative.expanding().max()
        portfolio_drawdown = (portfolio_cumulative - portfolio_peak) / portfolio_peak
    except Exception as e:
        st.warning(f"Could not calculate drawdown: {str(e)}")
        portfolio_drawdown = pd.Series(0, index=portfolio_cumulative.index)
    
    # Calculate rolling metrics (1-year window)
    window = min(252, len(portfolio_returns) // 2)  # Use window of 252 days or half the data, whichever is smaller
    
    try:
        if len(portfolio_returns) > window:
            rolling_vol = portfolio_returns.rolling(window=window).std() * np.sqrt(252)
            rolling_ret = portfolio_returns.rolling(window=window).mean() * 252
            rolling_sharpe = rolling_ret / rolling_vol
        else:
            rolling_vol = rolling_ret = rolling_sharpe = None
    except Exception as e:
        st.warning(f"Could not calculate rolling metrics: {str(e)}")
        rolling_vol = rolling_ret = rolling_sharpe = None
    
    # Calculate Value at Risk and Conditional Value at Risk
    try:
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
    except Exception as e:
        st.warning(f"Could not calculate VaR/CVaR: {str(e)}")
        var_95 = cvar_95 = 0
    
    # Calculate correlation with benchmark
    correlation_with_benchmark = None
    if benchmark_returns is not None:
        try:
            correlation_with_benchmark = np.corrcoef(portfolio_returns, benchmark_returns)[0, 1]
        except Exception as e:
            st.warning(f"Could not calculate correlation with benchmark: {str(e)}")
    
    return {
        'returns': portfolio_returns,
        'cumulative_returns': portfolio_cumulative,
        'benchmark_returns': benchmark_returns,
        'benchmark_cumulative': benchmark_cumulative,
        'drawdown': portfolio_drawdown,
        'rolling_volatility': rolling_vol,
        'rolling_returns': rolling_ret,
        'rolling_sharpe': rolling_sharpe,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'correlation_with_benchmark': correlation_with_benchmark
    }

def plot_portfolio_weights(tickers, weights, title="Portfolio Allocation"):
    """Create pie chart for portfolio weights"""
    # Handle case where lengths don't match
    if len(tickers) != len(weights):
        st.warning(f"Length mismatch: {len(tickers)} tickers but {len(weights)} weights")
        # Use the shorter length
        min_len = min(len(tickers), len(weights))
        tickers = tickers[:min_len]
        weights = weights[:min_len]
    
    # Make sure weights is a numpy array
    weights = np.array(weights)
    
    # Filter out negligible weights for cleaner chart
    threshold = 0.01
    nonzero_indices = weights > threshold
    
    # Error check before indexing
    if len(nonzero_indices) == 0:
        st.warning("No weights above threshold found")
        # Show all weights instead
        filtered_tickers = tickers
        filtered_weights = weights
    else:
        try:
            filtered_tickers = np.array(tickers)[nonzero_indices]
            filtered_weights = weights[nonzero_indices]
        except IndexError:
            # Fallback if indexing fails
            st.warning("Error filtering weights. Showing all weights instead.")
            filtered_tickers = tickers
            filtered_weights = weights
    
    # If all weights are below threshold or we have no values, show the top 5
    if len(filtered_tickers) == 0:
        try:
            # Get top 5 weights by value
            top_indices = np.argsort(weights)[-5:]
            filtered_tickers = np.array(tickers)[top_indices]
            filtered_weights = weights[top_indices]
        except IndexError:
            # Last resort fallback
            st.error("Unable to process weights for visualization")
            # Create a dummy chart
            filtered_tickers = ["Error"]
            filtered_weights = [1.0]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Use a visually appealing color scheme
    cmap = plt.cm.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(len(filtered_tickers))]
    
    # Create the pie chart with error handling
    try:
        wedges, texts, autotexts = ax.pie(
            filtered_weights, 
            labels=filtered_tickers, 
            autopct='%1.1f%%', 
            startangle=140, 
            colors=colors,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1}
        )
        
        # Styling for better readability
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_color('white')
    except Exception as e:
        # If pie chart fails, create a simple placeholder
        st.error(f"Error creating pie chart: {str(e)}")
        ax.text(0.5, 0.5, "Chart creation failed", ha='center', va='center')
    
    ax.set_title(title, fontsize=14, pad=20)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    return fig

def plot_efficient_frontier(simulation_results, optimized_metrics, opt_weights, returns, method):
    """Plot the efficient frontier with the optimized portfolio highlighted"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Get simulation results
    vol_arr = simulation_results['volatility']
    ret_arr = simulation_results['returns']
    sharpe_arr = simulation_results['sharpe']
    
    # Plot random portfolios
    scatter = ax.scatter(
        vol_arr, ret_arr, 
        c=sharpe_arr, 
        cmap='viridis', 
        alpha=0.5,
        s=30
    )
    
    # Plot optimized portfolio
    opt_vol = optimized_metrics['annual_volatility']
    opt_ret = optimized_metrics['annual_return']
    
    ax.scatter(
        opt_vol, opt_ret, 
        c='red', 
        s=100, 
        edgecolors='black', 
        label=f'Optimized Portfolio ({method})'
    )
    
    # Plot individual assets
    asset_returns = returns.mean() * 252
    asset_volatility = returns.std() * np.sqrt(252)
    
    for i, ticker in enumerate(returns.columns):
        weight = opt_weights[i]
        # Only show tickers with significant weight
        if weight > 0.05:
            ax.scatter(
                asset_volatility[i], 
                asset_returns[i], 
                marker='o',
                s=80,
                label=f"{ticker} ({weight*100:.1f}%)"
            )
    
    # Add labels and title
    ax.set_xlabel('Annual Volatility', fontsize=12)
    ax.set_ylabel('Annual Expected Return', fontsize=12)
    ax.set_title('Efficient Frontier & Simulated Portfolios', fontsize=14)
    
    # Format axis as percentage
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # Add colorbar
    cbar = fig.colorbar(scatter, pad=0.02)
    cbar.set_label('Sharpe Ratio', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_performance_comparison(portfolio_data, individual_assets=None):
    """Plot comparative performance of portfolio vs benchmark and/or assets"""
    portfolio_cum = portfolio_data['cumulative_returns']
    benchmark_cum = portfolio_data['benchmark_cumulative']
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot portfolio
    ax.plot(portfolio_cum, label='Optimized Portfolio', linewidth=2.5)
    
    # Plot benchmark if available
    if benchmark_cum is not None:
        ax.plot(benchmark_cum, label='Benchmark', linewidth=2, linestyle='--')
    
    # Plot individual assets if requested and available
    if individual_assets is not None:
        for ticker in individual_assets:
            if ticker in individual_assets.columns:
                asset_cum = (1 + individual_assets[ticker].pct_change().dropna()).cumprod()
                ax.plot(asset_cum, label=ticker, linewidth=1, alpha=0.7)
    
    # Styling
    ax.set_title("Cumulative Returns Comparison", fontsize=14)
    ax.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format y-axis as multiplier
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}x'))
    
    plt.tight_layout()
    return fig

def plot_drawdown(drawdown):
    """Plot portfolio drawdown over time"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.fill_between(
        drawdown.index, 
        drawdown.values * 100, 
        0, 
        color='coral', 
        alpha=0.7
    )
    ax.plot(drawdown.index, drawdown.values * 100, color='darkred', alpha=0.7)
    
    # Add horizontal lines at common drawdown levels
    ax.axhline(y=-10, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=-20, color='gray', linestyle='--', alpha=0.5)
    
    # Set labels and title
    ax.set_title('Portfolio Drawdown', fontsize=14)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    
    # Invert y-axis so drawdowns go down
    ax.invert_yaxis()
    
    # Grid and ticks
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    
    # Add annotations for maximum drawdown
    max_drawdown = drawdown.min() * 100
    max_drawdown_date = drawdown.idxmin()
    
    # Add annotation for max drawdown
    ax.annotate(
        f'Max Drawdown: {max_drawdown:.1f}%',
        xy=(max_drawdown_date, max_drawdown),
        xytext=(max_drawdown_date, max_drawdown - 5),
        arrowprops=dict(facecolor='black', arrowstyle='->'),
        fontsize=10,
        backgroundcolor='white',
        alpha=0.8
    )
    
    plt.tight_layout()
    return fig

def plot_rolling_metrics(portfolio_data):
    """Plot rolling performance metrics"""
    rolling_vol = portfolio_data['rolling_volatility']
    rolling_ret = portfolio_data['rolling_returns']
    rolling_sharpe = portfolio_data['rolling_sharpe']
    
    if rolling_vol is None or rolling_ret is None or rolling_sharpe is None:
        return None
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot rolling return
    axes[0].plot(rolling_ret * 100, color='darkgreen')
    axes[0].set_title('Rolling Annual Return (%)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    
    # Plot rolling volatility
    axes[1].plot(rolling_vol * 100, color='darkred')
    axes[1].set_title('Rolling Annual Volatility (%)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    
    # Plot rolling Sharpe
    axes[2].plot(rolling_sharpe, color='darkblue')
    axes[2].set_title('Rolling Sharpe Ratio', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    return fig

def display_metrics_dashboard(metrics, portfolio_data):
    """Display key metrics in a dashboard format"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{metrics['annual_return']*100:.2f}%</div>
            <div class="metric-label">Annual Return</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{metrics['sharpe_ratio']:.2f}</div>
            <div class="metric-label">Sharpe Ratio</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{metrics['annual_volatility']*100:.2f}%</div>
            <div class="metric-label">Annual Volatility</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{metrics['sortino_ratio']:.2f}</div>
            <div class="metric-label">Sortino Ratio</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{portfolio_data['var_95']*100:.2f}%</div>
            <div class="metric-label">Daily VaR (95%)</div>
        </div>
        """, unsafe_allow_html=True)
        
        if metrics['beta'] is not None:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{metrics['beta']:.2f}</div>
                <div class="metric-label">Portfolio Beta</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{metrics['diversification_ratio']:.2f}</div>
                <div class="metric-label">Diversification Ratio</div>
            </div>
            """, unsafe_allow_html=True)

# ----- MAIN APP -----

# Sidebar inputs
with st.sidebar:
    st.header("Portfolio Settings")
    
    # Ticker input
    ticker_input = st.text_input(
        "Enter tickers (comma separated)", 
        "AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,JPM,JNJ,V,PG,HD"
    ).upper()
    
    # Process tickers
    tickers = [t.strip() for t in ticker_input.split(',') if t.strip()]
    
    # Basic validation
    if len(tickers) < 2:
        st.sidebar.error("Please enter at least two tickers")
        st.stop()
    
    # Date range
    st.subheader("Time Period")
    end_date = st.date_input("End Date", datetime.now() - timedelta(days=1))
    
    time_period = st.selectbox(
        "Select period",
        ["1 Year", "3 Years", "5 Years", "10 Years", "Custom"],
        index=1
    )
    
    if time_period == "Custom":
        start_date = st.date_input("Start Date", end_date - timedelta(days=3*365))
    else:
        years = int(time_period.split()[0])
        start_date = end_date - timedelta(days=years*365)
    
    # Validate dates
    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date")
        st.stop()
    
    # Benchmark selection
    include_benchmark = st.checkbox("Include benchmark", True)
    if include_benchmark:
        benchmark = st.selectbox(
            "Select benchmark",
            ["SPY", "QQQ", "IWM", "VTI", "None"],
            index=0
        )
        if benchmark == "None":
            benchmark = None
    else:
        benchmark = None
    
    # Optimization settings
    st.subheader("Optimization Settings")
    
    opt_method = st.selectbox(
        "Optimization Method", 
        ["Max Sharpe Ratio", "Minimum Variance", "Maximum Diversification", "Equal Weight"],
        index=0
    )
    
    risk_tolerance = st.slider(
        "Risk Tolerance", 
        min_value=0.0, 
        max_value=2.0, 
        value=1.0, 
        step=0.1,
        help="Higher values allow more concentrated positions"
    )
    
    risk_free_rate = st.number_input(
        "Risk-free Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=2.5,
        step=0.25
    ) / 100.0
    
    # Advanced options
    with st.expander("Advanced Settings"):
        n_simulations = st.slider(
            "Monte Carlo Simulations",
            min_value=1000,
            max_value=10000,
            value=3000,
            step=1000
        )
        
        include_individual_assets = st.checkbox(
            "Show individual assets in performance chart",
            False
        )

    run_optimization = st.button("Optimize Portfolio", type="primary", use_container_width=True)

# Strategy explanations
with st.expander("ℹ️ Optimization Methods Explained"):
    st.markdown("""
    - **Max Sharpe Ratio**: Maximizes risk-adjusted return, balancing expected return and volatility.
    - **Minimum Variance**: Creates the portfolio with the lowest possible volatility, regardless of expected return.
    - **Maximum Diversification**: Maximizes the ratio of weighted volatilities to portfolio volatility, seeking to diversify risk sources.
    - **Equal Weight**: Assigns equal weight to all assets, a naive diversification approach that often performs surprisingly well.
    """)

with st.expander("ℹ️ Risk Tolerance Explained"):
    st.markdown("""
    The Risk Tolerance slider controls how concentrated your portfolio can be:
    - **Lower values** (0.0-0.9): More conservative, enforces greater diversification, may underweight high-performing assets.
    - **Medium values** (1.0): Balanced approach, moderate concentration allowed.
    - **Higher values** (1.1-2.0): More aggressive, allows greater concentration in promising assets, potentially higher risk.
    """)

# Main app logic
if run_optimization:
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Fetch data
    status_text.text("Fetching stock data...")
    price_data = fetch_data(tickers, start_date, end_date, benchmark)
    progress_bar.progress(20)
    
    if price_data is None or price_data.empty:
        st.error("Failed to retrieve valid price data.")
        st.stop()
    
    # Update tickers list if some were removed due to missing data
    tickers = [col for col in price_data.columns if col != benchmark]
    
    # Calculate returns and cov matrix
    status_text.text("Calculating returns and risk model...")
    returns = price_data.pct_change().dropna()
    cov_matrix = returns.cov()
    progress_bar.progress(40)
    
    # Optimize portfolio
    status_text.text(f"Optimizing portfolio using {opt_method}...")
    opt_weights = optimize_portfolio(
        returns,
        cov_matrix,
        method=opt_method,
        risk_tolerance=risk_tolerance,
        risk_free_rate=risk_free_rate
    )
    progress_bar.progress(60)
    
    # Calculate performance metrics
    status_text.text("Calculating portfolio metrics...")
    metrics = calculate_portfolio_metrics(opt_weights, returns, cov_matrix, risk_free_rate)
    portfolio_data = analyze_optimized_portfolio(price_data, opt_weights, benchmark)
    progress_bar.progress(80)
    
    # Run Monte Carlo simulation
    status_text.text("Running Monte Carlo simulation...")
    simulation_results = simulate_portfolios(returns, cov_matrix, n_simulations)
    progress_bar.progress(100)
    
    # Clear status indicators
    status_text.empty()
    progress_bar.empty()
    
    # Display results
    st.header("📈 Portfolio Optimization Results")
    
    # Display key metrics dashboard
    display_metrics_dashboard(metrics, portfolio_data)
    
    # Create tabs for different visualizations
    tabs = st.tabs([
        "Portfolio Allocation", 
        "Performance", 
        "Efficient Frontier",
        "Drawdown Analysis",
        "Rolling Metrics",
        "Detailed Statistics"
    ])
    # In the Portfolio Allocation tab, fix the DataFrame creation
with tabs[0]:  # Portfolio Allocation
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Portfolio weights visualization
        weights_fig = plot_portfolio_weights(tickers, opt_weights, f"Optimized Portfolio: {opt_method}")
        st.pyplot(weights_fig)
    
    with col2:
        # Portfolio weights table - with length validation
        st.subheader("Asset Allocation")
        
        # Ensure tickers and weights have the same length
        min_len = min(len(tickers), len(opt_weights))
        display_tickers = tickers[:min_len]
        display_weights = opt_weights[:min_len]
        
        # Now create DataFrame with validated arrays
        weight_df = pd.DataFrame({
            "Ticker": display_tickers,
            "Weight (%)": display_weights * 100
        })
        weight_df = weight_df.sort_values("Weight (%)", ascending=False)
        
        # Format table
        st.dataframe(
            weight_df[weight_df["Weight (%)"] > 0.1].style.format({
                "Weight (%)": "{:.2f}%"
            }),
            use_container_width=True
        )
        
        # Add download button
        csv = weight_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Weights (CSV)",
            data=csv,
            file_name="optimized_portfolio.csv",
            mime="text/csv"
        )
    
    with tabs[1]:  # Performance
        # Show portfolio performance
        st.subheader("Portfolio Performance")
        
        # Select assets to show in performance chart
        assets_to_show = None
        if include_individual_assets:
            assets_to_show = price_data
        
        # Plot performance
        perf_fig = plot_performance_comparison(portfolio_data, assets_to_show)
        st.pyplot(perf_fig)
        
        # Show performance statistics
        if portfolio_data['benchmark_returns'] is not None:
            correlation = portfolio_data['correlation_with_benchmark']
            st.info(f"Correlation with benchmark: {correlation:.2f}")
    
    with tabs[2]:  # Efficient Frontier
        st.subheader("Efficient Frontier")
        frontier_fig = plot_efficient_frontier(
            simulation_results, 
            metrics, 
            opt_weights, 
            returns, 
            opt_method
        )
        st.pyplot(frontier_fig)
        
        # Add explanation
        st.markdown("""
        **Chart Explanation:**
        - **Blue dots**: Simulated random portfolios
        - **Red dot**: Your optimized portfolio
        - **Colored dots**: Individual assets with significant weight
        - **Color scale**: Sharpe ratio (higher is better)
        """)
    
    with tabs[3]:  # Drawdown Analysis
        st.subheader("Drawdown Analysis")
        
        # Plot drawdown
        drawdown_fig = plot_drawdown(portfolio_data['drawdown'])
        st.pyplot(drawdown_fig)
        
        # Add statistics
        max_drawdown = portfolio_data['drawdown'].min() * 100
        recovery_days = None
        
        # Try to calculate recovery time
        try:
            min_idx = portfolio_data['drawdown'].idxmin()
            after_min = portfolio_data['drawdown'].loc[min_idx:]
            recovery = after_min[after_min >= 0]
            
            if not recovery.empty:
                recovery_days = (recovery.index[0] - min_idx).days
        except:
            pass
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Maximum Drawdown", f"{max_drawdown:.2f}%")
        with col2:
            if recovery_days:
                st.metric("Recovery Period", f"{recovery_days} days")
            else:
                st.metric("Recovery Period", "Not recovered")
    
    with tabs[4]:  # Rolling Metrics
        st.subheader("Rolling Performance Metrics")
        
        # Plot rolling metrics
        rolling_fig = plot_rolling_metrics(portfolio_data)
        if rolling_fig:
            st.pyplot(rolling_fig)
        else:
            st.info("Insufficient data for rolling metrics. Need at least 1 year of data.")
    
    with tabs[5]:  # Detailed Statistics
        st.subheader("Detailed Statistics")
        
        # Create columns for different statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Return Statistics")
            
            # Calculate various return statistics
            annual_return = metrics['annual_return'] * 100
            monthly_returns = portfolio_data['returns'].resample('M').apply(
                lambda x: (1 + x).prod() - 1
            ) * 100
            
            positive_months = (monthly_returns > 0).sum()
            total_months = len(monthly_returns)
            
            if total_months > 0:
                win_rate = positive_months / total_months * 100
            else:
                win_rate = 0
            
            # Display statistics
            stats_df = pd.DataFrame({
                "Metric": [
                    "Annual Return",
                    "Monthly Avg Return",
                    "Best Month",
                    "Worst Month",
                    "Positive Months",
                    "Win Rate"
                ],
                "Value": [
                    f"{annual_return:.2f}%",
                    f"{monthly_returns.mean():.2f}%",
                    f"{monthly_returns.max():.2f}%",
                    f"{monthly_returns.min():.2f}%",
                    f"{positive_months} / {total_months}",
                    f"{win_rate:.1f}%"
                ]
            })
            
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("Risk Statistics")
            
            # Calculate various risk statistics
            annual_vol = metrics['annual_volatility'] * 100
            var_95 = portfolio_data['var_95'] * 100
            cvar_95 = portfolio_data['cvar_95'] * 100 if portfolio_data['cvar_95'] else 0
            max_drawdown = portfolio_data['drawdown'].min() * 100
            
            # Calculate additional metrics if benchmark available
            beta = metrics['beta']
            tracking_error = None
            
            if portfolio_data['benchmark_returns'] is not None:
                # Calculate tracking error
                active_returns = portfolio_data['returns'] - portfolio_data['benchmark_returns']
                tracking_error = active_returns.std() * np.sqrt(252) * 100
            
            # Display statistics
            risk_df = pd.DataFrame({
                "Metric": [
                    "Annual Volatility",
                    "Daily VaR (95%)",
                    "Daily CVaR (95%)",
                    "Maximum Drawdown",
                    "Beta" if beta is not None else "Diversification Ratio",
                    "Tracking Error" if tracking_error is not None else "Sortino Ratio"
                ],
                "Value": [
                    f"{annual_vol:.2f}%",
                    f"{var_95:.2f}%",
                    f"{cvar_95:.2f}%",
                    f"{max_drawdown:.2f}%",
                    f"{beta:.2f}" if beta is not None else f"{metrics['diversification_ratio']:.2f}",
                    f"{tracking_error:.2f}%" if tracking_error is not None else f"{metrics['sortino_ratio']:.2f}"
                ]
            })
            
            st.dataframe(risk_df, use_container_width=True, hide_index=True)
        
        # Show correlation matrix
        st.subheader("Asset Correlation Matrix")
        
        # Generate correlation matrix
        corr_matrix = returns.corr()
        
        # Plot correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            fmt=".2f",
            linewidths=0.5,
            ax=ax
        )
        ax.set_title("Correlation Between Assets")
        st.pyplot(fig)
        
        # Show rebalancing information
        st.subheader("Portfolio Maintenance")
        st.info("""
        **Rebalancing Recommendation:**
        - For long-term investments: Quarterly rebalancing is recommended to maintain target weights
        - For higher volatility assets: Consider monthly rebalancing
        - Set threshold-based rebalancing: When individual positions drift more than 5% from target weights
        """)

# If not running optimization, show instructions
if not run_optimization:  # This is the missing if statement
    st.info("👈 Configure your portfolio settings in the sidebar and click 'Optimize Portfolio' to generate results.")
    
    # Show example portfolio allocation
    st.subheader("Example Portfolio Optimization")
    st.image("https://miro.medium.com/max/1400/1*_6kV_NJ0LdjiQCOM62vBlA.webp", 
             caption="Example efficient frontier visualization")
    
    # Add educational content
    st.markdown("""
    ## Understanding Portfolio Optimization
    
    Modern portfolio theory focuses on building diversified portfolios that optimize the risk-return tradeoff. 
    The key principles include:
    
    - **Diversification**: Not putting all eggs in one basket
    - **Risk-Return Tradeoff**: Higher expected returns generally come with higher risk
    - **Efficient Frontier**: The set of optimal portfolios that offer the highest expected return for a given level of risk
    
    This app allows you to explore different optimization strategies and visualize their potential impact on your portfolio.
    """)
