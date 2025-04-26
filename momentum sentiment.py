"""
Multi-Factor Portfolio Optimization System

This script implements a comprehensive multi-factor portfolio optimization system with:
1. Factor calculation (Value, Momentum, Quality, Size)
2. Portfolio optimization with risk constraints
3. Performance analysis and attribution

Author: Muchiri Kahwai
Date: April 2025
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(page_title="Multi-Factor Portfolio Optimizer", page_icon="📈", layout="wide")

# Add custom CSS
st.markdown("""
<style>
.metric-card {
    background-color: #f9f9f9;
    border-radius: 5px;
    border-left: 5px solid #4CAF50;
    padding: 15px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
.metric-card.negative {
    border-left: 5px solid #f44336;
}
.metric-value {
    font-size: 24px;
    font-weight: bold;
}
.metric-label {
    font-size: 14px;
    color: #666;
}
</style>
""", unsafe_allow_html=True)

# App title
st.title("Multi-Factor Portfolio Optimizer")
st.markdown("Build and backtest a portfolio based on established factors: Value, Momentum, Quality, and Size.")

# ------ Data Loading and Processing Functions ------

@st.cache_data(ttl=24*3600)
def load_sp500_tickers():
    """Load S&P 500 tickers and sector information"""
    try:
        # Get the list of S&P 500 companies from Wikipedia
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        df = df[['Symbol', 'GICS Sector']]
        df = df.rename(columns={'Symbol': 'ticker', 'GICS Sector': 'sector'})
        return df
    except:
        # Fallback to a smaller list
        sectors = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'AMZN': 'Consumer Discretionary',
            'GOOGL': 'Communication Services', 'META': 'Communication Services',
            'BRK-B': 'Financials', 'JPM': 'Financials', 'JNJ': 'Healthcare',
            'V': 'Financials', 'PG': 'Consumer Staples', 'UNH': 'Healthcare',
            'NVDA': 'Technology', 'HD': 'Consumer Discretionary', 'MA': 'Financials',
            'DIS': 'Communication Services'
        }
        df = pd.DataFrame(list(sectors.items()), columns=['ticker', 'sector'])
        return df

@st.cache_data(ttl=24*3600)
def get_price_data(tickers, start_date, end_date):
    """Fetch historical price data for the given tickers"""
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    return data

@st.cache_data(ttl=24*3600)
def get_fundamental_data(tickers):
    """Fetch fundamental data for factor calculations"""
    fundamental_data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            fundamental_data[ticker] = {
                'balance_sheet': stock.balance_sheet,
                'income_stmt': stock.income_stmt,
                'cash_flow': stock.cashflow,
                'info': stock.info
            }
        except Exception as e:
            continue
    
    return fundamental_data

# ------ Factor Calculation Functions ------

def calculate_value_factor(prices, fundamental_data, tickers):
    """Calculate value factor based on P/B, P/E, and EV/EBITDA ratios"""
    value_metrics = pd.DataFrame(index=tickers)
    
    for ticker in tickers:
        if ticker not in fundamental_data:
            continue
            
        try:
            # Get the most recent price
            latest_price = prices['Adj Close'][ticker].iloc[-1]
            
            # Get fundamental data
            fund_data = fundamental_data[ticker]
            bs = fund_data['balance_sheet']
            income = fund_data['income_stmt']
            
            # Calculate P/B ratio
            if 'Total Assets' in bs.index and 'Total Liabilities Net Minority Interest' in bs.index:
                total_assets = bs.loc['Total Assets'].iloc[0]
                total_liabilities = bs.loc['Total Liabilities Net Minority Interest'].iloc[0]
                book_value = total_assets - total_liabilities
                shares_outstanding = fund_data['info'].get('sharesOutstanding', None)
                
                if shares_outstanding and book_value > 0:
                    book_value_per_share = book_value / shares_outstanding
                    pb_ratio = latest_price / book_value_per_share
                    value_metrics.loc[ticker, 'P/B'] = pb_ratio
            
            # Calculate P/E ratio
            if 'Net Income' in income.index:
                net_income = income.loc['Net Income'].iloc[0]
                if shares_outstanding and net_income > 0:
                    eps = net_income / shares_outstanding
                    pe_ratio = latest_price / eps
                    value_metrics.loc[ticker, 'P/E'] = pe_ratio
                
        except Exception as e:
            continue
    
    # Calculate Z-scores for each metric (lower is better for value)
    z_scores = pd.DataFrame(index=value_metrics.index)
    for column in value_metrics.columns:
        if value_metrics[column].notna().sum() > 0:
            mean = value_metrics[column].mean()
            std = value_metrics[column].std()
            if std > 0:
                z_scores[column] = -(value_metrics[column] - mean) / std
    
    # Combine Z-scores to get final value factor
    value_factor = z_scores.mean(axis=1)
    return value_factor

def calculate_momentum_factor(prices, tickers, lookback_months=12, skip_recent_month=True):
    """Calculate momentum factor based on price returns"""
    momentum = pd.Series(index=tickers, dtype=float)
    
    # Get Adjusted Close prices
    adj_close = prices['Adj Close']
    
    # Calculate lookback period
    if skip_recent_month:
        end_idx = -21  # Skip most recent month (approximately 21 trading days)
    else:
        end_idx = -1
        
    start_idx = min(-(lookback_months * 21), -252)  # Limit to ~1 year max lookback
    
    for ticker in tickers:
        try:
            # Calculate price return
            start_price = adj_close[ticker].iloc[start_idx]
            end_price = adj_close[ticker].iloc[end_idx]
            price_return = (end_price / start_price) - 1
            momentum[ticker] = price_return
        except:
            continue
    
    # Calculate Z-scores
    mean_return = momentum.mean()
    std_return = momentum.std()
    if std_return > 0:
        momentum_factor = (momentum - mean_return) / std_return
    else:
        momentum_factor = pd.Series(0, index=momentum.index)
    
    return momentum_factor

def calculate_quality_factor(fundamental_data, tickers):
    """Calculate quality factor based on ROE, debt/equity, and earnings stability"""
    quality_metrics = pd.DataFrame(index=tickers)
    
    for ticker in tickers:
        if ticker not in fundamental_data:
            continue
            
        try:
            # Get fundamental data
            fund_data = fundamental_data[ticker]
            bs = fund_data['balance_sheet']
            income = fund_data['income_stmt']
            
            # Calculate ROE (higher is better)
            if 'Net Income' in income.index and 'Stockholders Equity' in bs.index:
                net_income = income.loc['Net Income'].iloc[0]
                total_equity = bs.loc['Stockholders Equity'].iloc[0]
                if total_equity > 0:
                    roe = net_income / total_equity
                    quality_metrics.loc[ticker, 'ROE'] = roe
            
            # Calculate Debt to Equity ratio (lower is better)
            if 'Total Debt' in bs.index and 'Stockholders Equity' in bs.index:
                total_debt = bs.loc['Total Debt'].iloc[0]
                if total_equity > 0:
                    debt_to_equity = total_debt / total_equity
                    quality_metrics.loc[ticker, 'Debt/Equity'] = debt_to_equity
                
        except Exception as e:
            continue
    
    # Calculate Z-scores for each metric
    z_scores = pd.DataFrame(index=quality_metrics.index)
    for column in quality_metrics.columns:
        if quality_metrics[column].notna().sum() > 0:
            mean = quality_metrics[column].mean()
            std = quality_metrics[column].std()
            if std > 0:
                if column in ['ROE']:  # Higher is better
                    z_scores[column] = (quality_metrics[column] - mean) / std
                else:  # Lower is better (Debt/Equity)
                    z_scores[column] = -(quality_metrics[column] - mean) / std
    
    # Combine Z-scores to get final quality factor
    quality_factor = z_scores.mean(axis=1)
    return quality_factor

def calculate_size_factor(prices, tickers):
    """Calculate size factor based on market capitalization"""
    market_caps = pd.Series(index=tickers, dtype=float)
    
    # Get most recent prices
    latest_prices = prices['Adj Close'].iloc[-1]
    
    for ticker in tickers:
        try:
            # Get stock info
            stock = yf.Ticker(ticker)
            shares_outstanding = stock.info.get('sharesOutstanding', None)
            
            # Calculate market cap
            if shares_outstanding and ticker in latest_prices:
                market_cap = latest_prices[ticker] * shares_outstanding
                market_caps[ticker] = market_cap
        except:
            continue
    
    # Calculate Z-scores (negative because smaller cap historically outperforms)
    mean_market_cap = market_caps.mean()
    std_market_cap = market_caps.std()
    if std_market_cap > 0:
        size_factor = -(market_caps - mean_market_cap) / std_market_cap
    else:
        size_factor = pd.Series(0, index=market_caps.index)
    
    return size_factor

def combine_factors(factor_data, weights):
    """Combine individual factors using specified weights"""
    # Ensure all factors have the same index
    common_tickers = factor_data[list(factor_data.keys())[0]].index
    for factor_name in factor_data.keys():
        common_tickers = common_tickers.intersection(factor_data[factor_name].index)
    
    # Create dataframe for combined factors
    combined_factors = pd.DataFrame(index=common_tickers)
    
    # Add individual factors
    for factor_name, factor_series in factor_data.items():
        combined_factors[factor_name] = factor_series.loc[common_tickers]
    
    # Calculate weighted sum
    combined_factors['combined_score'] = 0
    for factor_name, weight in weights.items():
        if factor_name in combined_factors.columns:
            combined_factors['combined_score'] += weight * combined_factors[factor_name]
    
    return combined_factors

# ------ Portfolio Optimization Functions ------

def get_risk_model(prices, method='ledoit_wolf'):
    """Generate a risk model based on historical returns"""
    # Calculate daily returns
    returns = prices['Adj Close'].pct_change().dropna()
    
    if method == 'sample':
        # Simple sample covariance
        return returns.cov()
    elif method == 'ledoit_wolf':
        # Ledoit-Wolf shrinkage estimator
        lw = LedoitWolf().fit(returns)
        cov_matrix = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
        return cov_matrix
    else:
        # Default to sample covariance
        return returns.cov()

def optimize_portfolio(returns, cov_matrix, factor_scores, sector_data, constraints):
    """Optimize portfolio weights using mean-variance optimization"""
    n = len(returns)
    expected_returns = factor_scores.values
    
    def objective(weights):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
        return -portfolio_return + constraints['risk_aversion'] * portfolio_risk
    
    # Initial guess - equal weights
    initial_weights = np.ones(n) / n
    
    # Constraints
    bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n)]
    
    # Sum of weights = 1
    constraints_list = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
    
    # Long-only constraint
    if constraints['long_only']:
        bounds = [(max(0, constraints['min_weight']), constraints['max_weight']) for _ in range(n)]
    
    # Sector constraints
    if constraints['sector_constraints'] and sector_data is not None:
        unique_sectors = sector_data['sector'].unique()
        tickers = returns.index
        
        for sector in unique_sectors:
            # Get indices of tickers in this sector
            sector_indices = [i for i, ticker in enumerate(tickers) 
                            if ticker in sector_data.index and sector_data.loc[ticker, 'sector'] == sector]
            
            if sector_indices:
                # Create a constraint function for this sector
                def sector_constraint(weights, indices=sector_indices, max_weight=constraints['max_sector_weight']):
                    sector_weight = sum(weights[i] for i in indices)
                    return max_weight - sector_weight
                
                constraints_list.append({'type': 'ineq', 'fun': sector_constraint})
    
    # Solve the optimization problem
    try:
        result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, 
                         constraints=constraints_list, options={'maxiter': 1000})
        
        if result.success:
            optimized_weights = {ticker: max(weight, 0) for ticker, weight in zip(returns.index, result.x)}
            # Normalize weights to ensure they sum to 1
            weight_sum = sum(optimized_weights.values())
            optimized_weights = {ticker: weight / weight_sum for ticker, weight in optimized_weights.items()}
            return optimized_weights
        else:
            st.warning(f"Optimization did not converge: {result.message}")
            # Return equal weights as fallback
            return {ticker: 1/n for ticker in returns.index}
    except Exception as e:
        st.error(f"Optimization error: {str(e)}")
        # Return equal weights as fallback
        return {ticker: 1/n for ticker in returns.index}

def backtest_portfolio(prices, weights_history, rebalance_dates):
    """Perform a backtest of the portfolio using historical weights"""
    # Get daily returns
    daily_returns = prices['Adj Close'].pct_change().dropna()
    
    # Create a dataframe to store portfolio returns
    portfolio_returns = pd.Series(index=daily_returns.index, dtype=float)
    
    # Track current weights
    current_weights = None
    
    # Fill in portfolio returns
    for date in portfolio_returns.index:
        # Check if we need to rebalance
        if date in rebalance_dates:
            current_weights = weights_history[date]
        
        # Skip if we don't have weights yet
        if current_weights is None:
            continue
        
        # Calculate portfolio return for the day
        if date in daily_returns.index:
            # Get only the tickers that exist in both weights and returns
            common_tickers = set(current_weights.keys()) & set(daily_returns.columns)
            
            # Calculate weighted return
            day_return = 0
            weight_sum = sum(current_weights[ticker] for ticker in common_tickers)
            
            if weight_sum > 0:  # Avoid division by zero
                for ticker in common_tickers:
                    normalized_weight = current_weights[ticker] / weight_sum
                    if pd.notna(daily_returns.loc[date, ticker]):
                        day_return += normalized_weight * daily_returns.loc[date, ticker]
            
            portfolio_returns[date] = day_return
    
    # Calculate cumulative returns
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(portfolio_returns)
    
    return portfolio_returns, portfolio_cumulative, metrics

def calculate_performance_metrics(returns):
    """Calculate key performance metrics for a return series"""
    metrics = {}
    
    # Annualized return
    metrics['annualized_return'] = returns.mean() * 252
    
    # Annualized volatility
    metrics['annualized_volatility'] = returns.std() * np.sqrt(252)
    
    # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
    metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['annualized_volatility'] if metrics['annualized_volatility'] > 0 else 0
    
    # Maximum drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    metrics['max_drawdown'] = drawdown.min()
    
    # Win rate
    metrics['win_rate'] = (returns > 0).mean()
    
    return metrics

# ------ Main Application Function ------

def run_app():
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Universe selection
    with st.sidebar.expander("Investment Universe", expanded=True):
        universe_option = st.radio(
            "Select universe",
            options=["S&P 500", "Custom Tickers"],
            index=0
        )
        
        if universe_option == "Custom Tickers":
            custom_tickers = st.text_area(
                "Enter tickers (comma-separated)",
                value="AAPL, MSFT, AMZN, GOOGL, META, BRK-B, JNJ, JPM, V, PG"
            )
            tickers = [ticker.strip() for ticker in custom_tickers.split(',')]
        else:
            # Use S&P 500 tickers
            sp500_df = load_sp500_tickers()
            
            # Option to filter by sector
            sector_filter = st.multiselect(
                "Filter by sector",
                options=sorted(sp500_df['sector'].unique().tolist()),
                default=[]
            )
            
            if sector_filter:
                filtered_df = sp500_df[sp500_df['sector'].isin(sector_filter)]
                tickers = filtered_df['ticker'].tolist()
            else:
                tickers = sp500_df['ticker'].tolist()
            
            # Limit number of tickers for performance
            max_tickers = st.slider(
                "Maximum number of tickers",
                min_value=10,
                max_value=100,
                value=30,
                step=5
            )
            
            if len(tickers) > max_tickers:
                tickers = tickers[:max_tickers]
    
    # Backtest period
    with st.sidebar.expander("Backtest Period", expanded=True):
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365*3)  # 3 years by default
        
        start_date = st.date_input(
            "Start date",
            value=start_date,
            max_value=end_date - timedelta(days=30)
        )
        
        end_date = st.date_input(
            "End date",
            value=end_date,
            min_value=start_date + timedelta(days=30)
        )
        
        # Rebalancing frequency
        rebalance_freq = st.selectbox(
            "Rebalancing frequency",
            options=["Monthly", "Quarterly", "Semi-Annually", "Annually"],
            index=1
        )
    
    # Factor weights
    with st.sidebar.expander("Factor Weights", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            value_weight = st.slider("Value", 0.0, 1.0, 0.25, 0.05)
            momentum_weight = st.slider("Momentum", 0.0, 1.0, 0.25, 0.05)
        with col2:
            quality_weight = st.slider("Quality", 0.0, 1.0, 0.25, 0.05)
            size_weight = st.slider("Size", 0.0, 1.0, 0.25, 0.05)
        
        # Normalize weights to sum to 1
        total_weight = value_weight + momentum_weight + quality_weight + size_weight
        if total_weight > 0:
            factor_weights = {
                'value_factor': value_weight / total_weight,
                'momentum_factor': momentum_weight / total_weight,
                'quality_factor': quality_weight / total_weight,
                'size_factor': size_weight / total_weight
            }
        else:
            factor_weights = {
                'value_factor': 0.25,
                'momentum_factor': 0.25,
                'quality_factor': 0.25,
                'size_factor': 0.25
            }
    
    # Portfolio constraints
    with st.sidebar.expander("Portfolio Constraints", expanded=True):
        long_only = st.checkbox("Long-only", value=True)
        
        col1, col2 = st.columns(2)
        with col1:
            min_weight = st.slider("Min weight (%)", 0.0, 5.0, 0.0, 0.5) / 100
            max_weight = st.slider("Max weight (%)", 1.0, 20.0, 5.0, 0.5) / 100
        
        with col2:
            sector_constraints = st.checkbox("Sector constraints", value=True)
            max_sector_weight = st.slider("Max sector (%)", 10.0, 50.0, 30.0, 5.0) / 100
        
        risk_aversion = st.slider(
            "Risk aversion", 
            0.1, 10.0, 1.0, 0.1,
            help="Higher values prioritize risk reduction over returns"
        )
        
        risk_model_method = st.selectbox(
            "Risk model",
            options=["Sample Covariance", "Ledoit-Wolf Shrinkage"],
            index=1
        )
        risk_model_method = 'sample' if risk_model_method == "Sample Covariance" else 'ledoit_wolf'
    
    # Run backtest button
    run_backtest = st.sidebar.button("Run Backtest", type="primary", use_container_width=True)
    
    # Create tabs for results
    tab1, tab2, tab3 = st.tabs(["Portfolio Performance", "Factor Analysis", "Holdings"])
    
    # Main backtest logic
    if run_backtest:
        with st.spinner("Running backtest..."):
            # Load price data
            prices = get_price_data(tickers, start_date, end_date)
            
            # Generate rebalance dates
            rebalance_dates = []
            current_date = pd.Timestamp(start_date)
            
            while current_date <= pd.Timestamp(end_date):
                rebalance_dates.append(current_date)
                
                if rebalance_freq == "Monthly":
                    current_date = current_date + pd.DateOffset(months=1)
                elif rebalance_freq == "Quarterly":
                    current_date = current_date + pd.DateOffset(months=3)
                elif rebalance_freq == "Semi-Annually":
                    current_date = current_date + pd.DateOffset(months=6)
                else:  # Annually
                    current_date = current_date + pd.DateOffset(years=1)
            
            # Align with trading days
            trading_dates = prices.index
            aligned_dates = []
            
            for date in rebalance_dates:
                # Find the closest trading day on or after the target date
                if date in trading_dates:
                    aligned_dates.append(date)
                else:
                    future_dates = trading_dates[trading_dates >= date]
                    if not future_dates.empty:
                        aligned_dates.append(future_dates[0])
            
            rebalance_dates = aligned_dates
            
            # Load sector data if using S&P 500
            if universe_option == "S&P 500":
                sector_data = load_sp500_tickers().set_index('ticker')
            else:
                sector_data = None
            
            # Load fundamental data
            fundamental_data = get_fundamental_data(tickers)
            
            # Initialize weights history
            weights_history = {}
            previous_weights = None
            
            # Run backtest for each rebalance date
            for rebalance_date in rebalance_dates:
                # Get data up to rebalance date
                historical_prices = prices.loc[:rebalance_date]
                
                # Calculate factors
                value_factor = calculate_value_factor(historical_prices, fundamental_data, tickers)
                momentum_factor = calculate_momentum_factor(historical_prices, tickers)
                quality_factor = calculate_quality_factor(fundamental_data, tickers)
                size_factor = calculate_size_factor(historical_prices, tickers)
                
                # Combine factors
                factor_data = {
                    'value_factor': value_factor,
                    'momentum_factor': momentum_factor,
                    'quality_factor': quality_factor,
                    'size_factor': size_factor
                }
                
                combined_factors = combine_factors(factor_data, factor_weights)
                
                # Get risk model
                cov_matrix = get_risk_model(historical_prices, method=risk_model_method)
                
                # Set up constraints
                constraints = {
                    'long_only': long_only,
                    'min_weight': min_weight,
                    'max_weight': max_weight,
                    'sector_constraints': sector_constraints,
                    'max_sector_weight': max_sector_weight,
                    'risk_aversion': risk_aversion,
                    'previous_weights': previous_weights
                }
                
                # Optimize portfolio
                weights = optimize_portfolio(
                    combined_factors['combined_score'],
                    cov_matrix,
                    combined_factors['combined_score'],
                    sector_data,
                    constraints
                )
                
                # Store weights
                weights_history[rebalance_date] = weights
                previous_weights = weights
            
            # Backtest portfolio
            portfolio_returns, portfolio_cumulative, metrics = backtest_portfolio(
                prices, weights_history, rebalance_dates
            )
            
            # Get benchmark returns (S&P 500)
            spy_data = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
            benchmark_returns = spy_data['Adj Close'].pct_change().dropna()
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            benchmark_metrics = calculate_performance_metrics(benchmark_returns)
            
            # Display results in each tab
            with tab1:
                st.subheader("Portfolio Performance")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card {'negative' if metrics['annualized_return'] < 0 else ''}">
                        <div class="metric-value">{metrics['annualized_return']*100:.2f}%</div>
                        <div class="metric-label">Annual Return</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{metrics['sharpe_ratio']:.2f}</div>
                        <div class="metric-label">Sharpe Ratio</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card negative">
                        <div class="metric-value">{metrics['max_drawdown']*100:.2f}%</div>
                        <div class="metric-label">Max Drawdown</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{metrics['annualized_volatility']*100:.2f}%</div>
                        <div class="metric-label">Volatility</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Performance chart
                st.subheader("Cumulative Returns")
                chart_data = pd.DataFrame({
                    'Portfolio': portfolio_cumulative,
                    'S&P 500': benchmark_cumulative.reindex(portfolio_cumulative.index)
                })
                
                fig = px.line(chart_data)
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return",
                    legend_title="",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Drawdown chart
                st.subheader("Portfolio Drawdown")
                running_max = portfolio_cumulative.cummax()
                drawdown = ((portfolio_cumulative / running_max) - 1) * 100
                
                fig = px.area(
                    drawdown,
                    color_discrete_sequence=['rgba(239, 85, 59, 0.7)']
                )
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Drawdown (%)",
                    showlegend=False,
                    yaxis=dict(autorange="reversed")  # Invert
