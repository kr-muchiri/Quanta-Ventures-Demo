"""
Multi-Factor Portfolio Optimization System
A streamlined system for factor-based portfolio construction and backtesting.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from datetime import datetime, timedelta

# Setup page
st.set_page_config(page_title="Multi-Factor Portfolio Optimizer", layout="wide")
st.title("Multi-Factor Portfolio Optimizer")
st.markdown("Optimize and backtest portfolios using Value, Momentum, Quality, and Size factors")

# Add simple styling
st.markdown("""
<style>
.metric-card {background-color:#f9f9f9; border-left:5px solid #4CAF50; padding:15px; border-radius:5px;}
.metric-card.negative {border-left:5px solid #f44336;}
.metric-value {font-size:24px; font-weight:bold;}
.metric-label {font-size:14px; color:#666;}
</style>
""", unsafe_allow_html=True)

# ---------- Data Loading Functions ----------
@st.cache_data(ttl=24*3600)
def load_sp500_tickers():
    """Load S&P 500 tickers and sector information"""
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0][['Symbol', 'GICS Sector']]
        return df.rename(columns={'Symbol': 'ticker', 'GICS Sector': 'sector'})
    except:
        # Fallback to a small list
        sectors = {'AAPL': 'Technology', 'MSFT': 'Technology', 'AMZN': 'Consumer Discretionary',
                  'GOOGL': 'Communication', 'META': 'Communication', 'BRK-B': 'Financials',
                  'JPM': 'Financials', 'JNJ': 'Healthcare', 'NVDA': 'Technology'}
        return pd.DataFrame(list(sectors.items()), columns=['ticker', 'sector'])

@st.cache_data(ttl=24*3600)
def get_price_data(tickers, start_date, end_date):
    """Fetch historical price data"""
    return yf.download(tickers, start=start_date, end=end_date, progress=False)

@st.cache_data(ttl=24*3600)
def get_fundamental_data(tickers):
    """Fetch fundamental data for factor calculations"""
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            data[ticker] = {
                'balance_sheet': stock.balance_sheet,
                'income_stmt': stock.income_stmt,
                'info': stock.info
            }
        except:
            continue
    return data

# ---------- Factor Calculation Functions ----------
def calculate_value_factor(prices, fundamental_data, tickers):
    """Calculate value factor based on P/B and P/E ratios"""
    value_metrics = pd.DataFrame(index=tickers)
    
    for ticker in tickers:
        if ticker not in fundamental_data:
            continue
            
        try:
            price = prices['Adj Close'][ticker].iloc[-1]
            fund_data = fundamental_data[ticker]
            bs = fund_data['balance_sheet']
            income = fund_data['income_stmt']
            shares = fund_data['info'].get('sharesOutstanding', None)
            
            # P/B ratio
            if 'Total Assets' in bs.index and 'Total Liabilities Net Minority Interest' in bs.index and shares:
                book_value = (bs.loc['Total Assets'].iloc[0] - bs.loc['Total Liabilities Net Minority Interest'].iloc[0]) / shares
                value_metrics.loc[ticker, 'P/B'] = price / book_value if book_value > 0 else np.nan
            
            # P/E ratio
            if 'Net Income' in income.index and shares:
                eps = income.loc['Net Income'].iloc[0] / shares
                value_metrics.loc[ticker, 'P/E'] = price / eps if eps > 0 else np.nan
                
        except:
            continue
    
    # Z-scores (lower values = better value)
    z_scores = pd.DataFrame(index=value_metrics.index)
    for col in value_metrics.columns:
        if value_metrics[col].notna().sum() > 0:
            mean, std = value_metrics[col].mean(), value_metrics[col].std()
            if std > 0:
                z_scores[col] = -(value_metrics[col] - mean) / std
    
    # Combine scores
    return z_scores.mean(axis=1)

def calculate_momentum_factor(prices, tickers, lookback_months=12, skip_recent=True):
    """Calculate momentum factor based on price returns"""
    momentum = pd.Series(index=tickers)
    adj_close = prices['Adj Close']
    
    # Calculate lookback periods
    if skip_recent:
        end_idx = -21  # Skip most recent month
    else:
        end_idx = -1
    start_idx = min(-(lookback_months * 21), -252)
    
    for ticker in tickers:
        try:
            # Calculate return
            start_price = adj_close[ticker].iloc[start_idx]
            end_price = adj_close[ticker].iloc[end_idx]
            momentum[ticker] = (end_price / start_price) - 1
        except:
            continue
    
    # Z-score
    mean, std = momentum.mean(), momentum.std()
    return (momentum - mean) / std if std > 0 else pd.Series(0, index=momentum.index)

def calculate_quality_factor(fundamental_data, tickers):
    """Calculate quality factor based on ROE and debt/equity"""
    quality_metrics = pd.DataFrame(index=tickers)
    
    for ticker in tickers:
        if ticker not in fundamental_data:
            continue
            
        try:
            fund_data = fundamental_data[ticker]
            bs = fund_data['balance_sheet']
            income = fund_data['income_stmt']
            
            # ROE (higher is better)
            if 'Net Income' in income.index and 'Stockholders Equity' in bs.index:
                net_income = income.loc['Net Income'].iloc[0]
                equity = bs.loc['Stockholders Equity'].iloc[0]
                if equity > 0:
                    quality_metrics.loc[ticker, 'ROE'] = net_income / equity
            
            # Debt/Equity (lower is better)
            if 'Total Debt' in bs.index and 'Stockholders Equity' in bs.index:
                debt = bs.loc['Total Debt'].iloc[0] 
                if equity > 0:
                    quality_metrics.loc[ticker, 'Debt/Equity'] = debt / equity
                
        except:
            continue
    
    # Z-scores
    z_scores = pd.DataFrame(index=quality_metrics.index)
    for col in quality_metrics.columns:
        if quality_metrics[col].notna().sum() > 0:
            mean, std = quality_metrics[col].mean(), quality_metrics[col].std()
            if std > 0:
                # Higher ROE is better, lower Debt/Equity is better
                sign = 1 if col == 'ROE' else -1
                z_scores[col] = sign * (quality_metrics[col] - mean) / std
    
    return z_scores.mean(axis=1)

def calculate_size_factor(prices, tickers):
    """Calculate size factor based on market cap (smaller = better)"""
    market_caps = pd.Series(index=tickers)
    latest_prices = prices['Adj Close'].iloc[-1]
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            shares = stock.info.get('sharesOutstanding', None)
            if shares and ticker in latest_prices:
                market_caps[ticker] = latest_prices[ticker] * shares
        except:
            continue
    
    # Z-score (negative because smaller is better)
    mean, std = market_caps.mean(), market_caps.std()
    return -(market_caps - mean) / std if std > 0 else pd.Series(0, index=market_caps.index)

def combine_factors(factor_data, weights):
    """Combine factors using specified weights"""
    # Find common tickers across all factors
    common_tickers = set.intersection(*[set(f.index) for f in factor_data.values()])
    
    # Create combined score dataframe
    combined = pd.DataFrame(index=common_tickers)
    for name, factor in factor_data.items():
        combined[name] = factor.loc[common_tickers]
    
    # Apply weights
    combined['combined_score'] = 0
    for name, weight in weights.items():
        if name in combined.columns:
            combined['combined_score'] += weight * combined[name]
    
    return combined

# ---------- Portfolio Optimization ----------
def get_risk_model(prices, method='ledoit_wolf'):
    """Generate a risk model from historical returns"""
    returns = prices['Adj Close'].pct_change().dropna()
    
    if method == 'ledoit_wolf':
        # Ledoit-Wolf shrinkage
        lw = LedoitWolf().fit(returns)
        return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
    else:
        # Sample covariance
        return returns.cov()

def optimize_portfolio(returns, cov_matrix, factor_scores, sector_data, constraints):
    """Optimize portfolio weights"""
    n = len(returns)
    expected_returns = factor_scores.values
    
    # Objective function: maximize return - risk_aversion * risk
    def objective(weights):
        port_return = np.dot(weights, expected_returns)
        port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
        return -port_return + constraints['risk_aversion'] * port_risk
    
    # Initial weights
    initial_weights = np.ones(n) / n
    
    # Constraints
    bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n)]
    constraints_list = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # Sum = 1
    
    # Long-only constraint
    if constraints['long_only']:
        bounds = [(max(0, constraints['min_weight']), constraints['max_weight']) for _ in range(n)]
    
    # Sector constraints
    if constraints['sector_constraints'] and sector_data is not None:
        tickers = returns.index
        for sector in sector_data['sector'].unique():
            # Find tickers in this sector
            sector_indices = [i for i, ticker in enumerate(tickers) 
                             if ticker in sector_data.index and sector_data.loc[ticker, 'sector'] == sector]
            
            if sector_indices:
                # Add constraint: sector weight <= max_sector_weight
                def sector_constraint(w, indices=sector_indices, max_weight=constraints['max_sector_weight']):
                    return max_weight - sum(w[i] for i in indices)
                
                constraints_list.append({'type': 'ineq', 'fun': sector_constraint})
    
    # Solve optimization
    try:
        result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, 
                         constraints=constraints_list, options={'maxiter': 1000})
        
        if result.success:
            # Convert to dictionary and normalize
            weights = {ticker: max(0, weight) for ticker, weight in zip(returns.index, result.x)}
            total = sum(weights.values())
            return {ticker: w/total for ticker, w in weights.items()}
        else:
            # Fallback to equal weight
            return {ticker: 1/n for ticker in returns.index}
    except Exception as e:
        st.warning(f"Optimization error: {str(e)}. Using equal weights.")
        return {ticker: 1/n for ticker in returns.index}

def backtest_portfolio(prices, weights_history, rebalance_dates):
    """Run portfolio backtest"""
    daily_returns = prices['Adj Close'].pct_change().dropna()
    portfolio_returns = pd.Series(index=daily_returns.index, dtype=float)
    current_weights = None
    
    # Calculate returns for each day
    for date in portfolio_returns.index:
        # Check if rebalance day
        if date in rebalance_dates:
            current_weights = weights_history[date]
        
        if current_weights is None:
            continue
        
        # Calculate portfolio return
        if date in daily_returns.index:
            common_tickers = set(current_weights.keys()) & set(daily_returns.columns)
            day_return = 0
            weight_sum = sum(current_weights[ticker] for ticker in common_tickers)
            
            if weight_sum > 0:
                for ticker in common_tickers:
                    normalized_weight = current_weights[ticker] / weight_sum
                    if pd.notna(daily_returns.loc[date, ticker]):
                        day_return += normalized_weight * daily_returns.loc[date, ticker]
            
            portfolio_returns[date] = day_return
    
    # Calculate metrics
    cumulative = (1 + portfolio_returns).cumprod()
    
    # Performance metrics
    metrics = {
        'annualized_return': portfolio_returns.mean() * 252,
        'annualized_volatility': portfolio_returns.std() * np.sqrt(252),
        'max_drawdown': ((cumulative / cumulative.cummax()) - 1).min(),
        'win_rate': (portfolio_returns > 0).mean()
    }
    
    # Add Sharpe ratio
    metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['annualized_volatility'] if metrics['annualized_volatility'] > 0 else 0
    
    return portfolio_returns, cumulative, metrics

# ---------- Main Application ----------
# Create layout
tab1, tab2, tab3 = st.tabs(["Portfolio Performance", "Factor Analysis", "Holdings"])

# Sidebar for input parameters
with st.sidebar:
    st.header("Configuration")
    
    # Universe selection
    st.subheader("Investment Universe")
    universe = st.radio("Select universe", ["S&P 500", "Custom Tickers"])
    
    if universe == "Custom Tickers":
        custom_tickers = st.text_area("Enter tickers (comma-separated)",
                                     "AAPL, MSFT, AMZN, GOOGL, META, BRK-B, JNJ")
        tickers = [t.strip() for t in custom_tickers.split(',')]
    else:
        sp500 = load_sp500_tickers()
        sector_filter = st.multiselect("Filter by sector", sorted(sp500['sector'].unique()))
        
        if sector_filter:
            filtered_df = sp500[sp500['sector'].isin(sector_filter)]
            tickers = filtered_df['ticker'].tolist()
        else:
            tickers = sp500['ticker'].tolist()
            
        # Limit number of tickers
        max_tickers = st.slider("Maximum tickers", 10, 100, 30, 5)
        if len(tickers) > max_tickers:
            tickers = tickers[:max_tickers]
    
    # Backtest period
    st.subheader("Backtest Period")
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365*3)
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", start_date)
    with col2:
        end_date = st.date_input("End date", end_date)
    
    rebalance_freq = st.selectbox("Rebalancing frequency", 
                                ["Monthly", "Quarterly", "Semi-Annually", "Annually"], 1)
    
    # Factor weights
    st.subheader("Factor Weights")
    value_weight = st.slider("Value", 0.0, 1.0, 0.25, 0.05)
    momentum_weight = st.slider("Momentum", 0.0, 1.0, 0.25, 0.05)
    quality_weight = st.slider("Quality", 0.0, 1.0, 0.25, 0.05)
    size_weight = st.slider("Size", 0.0, 1.0, 0.25, 0.05)
    
    # Normalize weights
    total = value_weight + momentum_weight + quality_weight + size_weight
    if total > 0:
        factor_weights = {
            'value_factor': value_weight / total,
            'momentum_factor': momentum_weight / total,
            'quality_factor': quality_weight / total,
            'size_factor': size_weight / total
        }
    else:
        factor_weights = {'value_factor': 0.25, 'momentum_factor': 0.25, 
                         'quality_factor': 0.25, 'size_factor': 0.25}
    
    # Portfolio constraints
    st.subheader("Portfolio Constraints")
    long_only = st.checkbox("Long-only", True)
    min_weight = st.slider("Min weight (%)", 0.0, 5.0, 0.0, 0.5) / 100
    max_weight = st.slider("Max weight (%)", 1.0, 20.0, 5.0, 0.5) / 100
    
    sector_constraints = st.checkbox("Sector constraints", True)
    max_sector_weight = st.slider("Max sector weight (%)", 10.0, 50.0, 30.0, 5.0) / 100
    
    risk_aversion = st.slider("Risk aversion", 0.1, 10.0, 1.0, 0.1)
    risk_model = st.selectbox("Risk model", ["Ledoit-Wolf Shrinkage", "Sample Covariance"])
    risk_model_method = 'ledoit_wolf' if risk_model == "Ledoit-Wolf Shrinkage" else 'sample'
    
    # Run button
    run_backtest = st.button("Run Backtest", use_container_width=True)

# Run backtest if button pressed
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
                current_date += pd.DateOffset(months=1)
            elif rebalance_freq == "Quarterly":
                current_date += pd.DateOffset(months=3)
            elif rebalance_freq == "Semi-Annually":
                current_date += pd.DateOffset(months=6)
            else:  # Annually
                current_date += pd.DateOffset(years=1)
        
        # Align with trading days
        trading_dates = prices.index
        aligned_dates = []
        
        for date in rebalance_dates:
            if date in trading_dates:
                aligned_dates.append(date)
            else:
                future_dates = trading_dates[trading_dates >= date]
                if not future_dates.empty:
                    aligned_dates.append(future_dates[0])
        
        rebalance_dates = aligned_dates
        
        # Get sector data
        if universe == "S&P 500":
            sector_data = load_sp500_tickers().set_index('ticker')
        else:
            sector_data = None
        
        # Get fundamental data
        fundamental_data = get_fundamental_data(tickers)
        
        # Run backtest
        weights_history = {}
        
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
            
            # Optimize portfolio
            constraints = {
                'long_only': long_only,
                'min_weight': min_weight,
                'max_weight': max_weight,
                'sector_constraints': sector_constraints,
                'max_sector_weight': max_sector_weight,
                'risk_aversion': risk_aversion
            }
            
            weights = optimize_portfolio(
                combined_factors['combined_score'],
                cov_matrix,
                combined_factors['combined_score'],
                sector_data,
                constraints
            )
            
            # Store weights
            weights_history[rebalance_date] = weights
        
        # Backtest portfolio
        portfolio_returns, portfolio_cumulative, metrics = backtest_portfolio(
            prices, weights_history, rebalance_dates
        )
        
        # Get benchmark (S&P 500)
        spy_data = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
        benchmark_returns = spy_data['Adj Close'].pct_change().dropna()
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        
        # Display results in tabs
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
            
            # Using matplotlib instead of plotly
            fig, ax = plt.subplots(figsize=(10, 6))
            chart_data.plot(ax=ax)
            ax.set_title('Cumulative Returns')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            
            # Drawdown chart
            st.subheader("Portfolio Drawdown")
            drawdown = ((portfolio_cumulative / portfolio_cumulative.cummax()) - 1) * 100
            
            fig, ax = plt.subplots(figsize=(10, 6))
            drawdown.plot(ax=ax, color='red', alpha=0.7, linewidth=2)
            ax.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
            ax.set_title('Portfolio Drawdown')
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown (%)')
            ax.grid(True)
            ax.invert_yaxis()  # Invert y-axis to show drawdowns as negative
            st.pyplot(fig)
        
        with tab2:
            st.subheader("Factor Analysis")
            
            # Get factor exposures of final portfolio
            latest_date = max(weights_history.keys())
            latest_weights = weights_history[latest_date]
            
            # Calculate exposures
            factor_exposures = {}
            for factor_name, factor_scores in factor_data.items():
                common_tickers = set(latest_weights.keys()) & set(factor_scores.index)
                
                if common_tickers:
                    weighted_score = sum(
                        latest_weights[ticker] * factor_scores[ticker] 
                        for ticker in common_tickers
                        if pd.notna(factor_scores[ticker])
                    )
                    factor_exposures[factor_name] = weighted_score
            
            # Display factor exposures
            exposures_df = pd.DataFrame({
                'Factor': list(factor_exposures.keys()),
                'Exposure': list(factor_exposures.values())
            })
            
            # Using matplotlib for bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(exposures_df['Factor'], exposures_df['Exposure'], 
                  color=['royalblue' if x > 0 else 'tomato' for x in exposures_df['Exposure']])
            ax.set_title('Factor Exposures')
            ax.set_ylabel('Factor Exposure (Z-Score)')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            for i, v in enumerate(exposures_df['Exposure']):
                ax.text(i, v + 0.05 * np.sign(v), f"{v:.2f}", ha='center')
            st.pyplot(fig)
        
        with tab3:
            st.subheader("Portfolio Holdings")
            
            # Current holdings
            latest_date = max(weights_history.keys())
            latest_weights = weights_history[latest_date]
            
            weights_df = pd.DataFrame({
                'Ticker': list(latest_weights.keys()),
                'Weight': list(latest_weights.values())
            })
            weights_df['Weight (%)'] = weights_df['Weight'] * 100
            weights_df = weights_df.sort_values('Weight', ascending=False)
            
            # Display top 10 holdings
            st.subheader("Top 10 Holdings")
            
            top_10 = weights_df.head(10).copy()
            
            # Using matplotlib for bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(top_10['Ticker'], top_10['Weight (%)'], color='steelblue')
            ax.set_title('Top 10 Holdings')
            ax.set_ylabel('Weight (%)')
            ax.set_xlabel('Ticker')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            for i, v in enumerate(top_10['Weight (%)']):
                ax.text(i, v + 0.5, f"{v:.1f}%", ha='center')
            st.pyplot(fig)
            
            # Show all holdings
            if sector_data is not None:
                weights_with_sectors = weights_df.copy()
                weights_with_sectors['Sector'] = weights_with_sectors['Ticker'].map(
                    lambda t: sector_data.loc[t, 'sector'] if t in sector_data.index else 'Unknown'
                )
                
                # Sector allocation pie chart
                st.subheader("Sector Allocation")
                
                sector_weights = weights_with_sectors.groupby('Sector')['Weight'].sum()
                
                # Using matplotlib for pie chart
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.pie(sector_weights.values, labels=sector_weights.index, autopct='%1.1f%%', 
                      textprops={'fontsize': 12})
                ax.set_title('Sector Allocation')
                st.pyplot(fig)
                
                # Display all holdings with sectors
                st.dataframe(
                    weights_with_sectors[['Ticker', 'Sector', 'Weight (%)']].sort_values('Weight (%)', ascending=False),
                    use_container_width=True
                )
            else:
                st.dataframe(
                    weights_df[['Ticker', 'Weight (%)']].sort_values('Weight (%)', ascending=False),
                    use_container_width=True
                )
            
            # Download button
            csv = weights_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Portfolio CSV",
                data=csv,
                file_name="portfolio_weights.csv",
                mime="text/csv",
            )
else:
    # Show initial instructions when app first loads
    with tab1:
        st.info("Configure your portfolio parameters in the sidebar and click 'Run Backtest' to view results.")
    with tab2:
        st.info("Factor analysis will appear here after running the backtest.")
    with tab3:
        st.info("Portfolio holdings will appear here after running the backtest.")
