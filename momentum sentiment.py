"""
Streamlit app for Momentum + Sentiment strategy backtester
This app allows users to backtest a trading strategy that combines:
1. Price momentum (3-month historical performance)
2. Sentiment analysis on recent news headlines using FinBERT
"""

import os
import sys
import datetime as dt
import math
import requests
import io
from typing import List, Dict
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from transformers import pipeline
import streamlit as st

# --------------------- Config ---------------------
DEFAULT_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA"]
# --------------------------------------------------

# Set page config
st.set_page_config(
    page_title="Momentum + Sentiment Backtester",
    page_icon="📈",
    layout="wide"
)

# App title and description
st.title("Momentum + Sentiment Strategy Backtester")
st.markdown("""
This app backtests a trading strategy that combines:
* **Price momentum** (lookback period of your choice)
* **News sentiment** analysis using FinBERT on recent headlines

Enter your tickers and parameters to run the backtest.
""")

# Sidebar for parameters
st.sidebar.header("Parameters")
lookback_days = st.sidebar.slider(
    "Momentum Lookback (Trading Days)",
    min_value=21,
    max_value=252,
    value=63,
    step=21,
    help="Number of trading days to look back for momentum calculation (63 ≈ 3 months)"
)

news_lookback = st.sidebar.slider(
    "News Lookback (Days)",
    min_value=1,
    max_value=30,
    value=7,
    help="Number of days to look back for news headlines"
)

# Try to get NewsAPI key from various sources, prioritizing secrets
news_api_key = None

# First try to get from streamlit secrets if deployed
try:
    news_api_key = st.secrets["NEWSAPI_KEY"]
    st.sidebar.success("✅ NewsAPI key found in secrets!")
except (KeyError, FileNotFoundError):
    # If not in secrets, try environment variable
    news_api_key = os.getenv("NEWSAPI_KEY")
    if news_api_key:
        st.sidebar.success("✅ NewsAPI key found in environment!")
    else:
        # If no key is found elsewhere, allow manual entry
        news_api_key = st.sidebar.text_input(
            "NewsAPI Key (Required)",
            value="",
            type="password",  # This hides the input
            help="Enter your NewsAPI key. Get one free at newsapi.org"
        )

sentence_limit = st.sidebar.slider(
    "Max Headlines per Ticker",
    min_value=10,
    max_value=100,
    value=50,
    help="Maximum number of headlines to analyze per ticker"
)

# Ticker input
ticker_input = st.text_area(
    "Enter Ticker Symbols (comma or space separated)",
    ", ".join(DEFAULT_TICKERS)
)

# Process tickers
tickers = [t.strip().upper() for t in ticker_input.replace(",", " ").split() if t.strip()]
if not tickers:
    tickers = DEFAULT_TICKERS

# Main functions
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_prices(tickers: List[str], period: str = "2y") -> pd.DataFrame:
    data = yf.download(tickers, period=period, interval="1d", auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    data = data.fillna(method="ffill")
    return data

def momentum_signal(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    returns = prices.pct_change(lookback)
    return returns.apply(np.sign)

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_headlines(ticker: str, api_key: str, days_back: int, limit: int) -> List[str]:
    if not api_key:
        return []
    url = ("https://newsapi.org/v2/everything?"
           f"q={ticker}&from={(dt.date.today()-dt.timedelta(days=days_back))}"
           f"&sortBy=publishedAt&language=en&pageSize={limit}&apiKey={api_key}")
    
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            st.warning(f"Error getting news for {ticker}: Status {r.status_code}")
            return []
        articles = r.json().get("articles", [])
        return [a["title"] for a in articles]
    except Exception as e:
        st.warning(f"Error fetching headlines for {ticker}: {str(e)}")
        return []

@st.cache_resource
def load_sentiment_model():
    with st.spinner("Loading FinBERT model - this may take a moment..."):
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def sentiment_score(tickers: List[str], api_key: str, days_back: int, limit: int) -> Dict[str, float]:
    scores = {t: 0.0 for t in tickers}
    sentiment_details = {t: {"positive": 0, "negative": 0, "neutral": 0, "headlines": []} for t in tickers}
    
    if not api_key:
        st.error("⚠️ No NewsAPI key found - sentiment analysis will not work")
        st.info("Please enter a NewsAPI key in the sidebar to enable sentiment analysis")
        return scores
    
    try:
        nlp = load_sentiment_model()
        progress_bar = st.progress(0)
        
        for i, t in enumerate(tickers):
            with st.status(f"Processing sentiment for {t}...", expanded=False) as status:
                headlines = fetch_headlines(t, api_key, days_back, limit)
                
                if not headlines:
                    status.update(label=f"No headlines found for {t}", state="error")
                    continue
                
                status.update(label=f"Analyzing {len(headlines)} headlines for {t}...")
                
                # Process headlines in smaller batches to avoid memory issues
                batch_size = 5
                all_results = []
                
                for j in range(0, len(headlines), batch_size):
                    batch = headlines[j:j+batch_size]
                    try:
                        batch_results = nlp(batch)
                        all_results.extend(batch_results)
                    except Exception as e:
                        st.warning(f"Error processing batch for {t}: {str(e)}")
                
                # Process sentiment results
                positive_count = 0
                negative_count = 0
                neutral_count = 0
                
                headline_sentiments = []
                
                for idx, (headline, result) in enumerate(zip(headlines[:len(all_results)], all_results)):
                    sentiment = result["label"]
                    confidence = result["score"]
                    
                    if sentiment == "POSITIVE":
                        positive_count += 1
                        sent_value = 1
                    elif sentiment == "NEGATIVE":
                        negative_count += 1
                        sent_value = -1
                    else:
                        neutral_count += 1
                        sent_value = 0
                    
                    headline_sentiments.append({
                        "headline": headline,
                        "sentiment": sentiment,
                        "confidence": confidence,
                        "value": sent_value
                    })
                
                # Calculate weighted sentiment
                if headline_sentiments:
                    # Use confidence scores as weights
                    weighted_sentiments = [(hs["value"] * hs["confidence"]) for hs in headline_sentiments]
                    weighted_mean = sum(weighted_sentiments) / len(headline_sentiments)
                    scores[t] = weighted_mean
                    
                    # Store detailed results
                    sentiment_details[t] = {
                        "positive": positive_count,
                        "negative": negative_count,
                        "neutral": neutral_count,
                        "headlines": headline_sentiments
                    }
                
                status.update(label=f"Completed sentiment analysis for {t}", state="complete")
            
            # Update progress
            progress_bar.progress((i + 1) / len(tickers))
        
        # Store sentiment details in session state for later use
        st.session_state.sentiment_details = sentiment_details
        
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")
        
    return scores

def sentiment_signal(scores: Dict[str, float]) -> pd.Series:
    return pd.Series({k: math.copysign(1, v) if abs(v) > 0.05 else 0 for k, v in scores.items()})

def combine_signals(momentum: pd.DataFrame, sentiment: pd.Series) -> pd.DataFrame:
    sentiment_df = pd.DataFrame([sentiment]).reindex(columns=momentum.columns).fillna(0)
    return momentum.mul(sentiment_df.iloc[0], axis=1)

def strategy_returns(prices: pd.DataFrame, signals: pd.DataFrame) -> pd.Series:
    daily_ret = prices.pct_change().shift(-1)  # next-day return
    strat = (signals * daily_ret).mean(axis=1)
    return strat.dropna()

def sharpe(series: pd.Series, rf: float = 0.0) -> float:
    ann_ret = series.mean() * 252
    ann_vol = series.std() * math.sqrt(252)
    return (ann_ret - rf) / ann_vol if ann_vol else np.nan

# Check if NewsAPI key is required but missing
missing_api_key = news_api_key == "" or news_api_key is None

# Run button
if st.button("Run Backtest", type="primary", disabled=missing_api_key):
    st.session_state.run_backtest = True
else:
    if 'run_backtest' not in st.session_state:
        st.session_state.run_backtest = False
        
# Show warning if API key is missing
if missing_api_key:
    st.warning("⚠️ Please enter a NewsAPI key in the sidebar to enable the backtest")
    st.info("Get a free NewsAPI key at [newsapi.org](https://newsapi.org/register)")
    st.markdown("---")

# Run the backtest if button clicked
if st.session_state.run_backtest:
    try:
        # Create a modern dashboard layout with tabs
        st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 18px;
            font-weight: 600;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            border-left: 5px solid #4CAF50;
        }
        .metric-card.negative {
            border-left: 5px solid #FF5252;
        }
        .metric-card.neutral {
            border-left: 5px solid #2196F3;
        }
        .metric-value {
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
        }
        .card-container {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        with st.spinner("Preparing dashboard..."):
            # Fetch all data first
            prices = get_prices(tickers)
            mom = momentum_signal(prices, lookback_days).loc[prices.index]
            sent_scores = sentiment_score(tickers, news_api_key, news_lookback, sentence_limit)
            sent_sig = sentiment_signal(sent_scores)
            
            # Combine signals
            latest_mom = mom.iloc[-1:]
            signals = combine_signals(latest_mom, sent_sig)
            
            # Backtest
            sig_fwd = combine_signals(mom, sent_sig)
            ret = strategy_returns(prices, sig_fwd)
            sr = sharpe(ret)
            annualized_return = ret.mean() * 252 * 100
            annualized_vol = ret.std() * math.sqrt(252) * 100
            cum_returns = (1 + ret).cumprod()
            
            # Create tabs for different sections
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Analysis", "🔍 Sentiment Details", "📑 Results"])
            
            with tab1:
                # Key metrics at the top
                st.markdown("<div class='card-container'>", unsafe_allow_html=True)
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                
                with metrics_col1:
                    metric_color = "positive" if sr > 1 else "negative" if sr < 0 else "neutral"
                    st.markdown(f"""
                    <div class='metric-card {metric_color}'>
                        <div class='metric-value'>{sr:.2f}</div>
                        <div class='metric-label'>Sharpe Ratio</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metrics_col2:
                    metric_color = "positive" if annualized_return > 0 else "negative"
                    st.markdown(f"""
                    <div class='metric-card {metric_color}'>
                        <div class='metric-value'>{annualized_return:.2f}%</div>
                        <div class='metric-label'>Annualized Return</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metrics_col3:
                    st.markdown(f"""
                    <div class='metric-card neutral'>
                        <div class='metric-value'>{annualized_vol:.2f}%</div>
                        <div class='metric-label'>Annualized Volatility</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Summary of signals
                st.markdown("<div class='card-container'>", unsafe_allow_html=True)
                st.subheader("Current Trading Signals")
                
                # Count signals by type
                signal_counts = signals.T['Signal'].map({1.0: "Buy", -1.0: "Sell", 0.0: "Neutral"}).value_counts().to_dict()
                buy_count = signal_counts.get("Buy", 0)
                sell_count = signal_counts.get("Sell", 0)
                neutral_count = signal_counts.get("Neutral", 0)
                
                # Display signal counts in a visually appealing way
                signal_cols = st.columns(3)
                with signal_cols[0]:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 10px; background-color: rgba(76, 175, 80, 0.1); border-radius: 5px;'>
                        <div style='font-size: 24px; font-weight: bold; color: #4CAF50;'>{buy_count}</div>
                        <div style='font-size: 16px;'>Buy Signals</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with signal_cols[1]:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 10px; background-color: rgba(255, 82, 82, 0.1); border-radius: 5px;'>
                        <div style='font-size: 24px; font-weight: bold; color: #FF5252;'>{sell_count}</div>
                        <div style='font-size: 16px;'>Sell Signals</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with signal_cols[2]:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 10px; background-color: rgba(33, 150, 243, 0.1); border-radius: 5px;'>
                        <div style='font-size: 24px; font-weight: bold; color: #2196F3;'>{neutral_count}</div>
                        <div style='font-size: 16px;'>Neutral Signals</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display detailed signals table
                signal_df = pd.DataFrame(signals.T)
                signal_df.columns = ['Signal']
                signal_df['Signal'] = signal_df['Signal'].map({1.0: "Buy", -1.0: "Sell", 0.0: "Neutral"})
                signal_df = signal_df.reset_index().rename(columns={'index': 'Ticker'})
                
                # Add color highlighting
                def highlight_signal(val):
                    if val == 'Buy':
                        return 'background-color: rgba(76, 175, 80, 0.2)'
                    elif val == 'Sell':
                        return 'background-color: rgba(255, 82, 82, 0.2)'
                    else:
                        return 'background-color: rgba(33, 150, 243, 0.1)'
                
                styled_df = signal_df.style.applymap(highlight_signal, subset=['Signal'])
                st.dataframe(styled_df, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Equity curve chart
                st.markdown("<div class='card-container'>", unsafe_allow_html=True)
                st.subheader("Strategy Performance")
                
                # Create a more visually appealing chart
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(cum_returns.index, cum_returns.values, linewidth=2, color='#4CAF50')
                ax.fill_between(cum_returns.index, 1, cum_returns.values, alpha=0.2, color='#4CAF50')
                ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
                ax.set_title("Momentum + Sentiment Strategy Equity Curve", fontsize=16)
                ax.set_ylabel("Cumulative Return", fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Format dates and y-axis
                from matplotlib.ticker import FuncFormatter
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.2f}x'.format(y)))
                
                st.pyplot(fig)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with tab2:
                # Analysis of the components
                st.markdown("<div class='card-container'>", unsafe_allow_html=True)
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.subheader("Price Data")
                    
                    # Create normalized price chart for comparison
                    norm_prices = prices / prices.iloc[0]
                    st.line_chart(norm_prices)
                    
                    # Momentum analysis
                    st.subheader("Momentum Signals")
                    # Get most recent momentum values
                    latest_mom_df = mom.iloc[-1:].T.reset_index()
                    latest_mom_df.columns = ['Ticker', 'Momentum']
                    latest_mom_df['Signal'] = latest_mom_df['Momentum'].apply(
                        lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Neutral"
                    )
                    
                    # Style the momentum dataframe
                    def highlight_momentum(val):
                        if val == 'Positive':
                            return 'background-color: rgba(76, 175, 80, 0.2)'
                        elif val == 'Negative':
                            return 'background-color: rgba(255, 82, 82, 0.2)'
                        else:
                            return 'background-color: rgba(33, 150, 243, 0.1)'
                    
                    styled_mom_df = latest_mom_df.style.applymap(highlight_momentum, subset=['Signal'])
                    st.dataframe(styled_mom_df, use_container_width=True)
                
                with col2:
                    st.subheader("Sentiment Analysis")
                    
                    # Create a more informative sentiment table
                    sent_df = pd.DataFrame({
                        'Ticker': list(sent_scores.keys()),
                        'Sentiment Score': list(sent_scores.values())
                    })
                    sent_df['Signal'] = sent_df['Sentiment Score'].apply(
                        lambda x: "Positive" if x > 0.05 else "Negative" if x < -0.05 else "Neutral"
                    )
                    
                    # Style the sentiment dataframe
                    def highlight_sentiment(val):
                        if val == 'Positive':
                            return 'background-color: rgba(76, 175, 80, 0.2)'
                        elif val == 'Negative':
                            return 'background-color: rgba(255, 82, 82, 0.2)'
                        else:
                            return 'background-color: rgba(33, 150, 243, 0.1)'
                    
                    styled_sent_df = sent_df.style.applymap(highlight_sentiment, subset=['Signal'])
                    st.dataframe(styled_sent_df, use_container_width=True)
                    
                    # Sentiment distribution visualization
                    st.subheader("Sentiment Distribution")
                    
                    sent_distribution = sent_df['Signal'].value_counts()
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    colors = ['#4CAF50', '#FF5252', '#2196F3']
                    wedges, texts, autotexts = ax.pie(
                        sent_distribution, 
                        labels=sent_distribution.index, 
                        autopct='%1.1f%%',
                        colors=['rgba(76, 175, 80, 0.7)', 'rgba(255, 82, 82, 0.7)', 'rgba(33, 150, 243, 0.7)'], 
                        startangle=90
                    )
                    ax.axis('equal')
                    
                    # Enhance the appearance
                    for text in texts:
                        text.set_fontsize(12)
                    for autotext in autotexts:
                        autotext.set_fontsize(12)
                        autotext.set_color('white')
                        
                    st.pyplot(fig)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with tab3:
                # Detailed sentiment analysis
                if hasattr(st.session_state, 'sentiment_details'):
                    sentiment_details = st.session_state.sentiment_details
                    
                    for ticker, details in sentiment_details.items():
                        if not details.get('headlines'):
                            continue
                            
                        with st.expander(f"Sentiment Details for {ticker}"):
                            # Sentiment distribution
                            pos = details['positive']
                            neg = details['negative']
                            neu = details['neutral']
                            
                            sentiment_cols = st.columns(3)
                            with sentiment_cols[0]:
                                st.markdown(f"""
                                <div style='text-align: center; padding: 10px; background-color: rgba(76, 175, 80, 0.1); border-radius: 5px;'>
                                    <div style='font-size: 24px; font-weight: bold; color: #4CAF50;'>{pos}</div>
                                    <div style='font-size: 16px;'>Positive Headlines</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with sentiment_cols[1]:
                                st.markdown(f"""
                                <div style='text-align: center; padding: 10px; background-color: rgba(255, 82, 82, 0.1); border-radius: 5px;'>
                                    <div style='font-size: 24px; font-weight: bold; color: #FF5252;'>{neg}</div>
                                    <div style='font-size: 16px;'>Negative Headlines</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with sentiment_cols[2]:
                                st.markdown(f"""
                                <div style='text-align: center; padding: 10px; background-color: rgba(33, 150, 243, 0.1); border-radius: 5px;'>
                                    <div style='font-size: 24px; font-weight: bold; color: #2196F3;'>{neu}</div>
                                    <div style='font-size: 16px;'>Neutral Headlines</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Headline details
                            st.subheader("Headlines and Sentiments")
                            
                            headlines_df = pd.DataFrame(details['headlines'])
                            
                            # Format the dataframe
                            if not headlines_df.empty:
                                headlines_df = headlines_df[['headline', 'sentiment', 'confidence']]
                                headlines_df.columns = ['Headline', 'Sentiment', 'Confidence']
                                
                                # Style function for sentiment
                                def color_sentiment(val):
                                    if val == 'POSITIVE':
                                        return 'background-color: rgba(76, 175, 80, 0.2)'
                                    elif val == 'NEGATIVE':
                                        return 'background-color: rgba(255, 82, 82, 0.2)'
                                    else:
                                        return 'background-color: rgba(33, 150, 243, 0.1)'
                                
                                # Format confidence as percentage
                                headlines_df['Confidence'] = headlines_df['Confidence'].apply(lambda x: f"{x*100:.1f}%")
                                
                                # Apply styling
                                styled_headlines = headlines_df.style.applymap(color_sentiment, subset=['Sentiment'])
                                st.dataframe(styled_headlines, use_container_width=True)
                else:
                    st.info("Run the analysis first to see detailed sentiment information.")
            
            with tab4:
                # Results and exports
                st.markdown("<div class='card-container'>", unsafe_allow_html=True)
                st.subheader("Performance Metrics")
                
                # Create a metrics dataframe
                metrics_df = pd.DataFrame({
                    'Metric': ['Sharpe Ratio', 'Annualized Return', 'Annualized Volatility', 'Final Cumulative Return'],
                    'Value': [
                        f"{sr:.2f}", 
                        f"{annualized_return:.2f}%", 
                        f"{annualized_vol:.2f}%",
                        f"{cum_returns.iloc[-1]:.2f}x"
                    ]
                })
                
                st.dataframe(metrics_df, use_container_width=True)
                
                # Monthly returns
                st.subheader("Monthly Returns")
                
                # Convert daily returns to monthly
                monthly_returns = ret.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
                
                # Format as a table with years as rows and months as columns
                monthly_returns_table = monthly_returns.to_frame('Return')
                monthly_returns_table['Year'] = monthly_returns_table.index.year
                monthly_returns_table['Month'] = monthly_returns_table.index.month
                pivot_table = monthly_returns_table.pivot_table(
                    index='Year', 
                    columns='Month', 
                    values='Return',
                    aggfunc='first'
                )
                
                # Rename month columns to names
                month_names = {
                    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
                }
                pivot_table = pivot_table.rename(columns=month_names)
                
                # Calculate YTD return
                pivot_table['YTD'] = pivot_table.sum(axis=1, skipna=True)
                
                # Format the values as percentages
                formatted_table = pivot_table.applymap(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
                
                # Display the formatted table
                st.dataframe(formatted_table, use_container_width=True)
                
                # Export options
                st.subheader("Export Data")
                
                download_cols = st.columns(3)
                with download_cols[0]:
                    strategy_returns_csv = ret.to_csv().encode('utf-8')
                    st.download_button(
                        label="Download Strategy Returns",
                        data=strategy_returns_csv,
                        file_name="strategy_returns.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with download_cols[1]:
                    signal_csv = signals.T.to_csv().encode('utf-8')
                    st.download_button(
                        label="Download Current Signals",
                        data=signal_csv,
                        file_name="current_signals.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with download_cols[2]:
                    # Create a comprehensive report
                    report_data = {
                        'Strategy': 'Momentum + Sentiment',
                        'Date': pd.Timestamp.now().strftime('%Y-%m-%d'),
                        'Tickers': ', '.join(tickers),
                        'Momentum Lookback Days': lookback_days,
                        'News Lookback Days': news_lookback,
                        'Sharpe Ratio': f"{sr:.2f}",
                        'Annualized Return': f"{annualized_return:.2f}%",
                        'Annualized Volatility': f"{annualized_vol:.2f}%",
                        'Final Cumulative Return': f"{cum_returns.iloc[-1]:.2f}x"
                    }
                    
                    report_df = pd.DataFrame([report_data])
                    report_csv = report_df.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        label="Download Summary Report",
                        data=report_csv,
                        file_name="strategy_summary.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                st.markdown("</div>", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
**Note**: This app requires:
* An internet connection to fetch stock data via yfinance
* A NewsAPI key for sentiment analysis (optional, but recommended)
* The first run may take longer as it downloads the FinBERT model
""")
