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

news_api_key = st.sidebar.text_input(
    "NewsAPI Key (Optional)",
    value=os.getenv("NEWSAPI_KEY", ""),
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
    
    if not api_key:
        st.warning("NewsAPI key not provided - skipping sentiment analysis")
        return scores
    
    nlp = load_sentiment_model()
    progress_bar = st.progress(0)
    
    for i, t in enumerate(tickers):
        headlines = fetch_headlines(t, api_key, days_back, limit)
        if not headlines:
            continue
            
        with st.expander(f"Headlines for {t} ({len(headlines)})"):
            st.write(headlines)
            
        outs = nlp(headlines)
        sentiment_values = [1 if o["label"] == "POSITIVE" else -1 if o["label"] == "NEGATIVE" else 0 for o in outs]
        mean = np.mean(sentiment_values) if sentiment_values else 0
        scores[t] = mean
        
        # Update progress
        progress_bar.progress((i + 1) / len(tickers))
        
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

# Run button
if st.button("Run Backtest", type="primary"):
    st.session_state.run_backtest = True
else:
    if 'run_backtest' not in st.session_state:
        st.session_state.run_backtest = False

# Run the backtest if button clicked
if st.session_state.run_backtest:
    try:
        # Create columns for results
        col1, col2 = st.columns([3, 2])
        
        with st.spinner("Fetching price data..."):
            prices = get_prices(tickers)
            
        with col1:
            st.subheader("Price Data")
            st.line_chart(prices)
        
        with st.spinner("Computing momentum signals..."):
            mom = momentum_signal(prices, lookback_days).loc[prices.index]
        
        with st.spinner("Fetching headlines & scoring sentiment..."):
            sent_scores = sentiment_score(tickers, news_api_key, news_lookback, sentence_limit)
            sent_sig = sentiment_signal(sent_scores)
        
        with col2:
            st.subheader("Sentiment Scores")
            sent_df = pd.DataFrame({
                'Ticker': list(sent_scores.keys()),
                'Sentiment Score': list(sent_scores.values())
            })
            sent_df['Signal'] = sent_df['Sentiment Score'].apply(lambda x: "Buy" if x > 0.05 else ("Sell" if x < -0.05 else "Neutral"))
            st.dataframe(sent_df, use_container_width=True)
        
        # Combine signals
        latest_mom = mom.iloc[-1:]
        signals = combine_signals(latest_mom, sent_sig)
        
        st.subheader("Combined Signal Snapshot (Latest Date)")
        signal_df = pd.DataFrame(signals.T)
        signal_df.columns = ['Signal']
        signal_df['Signal'] = signal_df['Signal'].map({1.0: "Buy", -1.0: "Sell", 0.0: "Neutral"})
        st.dataframe(signal_df, use_container_width=True)
        
        # Backtest
        st.subheader("Strategy Performance")
        sig_fwd = combine_signals(mom, sent_sig)
        ret = strategy_returns(prices, sig_fwd)
        sr = sharpe(ret)
        
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric("Sharpe Ratio", f"{sr:.2f}")
        with metrics_col2:
            annualized_return = ret.mean() * 252 * 100
            st.metric("Annualized Return", f"{annualized_return:.2f}%")
        with metrics_col3:
            annualized_vol = ret.std() * math.sqrt(252) * 100
            st.metric("Annualized Volatility", f"{annualized_vol:.2f}%")
        
        # Plot equity curve
        fig, ax = plt.subplots(figsize=(10, 6))
        cum_returns = (1 + ret).cumprod()
        cum_returns.plot(ax=ax)
        ax.set_title("Momentum + Sentiment Strategy Equity Curve")
        ax.set_ylabel("Cumulative Return")
        ax.grid(True)
        st.pyplot(fig)
        
        # Download buttons
        st.subheader("Download Results")
        strategy_returns_csv = ret.to_csv().encode('utf-8')
        st.download_button(
            label="Download Strategy Returns CSV",
            data=strategy_returns_csv,
            file_name="strategy_returns.csv",
            mime="text/csv"
        )
        
        signal_csv = signals.T.to_csv().encode('utf-8')
        st.download_button(
            label="Download Current Signals CSV",
            data=signal_csv,
            file_name="current_signals.csv",
            mime="text/csv"
        )
        
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
