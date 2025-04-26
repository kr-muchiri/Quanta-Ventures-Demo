import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from datetime import datetime, timedelta

# ----- Streamlit Page Setup -----
st.set_page_config("Multi-Factor Portfolio Optimizer", layout="wide")
st.title("Multi-Factor Portfolio Optimizer")
st.markdown("""
<style>
.metric-card {background:#f9f9f9;border-left:5px solid #4CAF50;padding:15px;border-radius:5px}
.metric-card.negative {border-left-color:#f44336}
.metric-value {font-size:24px;font-weight:bold}
.metric-label {font-size:14px;color:#666}
</style>
""", unsafe_allow_html=True)

# ----- Utility Functions -----
@st.cache_data(ttl=86400)
def load_sp500():
    try:
        df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        return df.rename(columns={'Symbol': 'ticker', 'GICS Sector': 'sector'})[['ticker', 'sector']]
    except:
        return pd.DataFrame([('AAPL', 'Technology'), ('MSFT', 'Technology')], columns=['ticker', 'sector'])

def validate_tickers(tickers):
    valid = []
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period='5d')
            if not hist.empty:
                valid.append(t)
        except:
            continue
    return valid

@st.cache_data(ttl=86400)
def get_data(tickers, start, end):
    valid_tickers = validate_tickers(tickers)
    invalid_tickers = set(tickers) - set(valid_tickers)

    if not valid_tickers:
        st.error("None of the tickers returned data. Please check your list.")
        st.stop()

    data = yf.download(valid_tickers, start=start, end=end, progress=False, group_by="ticker")
    if data.empty:
        st.error("No data was returned. Please check the tickers or your internet connection.")
        st.stop()

    # Reconstruct 'Adj Close'
    try:
        if isinstance(data.columns, pd.MultiIndex):
            adj_close = data.xs("Adj Close", axis=1, level=1, drop_level=False)
            adj_close.columns = adj_close.columns.droplevel(1)  # Drop 'Adj Close' level
        else:
            adj_close = data
    except Exception as e:
        st.error(f"Error extracting 'Adj Close': {str(e)}")
        st.stop()

    if invalid_tickers:
        st.warning(f"The following tickers were skipped due to missing data: {', '.join(invalid_tickers)}")

    fundamentals = {t: yf.Ticker(t) for t in valid_tickers}
    return adj_close, fundamentals

# -- Z-score utility --
def zscore(series, inverse=False):
    mean, std = series.mean(), series.std()
    return ((-1 if inverse else 1) * (series - mean) / std) if std > 0 else pd.Series(0, index=series.index)

# -- Factor computation --
def factors(prices, funds, tickers):
    latest = prices.iloc[-1]
    mom, val, qual, size = {}, {}, {}, {}
    for t in tickers:
        tk = funds.get(t)
        if not tk or t not in prices.columns: continue
        info, bs, inc = tk.info, tk.balance_sheet, tk.income_stmt
        try: mom[t] = (latest[t] / prices[t].iloc[0]) - 1
        except: pass
        try: val[t] = latest[t] / (inc.loc['Net Income'][0] / info['sharesOutstanding'])
        except: pass
        try: qual[t] = inc.loc['Net Income'][0] / bs.loc['Stockholders Equity'][0]
        except: pass
        try: size[t] = latest[t] * info['sharesOutstanding']
        except: pass
    return pd.DataFrame({
        'momentum': zscore(pd.Series(mom)),
        'value': zscore(pd.Series(val), inverse=True),
        'quality': zscore(pd.Series(qual)),
        'size': zscore(pd.Series(size), inverse=True)
    }).dropna(how='all')

# -- Covariance matrix --
def cov_matrix(returns, method='ledoit'):
    return LedoitWolf().fit(returns).covariance_ if method == 'ledoit' else returns.cov().values

# -- Optimization --
def optimize(prices, cov, scores, weights, sectors=None, cons=None):
    n = len(scores)
    expected = scores.values
    def objective(w): return -np.dot(w, expected) + cons['risk_aversion'] * np.sqrt(w @ cov @ w)
    bounds = [(0 if cons['long'] else cons['min'], cons['max'])] * n
    constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
    if sectors is not None and cons['sector']:
        for s in sectors['sector'].unique():
            idx = [i for i, t in enumerate(scores.index) if sectors.loc[t, 'sector'] == s]
            constraints.append({'type': 'ineq', 'fun': lambda w, ix=idx: cons['max_sec'] - w[ix].sum()})
    result = minimize(objective, np.ones(n) / n, method='SLSQP', bounds=bounds, constraints=constraints)
    w = dict(zip(scores.index, np.clip(result.x, 0, None)))
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}

# -- Backtesting --
def backtest(prices, weights_hist, rebalance_dates):
    ret = prices.pct_change().dropna()
    port = pd.Series(0, index=ret.index)
    current = None
    for d in port.index:
        if d in weights_hist: current = weights_hist[d]
        if current:
            w_sum = sum(current.values())
            port[d] = sum((current[t] / w_sum) * ret.loc[d, t] for t in current if t in ret.columns)
    cum = (1 + port).cumprod()
    metrics = {
        'return': port.mean() * 252,
        'vol': port.std() * np.sqrt(252),
        'drawdown': (cum / cum.cummax() - 1).min(),
        'sharpe': (port.mean() * 252) / (port.std() * np.sqrt(252)) if port.std() > 0 else 0
    }
    return port, cum, metrics

# ----- Streamlit App Layout -----
tab1, tab2, tab3 = st.tabs(["Performance", "Factors", "Holdings"])
with st.sidebar:
    st.subheader("Configuration")
    uni = st.radio("Universe", ["S&P 500", "Custom"])
    if uni == "Custom":
        tickers = st.text_input("Tickers", "AAPL,MSFT").split(',')
    else:
        sp = load_sp500()
        max_n = st.slider("Max tickers", 10, 100, 30)
        tickers = sp['ticker'].tolist()[:max_n]
    sd, ed = st.date_input("Start", datetime.now() - timedelta(days=365)), st.date_input("End", datetime.now())
    freq = st.selectbox("Rebalancing (months)", [1, 3, 6, 12])
    wts = {k: st.slider(k.capitalize(), 0.0, 1.0, 0.25) for k in ['value', 'momentum', 'quality', 'size']}
    total = sum(wts.values())
    wts = {k: v / total for k, v in wts.items()} if total else {k: 0.25 for k in wts}
    cons = {
        'long': st.checkbox("Long-only", True),
        'min': 0,
        'max': st.slider("Max weight (%)", 1.0, 20.0, 5.0) / 100,
        'sector': st.checkbox("Sector Constraint", True),
        'max_sec': st.slider("Max sector weight (%)", 10.0, 50.0, 30.0) / 100,
        'risk_aversion': st.slider("Risk Aversion", 0.1, 10.0, 1.0)
    }
    run = st.button("Run Backtest")

if run:
    prices, funds = get_data(tickers, sd, ed)
    df = factors(prices, funds, prices.columns.tolist())
    rebal_dates = [prices.index[prices.index >= d][0] for d in pd.date_range(sd, ed, freq=pd.DateOffset(months=freq)) if any(prices.index >= d)]
    sector_data = load_sp500().set_index('ticker') if uni == "S&P 500" else None
    weights_hist = {d: optimize(prices.loc[:d], cov_matrix(prices.loc[:d].pct_change().dropna()), df.loc[df.index.intersection(tickers)], wts, sector_data, cons) for d in rebal_dates}
    port_ret, port_cum, metrics = backtest(prices, weights_hist, rebal_dates)

    with tab1:
        st.subheader("Performance")
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"<div class='metric-card'><div class='metric-value'>{metrics['return']*100:.2f}%</div><div class='metric-label'>Return</div></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-card'><div class='metric-value'>{metrics['sharpe']:.2f}</div><div class='metric-label'>Sharpe</div></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='metric-card negative'><div class='metric-value'>{metrics['drawdown']*100:.2f}%</div><div class='metric-label'>Max Drawdown</div></div>", unsafe_allow_html=True)
        col4.markdown(f"<div class='metric-card'><div class='metric-value'>{metrics['vol']*100:.2f}%</div><div class='metric-label'>Volatility</div></div>", unsafe_allow_html=True)
        st.line_chart(port_cum)

    with tab2:
        st.subheader("Final Factor Exposures")
        latest_weights = weights_hist[max(weights_hist.keys())]
        exposures = {k: sum(latest_weights.get(t, 0) * v for t, v in df[k].items() if t in latest_weights) for k in ['value', 'momentum', 'quality', 'size']}
        st.bar_chart(pd.Series(exposures))

    with tab3:
        st.subheader("Holdings")
        dfw = pd.DataFrame.from_dict(latest_weights, orient='index', columns=['Weight']).sort_values('Weight', ascending=False)
        dfw['Weight %'] = dfw['Weight'] * 100
        if sector_data is not None:
            dfw['Sector'] = dfw.index.map(lambda x: sector_data.loc[x, 'sector'] if x in sector_data.index else 'Unknown')
        st.dataframe(dfw)
