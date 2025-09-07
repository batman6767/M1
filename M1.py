# app.py
"""
SPIDER — Single-page iOS-style Dark Streamlit app (mobile responsive)
- Yahoo Finance data
- Indicators: MA20, MA50, EMA, RSI, MACD, Bollinger Bands
- Forecasts: Prophet, ARIMA, Random Forest, LSTM
- All models run in one click
- Single-file Streamlit app, optimized for mobile and desktop
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta

# --------------------------
# ML libraries (optional)
# --------------------------
HAS_PROPHET = False
HAS_ARIMA = False
HAS_SKLEARN = False
HAS_TF = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    Prophet = None

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except Exception:
    ARIMA = None

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import MinMaxScaler
    HAS_SKLEARN = True
except Exception:
    RandomForestRegressor = None
    MinMaxScaler = None

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except Exception:
    tf = None
    keras = None
    layers = None

# --------------------------
# Streamlit page config + CSS (responsive)
# --------------------------
st.set_page_config(page_title="M1 — SPIDER", layout="wide", initial_sidebar_state="auto")

# Custom CSS for dark iOS-like look + mobile responsiveness
st.markdown(
    """
    <style>
    /* Base dark theme */
    :root {
        --bg: #000000;
        --card: #0B0B0C;
        --muted: #8E8E93;
        --accent: #1C1C1E;
        --border: #222224;
        --text: #E6E6E6;
    }
    body, .stApp {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    /* central container + mobile max width */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1150px;
        margin-left: auto;
        margin-right: auto;
    }

    /* Inputs & buttons */
    .stButton>button, .stDownloadButton>button {
        background-color: var(--accent) !important;
        color: var(--text) !important;
        border-radius: 12px !important;
        border: 1px solid #2C2C2E !important;
        padding: 0.7rem 1rem !important;
        font-size: 1rem !important;
        min-height: 44px !important; /* touch friendly */
    }
    .stSelectbox>div, .stTextInput>div>div>input, .stSlider>div>div, .stNumberInput>div>div>input {
        background-color: #0F0F10 !important;
        color: var(--text) !important;
        border-radius: 10px !important;
        border: 1px solid var(--border) !important;
        padding: 0.4rem 0.6rem !important;
        min-height: 44px !important;
    }
    .stCheckbox>div, .stRadio>div, .stMarkdown {
        color: var(--text) !important;
    }
    h1, h2, h3, h4 { color: var(--text) !important; text-align: center; margin: 0.25rem 0 0.6rem 0; }

    .model-availability {
        background-color: var(--card);
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #202022;
    }

    /* Responsive layout tweaks */
    @media (max-width: 900px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; max-width: 820px; }
        .stApp .css-1vq4p4l { padding: 0 0 !important; } /* try to reduce extra paddings */
    }
    @media (max-width: 600px) {
        /* stack and enlarge touch targets */
        .block-container { padding-left: 0.6rem; padding-right: 0.6rem; max-width: 420px; }
        .stButton>button, .stDownloadButton>button { font-size: 1.05rem !important; padding: 0.85rem 1rem !important; border-radius: 14px !important; }
        .stSelectbox>div, .stTextInput>div>div>input, .stSlider>div>div { min-height: 52px !important; font-size: 1rem !important; }
        /* Make metrics wrap nicely */
        .stMetric > div {
            min-width: 0 !important;
        }
        /* Make Plotly charts take full width and be taller for touch */
        .element-container .stPlotlyChart > div { height: auto !important; }
    }

    /* A small helper to center the title area on mobile */
    .title-wrapper { display:flex; align-items:center; justify-content:center; flex-direction:column; gap:6px; }

    /* Ensure dataframes don't overflow horizontally on small screens */
    .stDataFrame div[role="table"] { width: 100% !important; overflow-x: auto; }

    /* Improve sidebar styles for dark theme */
    .css-1d391kg .css-1v0mbdj { background-color: var(--bg) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------
# DISCLAIMER GATE (session-safe)
# --------------------------
if "disclaimer_accepted" not in st.session_state:
    st.session_state.disclaimer_accepted = False

st.sidebar.markdown("## ⚠️ Disclaimer")
st.sidebar.markdown(
    """
    This application (“the App”) provides stock market data, analysis, and predictive tools **for educational and informational purposes only**.
    - The App does **not** provide financial, investment, trading, or legal advice.
    - All forecasts and analytics are **estimates only** and may be inaccurate or outdated.
    - Stock market investments are **risky and volatile**. Past performance is not indicative of future results.
    - Developers and contributors are **not liable** for any financial losses arising from use.
    By using the App you agree to use it at your own risk. For personalized financial guidance consult a licensed advisor.
    """
)

# Create an explicit accept button (more mobile-friendly than a tiny checkbox)
if not st.session_state.disclaimer_accepted:
    st.sidebar.write("Please accept to continue")
    if st.sidebar.button("I have read and accept the disclaimer"):
        st.session_state.disclaimer_accepted = True
    else:
        st.sidebar.warning("You must accept the disclaimer to use the app.")
        st.stop()

# --------------------------
# Utilities (cached)
# --------------------------
@st.cache_data(ttl=300)
def fetch_yahoo_data(ticker: str, period="6mo", interval="1d") -> pd.DataFrame:
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        # remove timezone if present
        try:
            df['Date'] = df['Date'].dt.tz_localize(None)
        except Exception:
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        for col in ['Open','High','Low','Close','Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['Date','Close']).reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['BB_MID'] = df['Close'].rolling(20).mean()
    df['BB_STD'] = df['Close'].rolling(20).std(ddof=0).fillna(0)
    df['BB_UP'] = df['BB_MID'] + 2*df['BB_STD']
    df['BB_LOW'] = df['BB_MID'] - 2*df['BB_STD']
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-8)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].clip(0,100).fillna(50)
    return df

def plot_advanced(df: pd.DataFrame, title: str, show_indicators: bool = True):
    # Use a responsive height: more height on narrow screens for touch
    layout_height = 620 if st.runtime.exists and st.experimental_get_query_params().get("mobile", [None])[0] else None
    # Build subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.62,0.18,0.2],
                        vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    if show_indicators:
        # show moving averages and bands if present
        if 'MA20' in df.columns:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], name='MA20', line=dict(width=1.4)), row=1, col=1)
        if 'MA50' in df.columns:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['MA50'], name='MA50', line=dict(width=1.4, dash='dash')), row=1, col=1)
        if 'BB_UP' in df.columns and 'BB_LOW' in df.columns:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_UP'], name='BB_UP', line=dict(width=1, dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_LOW'], name='BB_LOW', line=dict(width=1, dash='dot')), row=1, col=1)
    if 'Volume' in df.columns:
        fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color='gray', opacity=0.3), row=2, col=1)
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(width=1.2)), row=3, col=1)
        fig.add_hline(y=70, line_dash='dot', row=3, col=1)
        fig.add_hline(y=30, line_dash='dot', row=3, col=1)

    fig.update_layout(template='plotly_dark', title=title,
                      margin=dict(l=16,r=16,t=36,b=8),
                      paper_bgcolor='#000000', plot_bgcolor='#000000',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    # Render responsive
    st.plotly_chart(fig, use_container_width=True, theme=None)

# --------------------------
# Forecasts (Prophet/ARIMA/RF/LSTM)
# --------------------------
def forecast_all(df: pd.DataFrame, periods: int = 30):
    forecasts = {}
    if HAS_PROPHET:
        try:
            prophet_df = df[['Date','Close']].rename(columns={'Date':'ds','Close':'y'}).copy()
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds']).dt.tz_localize(None)
            m = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=periods, freq='D')
            future['ds'] = pd.to_datetime(future['ds']).dt.tz_localize(None)
            fc = m.predict(future)[['ds','yhat','yhat_lower','yhat_upper']].rename(columns={'ds':'Date'})
            forecasts['Prophet'] = fc
        except Exception as e:
            st.error(f"Prophet error: {e}")

    if HAS_ARIMA:
        try:
            series = df.set_index('Date')['Close'].sort_index()
            series.index = pd.to_datetime(series.index).tz_localize(None)
            daily_idx = pd.date_range(series.index.min(), series.index.max(), freq='D')
            series = series.reindex(daily_idx).ffill()
            model = ARIMA(series, order=(5,1,0)).fit()
            fc = model.forecast(steps=periods)
            dates = pd.date_range(start=series.index[-1]+timedelta(days=1), periods=periods)
            forecasts['ARIMA'] = pd.DataFrame({'Date': dates, 'yhat': fc.values})
        except Exception as e:
            st.error(f"ARIMA error: {e}")

    if HAS_SKLEARN:
        try:
            data = df[['Close']].copy()
            n_lags = 5
            for lag in range(1,n_lags+1):
                data[f'lag_{lag}'] = data['Close'].shift(lag)
            data = data.dropna()
            X = data[[f'lag_{i}' for i in range(1,n_lags+1)]].values
            y = data['Close'].values
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X, y)
            last_window = X[-1].tolist()
            preds = []
            for _ in range(periods):
                p = float(model.predict([last_window]))
                preds.append(p)
                last_window = [p]+last_window[:-1]
            dates = pd.date_range(start=df['Date'].iloc[-1]+timedelta(days=1), periods=periods)
            forecasts['RandomForest'] = pd.DataFrame({'Date': dates, 'yhat': preds})
        except Exception as e:
            st.error(f"RandomForest error: {e}")

    if HAS_TF and HAS_SKLEARN:
        try:
            values = df['Close'].values.astype('float32')
            n_lags = 20
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(values.reshape(-1,1)).flatten()
            X, y = [], []
            for i in range(n_lags, len(scaled)):
                X.append(scaled[i-n_lags:i])
                y.append(scaled[i])
            X = np.array(X).reshape(-1, n_lags, 1)
            y = np.array(y)
            tf.keras.backend.clear_session()
            model = keras.Sequential([
                layers.Input(shape=(n_lags,1)),
                layers.LSTM(64, return_sequences=False),
                layers.Dense(32, activation='relu'),
                layers.Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, epochs=10, batch_size=16, verbose=0)
            last_window = list(scaled[-n_lags:])
            preds_scaled = []
            for _ in range(periods):
                x = np.array(last_window).reshape(1, n_lags, 1)
                p = float(model.predict(x, verbose=0)[0,0])
                preds_scaled.append(p)
                last_window = last_window[1:]+[p]
            preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten().tolist()
            dates = pd.date_range(start=df['Date'].iloc[-1]+timedelta(days=1), periods=periods)
            forecasts['LSTM'] = pd.DataFrame({'Date': dates, 'yhat': preds})
        except Exception as e:
            st.error(f"LSTM error: {e}")

    return forecasts

# --------------------------
# MAIN UI (controls in sidebar for mobile friendliness)
# --------------------------
st.markdown('<div class="title-wrapper"><h1>M1 — SPIDER</h1></div>', unsafe_allow_html=True)

tickers_list = [
    "AAPL","MSFT","GOOG","AMZN","TSLA","META","NVDA","JPM","V","JNJ","WMT",
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS",
    "BTC-USD","ETH-USD"
]

# Controls live in sidebar (mobile collapses into a hamburger automatically)
st.sidebar.markdown("### Controls")
ticker = st.sidebar.selectbox("Ticker", tickers_list, index=0)
period = st.sidebar.selectbox("History period", ["1mo","3mo","6mo","1y","2y","5y","10y","max"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d","1wk","1mo"], index=0)
show_indicators = st.sidebar.checkbox("Show Indicators", value=True)
run = st.sidebar.button("Run All")

st.markdown("---")

if run:
    with st.spinner("Fetching data and computing indicators..."):
        df = fetch_yahoo_data(ticker, period, interval)
    if df.empty:
        st.error("No data retrieved. Check ticker or network.")
        st.stop()

    # compute indicators
    df = compute_indicators(df)

    # Top metrics (responsive)
    last = df.iloc[-1]
    # Use columns but allow natural wrapping on mobile: set max column count
    metric_cols = st.columns(5)
    metric_cols[0].metric("Date", str(pd.to_datetime(last['Date']).date()))
    metric_cols[1].metric("Open", f"{last['Open']:.2f}")
    metric_cols[2].metric("High", f"{last['High']:.2f}")
    metric_cols[3].metric("Low", f"{last['Low']:.2f}")
    metric_cols[4].metric("Close", f"{last['Close']:.2f}")

    # Main chart
    plot_advanced(df, f"{ticker} — Price & Indicators", show_indicators)

    # Downloads + raw data
    with st.expander("Raw data & downloads", expanded=False):
        st.download_button("Download raw data (CSV)", df.to_csv(index=False), file_name=f"{ticker}_raw.csv", mime="text/csv")
        st.dataframe(df.tail(200), use_container_width=True)

    # Forecasts
    st.subheader("FORECASTS")
    forecasts = forecast_all(df, periods=30)
    if not forecasts:
        st.info("No forecasting models available in this environment (Prophet/ARIMA/scikit-learn/TensorFlow missing).")
    else:
        for model_name, fc in forecasts.items():
            st.markdown(f"### {model_name}")
            # If Prophet with bounds, show bounds
            st.dataframe(fc, use_container_width=True)
            st.download_button(f"Download {model_name} forecast CSV", fc.to_csv(index=False), file_name=f"{ticker}_{model_name}.csv", mime="text/csv")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Historical'))
            fig.add_trace(go.Scatter(x=fc['Date'], y=fc['yhat'], name=f'{model_name} Forecast'))
            if model_name == "Prophet" and 'yhat_lower' in fc.columns and 'yhat_upper' in fc.columns:
                fig.add_trace(go.Scatter(x=fc['Date'], y=fc['yhat_lower'], name='Lower Bound', line=dict(dash='dot')))
                fig.add_trace(go.Scatter(x=fc['Date'], y=fc['yhat_upper'], name='Upper Bound', line=dict(dash='dot')))
            st.plotly_chart(fig, use_container_width=True, theme=None)

    st.markdown("---")
    st.subheader("Recent data (tail)")
    st.dataframe(df.tail(50), use_container_width=True)

else:
    st.info("Select a ticker and hit 'Run All' in the sidebar to fetch data and forecasts.")

# Footer small credits
st.markdown(
    """
    <div style="text-align:center; margin-top:14px; color:#8E8E93; font-size:12px">
      SPIDER • Mobile-responsive Streamlit app — data from Yahoo Finance
    </div>
    """,
    unsafe_allow_html=True,
)
