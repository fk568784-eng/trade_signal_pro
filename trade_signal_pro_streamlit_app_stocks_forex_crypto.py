import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objs as go

st.set_page_config(page_title="Trade Signal Pro", layout="wide")

# Sidebar - Market Selection
market_type = st.sidebar.selectbox("Choose Market", ["Crypto (Binance)", "Forex", "Stocks"])
ticker = st.sidebar.text_input("Enter Symbol (e.g., BTC-USD, EURUSD=X, AAPL)", "BTC-USD")
interval = st.sidebar.selectbox("Select Timeframe", ["1d", "1h", "15m"])

# Load Data
df = yf.download(ticker, period="1mo", interval=interval)

# Check if the data is empty
if df.empty:
    st.error(f"No data found for {ticker} with interval {interval}. Try another symbol or timeframe.")
    st.stop()  # stop app to prevent crash

# Check if there are enough rows for indicators (RSI needs at least 14)
if len(df) < 15:
    st.error(f"Not enough data to calculate indicators for {ticker} with interval {interval}.")
    st.stop()


# Check if data is empty
if df.empty:
    st.error(f"No data found for {ticker} with interval {interval}. Try another symbol or timeframe.")
    st.stop()  # Stop the app to prevent errors

if df.empty:
    st.error("No data found. Try another symbol.")
    st.stop()

# Add Indicators
df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
macd = ta.trend.MACD(df["Close"])
df["MACD"] = macd.macd()
df["Signal_Line"] = macd.macd_signal()
bb = ta.volatility.BollingerBands(df["Close"])
df["BB_High"] = bb.bollinger_hband()
df["BB_Low"] = bb.bollinger_lband()

# Generate Trading Signal
latest = df.iloc[-1]
signal = "Hold"
if latest["RSI"] < 30 and latest["MACD"] > latest["Signal_Line"]:
    signal = "Strong Buy"
elif latest["RSI"] > 70 and latest["MACD"] < latest["Signal_Line"]:
    signal = "Strong Sell"

# Show Alerts
if signal == "Strong Buy":
    st.success(f"🚀 {signal} for {ticker}")
elif signal == "Strong Sell":
    st.error(f"⚠️ {signal} for {ticker}")
else:
    st.info(f"ℹ️ {signal} for {ticker}")

# Plot Chart
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'], high=df['High'],
    low=df['Low'], close=df['Close'],
    name="Candlesticks"
))
fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], line=dict(color='blue', dash='dot'), name='BB High'))
fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], line=dict(color='blue', dash='dot'), name='BB Low'))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Indicators")
st.line_chart(df[["RSI"]])
st.line_chart(df[["MACD", "Signal_Line"]])
