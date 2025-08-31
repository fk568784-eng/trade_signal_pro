# Trade Signal Pro – Streamlit app (BEST + Binance Live Wiring)
# -------------------------------------------------------------
# This file is the enhanced "BEST" edition with **Binance live-order wiring** integrated.
#
# SECURITY FIRST:
# • Never hardcode API keys. Prefer environment variables or paste them into the UI for a single session.
# • The app defaults to paper trading. Live trading requires explicitly enabling "Enable Live" and checking the risk acknowledgement box.
# • Use IP restrictions and limited permissions on your Binance API key.

import os
import math
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pytz
import streamlit as st
import yfinance as yf

# Optional: python-binance (install: pip install python-binance)
try:
    from binance.client import Client as BinanceClient
    BINANCE_AVAILABLE = True
except Exception:
    BinanceClient = None
    BINANCE_AVAILABLE = False

# ---------------------- Indicator helpers (same as BEST edition) ----------------------
# (omitted here for brevity in the developer summary — keep full indicator code in real file)

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=period - 1, adjust=False).mean()
    ma_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    ma = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = ma + std_mult * std
    lower = ma - std_mult * std
    return upper, ma, lower


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1/period, adjust=False).mean()

# SuperTrend, ADX, VWAP implementations (same as before) — keep them in real code

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close']
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    tr = true_range(df)
    atr_val = tr.rolling(window=period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=period).sum() / (atr_val+1e-9))
    minus_di = 100 * (minus_dm.rolling(window=period, min_periods=period).sum() / (atr_val+1e-9))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx_val = dx.rolling(window=period, min_periods=period).mean()
    return adx_val


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    hl2 = (df['High'] + df['Low']) / 2
    atrv = atr(df, period)
    upperband = hl2 + (multiplier * atrv)
    lowerband = hl2 - (multiplier * atrv)
    final_upper = upperband.copy()
    final_lower = lowerband.copy()
    supertrend = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i == 0:
            final_upper.iat[0] = upperband.iat[0]
            final_lower.iat[0] = lowerband.iat[0]
            supertrend.iat[0] = 1
            continue
        if (upperband.iat[i] < final_upper.iat[i-1]) or (df['Close'].iat[i-1] > final_upper.iat[i-1]):
            final_upper.iat[i] = upperband.iat[i]
        else:
            final_upper.iat[i] = final_upper.iat[i-1]
        if (lowerband.iat[i] > final_lower.iat[i-1]) or (df['Close'].iat[i-1] < final_lower.iat[i-1]):
            final_lower.iat[i] = lowerband.iat[i]
        else:
            final_lower.iat[i] = final_lower.iat[i-1]
        if supertrend.iat[i-1] == 1:
            supertrend.iat[i] = 1 if df['Close'].iat[i] <= final_upper.iat[i] else -1
        else:
            supertrend.iat[i] = -1 if df['Close'].iat[i] >= final_lower.iat[i] else 1
    return supertrend


def vwap(df: pd.DataFrame) -> pd.Series:
    if 'Volume' not in df.columns:
        return pd.Series(np.nan, index=df.index)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    cum_tp_vol = (tp * df['Volume']).groupby(df.index.date).cumsum()
    cum_vol = df['Volume'].groupby(df.index.date).cumsum()
    return cum_tp_vol / (cum_vol + 1e-9)

# ---------------------- Binance Integrator ----------------------
class BinanceIntegrator:
    """Simple Binance helper. Requires python-binance package.
    Designed for spot trading examples. Works in paper mode by default.
    """
    def __init__(self, api_key: str = None, api_secret: str = None, paper_trade: bool = True):
        self.paper_trade = paper_trade
        self.api_key = api_key or os.environ.get('BINANCE_API_KEY')
        self.api_secret = api_secret or os.environ.get('BINANCE_API_SECRET')
        self.client = None
        if BINANCE_AVAILABLE and self.api_key and self.api_secret:
            try:
                self.client = BinanceClient(self.api_key, self.api_secret)
            except Exception as e:
                self.client = None
                print("Binance client init failed:", e)

    def ping(self) -> bool:
        if self.paper_trade:
            return True
        if not self.client:
            return False
        try:
            self.client.ping()
            return True
        except Exception:
            return False

    def get_balance(self, asset: str = 'USDT') -> float:
        if self.paper_trade:
            return 10000.0
        if not self.client:
            return 0.0
        try:
            bal = self.client.get_asset_balance(asset=asset)
            return float(bal['free']) if bal and 'free' in bal else 0.0
        except Exception:
            return 0.0

    def get_price(self, symbol: str) -> float:
        # symbol example: 'BTCUSDT'
        if self.paper_trade:
            # not real-time; return NaN
            return float('nan')
        if not self.client:
            return float('nan')
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception:
            return float('nan')

    def create_order(self, symbol: str, side: str, type: str='MARKET', quantity: float = None, price: float = None) -> Dict[str, Any]:
        """Create order. In paper mode this returns a simulated fill.
        side: 'BUY' or 'SELL'
        type: 'MARKET' or 'LIMIT'
        """
        if self.paper_trade:
            # Simulate immediate fill at 'price' if given, else NaN
            fill_price = price if price is not None else float('nan')
            return {'status': 'FILLED', 'symbol': symbol, 'side': side, 'type': type, 'filledQty': quantity or 0, 'price': fill_price}
        if not self.client:
            raise RuntimeError('Binance client not initialized')
        try:
            if type == 'MARKET':
                order = self.client.create_order(symbol=symbol, side=side, type=type, quantity=quantity)
            else:
                order = self.client.create_order(symbol=symbol, side=side, type=type, timeInForce='GTC', quantity=quantity, price=str(price))
            return order
        except Exception as e:
            raise

# ---------------------- Streamlit UI & Binance wiring ----------------------
st.set_page_config(page_title="Trade Signal Pro — Binance", layout="wide")
st.title("Trade Signal Pro — Binance Live Wiring")
st.caption("Defaults to PAPER mode. Enable Live to execute real orders on Binance. Use carefully.")

with st.sidebar:
    st.header("Connection: Binance")
    st.markdown("**Note:** Install `python-binance` to enable live connectivity: `pip install python-binance`.")
    # Allow user to paste keys for session use only
    st.markdown("Paste API keys (session-only). Do not store in code.")
    api_key = st.text_input("Binance API Key", type='password')
    api_secret = st.text_input("Binance API Secret", type='password')
    use_env = st.checkbox("Or use environment variables (BINANCE_API_KEY/BINANCE_API_SECRET)", value=True)
    paper_trade_default = True
    live_enable = st.checkbox("Enable Live Trading (dangerous)", value=False)
    accept_risk = st.checkbox("I understand the risks and accept responsibility for live trades", value=False)
    test_conn = st.button("Test Binance Connection")

    st.markdown("---")
    st.header("Trading Settings")
    trade_symbol = st.text_input("Trading Symbol (e.g., BTCUSDT)", value="BTCUSDT")
    trade_size = st.number_input("Order size (units) — for market orders", value=0.001, format="%.8f")
    trade_side = st.selectbox("Side", ['BUY', 'SELL'])
    trade_type = st.selectbox("Order Type", ['MARKET','LIMIT'])
    limit_price = st.number_input("Limit Price (only for LIMIT)", value=0.0)
    place_order_btn = st.button("Place Order")

# Build integrator
paper_mode = not (live_enable and accept_risk and BINANCE_AVAILABLE)
if use_env and BINANCE_AVAILABLE and (not api_key) and (not api_secret):
    # Favor environment variables if user selected
    api_key = os.environ.get('BINANCE_API_KEY', '')
    api_secret = os.environ.get('BINANCE_API_SECRET', '')

binance = BinanceIntegrator(api_key=api_key or None, api_secret=api_secret or None, paper_trade=paper_mode)

if test_conn:
    ok = binance.ping()
    if ok:
        st.success("Binance connection OK (paper mode: %s)" % str(paper_mode))
    else:
        st.error("Binance connection failed. Check your API keys and network.")

# Simple price fetch for UI
if st.button("Fetch current price"):
    price = binance.get_price(trade_symbol)
    if math.isnan(price):
        st.info("Price not available (paper mode or client not initialized).")
    else:
        st.write(f"Price for {trade_symbol}: {price}")

if place_order_btn:
    if paper_mode:
        st.warning("Placing simulated (paper) order — no real funds used.")
    else:
        if not BINANCE_AVAILABLE:
            st.error("python-binance not installed on server. Install 'python-binance' to enable live orders.")
        if not accept_risk:
            st.error("You must accept the live-trade risk acknowledgement to place live orders.")
    try:
        if trade_type == 'MARKET':
            resp = binance.create_order(symbol=trade_symbol, side=trade_side, type='MARKET', quantity=float(trade_size))
        else:
            resp = binance.create_order(symbol=trade_symbol, side=trade_side, type='LIMIT', quantity=float(trade_size), price=float(limit_price))
        st.write("Order response:")
        st.json(resp)
        if not paper_mode:
            st.success("Live order placed — check Binance account and logs." )
    except Exception as e:
        st.error(f"Order failed: {e}")

st.markdown("---")

# The rest of the app (indicators, signals, backtesting, UI) should be included below — for brevity the file focuses on Binance wiring.
st.info("This file demonstrates secure Binance live wiring. Integrate the main app logic (signals, backtests) above or below as needed. The canvas contains the full 'BEST' app; this file shows the connection & order flow additions.")

# SECURITY REMINDERS
st.markdown("""
**Security checklist before live trading:**
- Use API keys with only required permissions (Spot: enable trading, disable withdrawal).  
- Add IP restrictions if available.  
- Test in paper mode extensively.  
- Monitor logs and handle exceptions — network failures and rejections must be retried safely.

**If you want, I can now:**
- Add order logging and reconcile fills to the paper-trade ledger.  
- Add positions view and P/L by symbol.  
- Add Binance Futures example (if you trade futures).  
- Integrate a secure key injection UI (vault-like) rather than pasting keys.
""")

# End of file
