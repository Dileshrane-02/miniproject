"""
Configuration and constants for Indian Stock Market AI Prediction System.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Indian Stocks (NSE Tickers in Yahoo Finance format)
SUPPORTED_STOCKS = {
    "RELIANCE": {"name": "Reliance Industries Ltd.", "ticker": "RELIANCE.NS", "sector": "Energy / Conglomerate"},
    "TCS": {"name": "Tata Consultancy Services Ltd.", "ticker": "TCS.NS", "sector": "Information Technology"},
    "INFY": {"name": "Infosys Ltd.", "ticker": "INFY.NS", "sector": "Information Technology"},
    "HDFCBANK": {"name": "HDFC Bank Ltd.", "ticker": "HDFCBANK.NS", "sector": "Banking & Finance"},
    "ICICIBANK": {"name": "ICICI Bank Ltd.", "ticker": "ICICIBANK.NS", "sector": "Banking & Finance"},
    "SBIN": {"name": "State Bank of India", "ticker": "SBIN.NS", "sector": "Public Sector Banking"},
    "ITC": {"name": "ITC Ltd.", "ticker": "ITC.NS", "sector": "FMCG / Conglomerate"},
    "LT": {"name": "Larsen & Toubro Ltd.", "ticker": "LT.NS", "sector": "Infrastructure & Engineering"},
}

BENCHMARK_INDEX = "^NSEI"  # NIFTY 50 Index

# Date Configuration
DEFAULT_START_DATE = "2019-01-01"
DEFAULT_END_DATE = None  # None indicates up to today

# Classification Thresholds (Next-day return threshold for BUY/HOLD/SELL)
# If Return > +THRESHOLD: BUY (1)
# If Return < -THRESHOLD: SELL (-1 or 2)
# Else: HOLD (0)
# A 0.5% (0.005) threshold accounts for typical daily Indian market volatility and transaction costs
DEFAULT_RETURN_THRESHOLD = 0.005  

# Target Labels Mapping
LABEL_MAP = {
    0: "SELL",
    1: "HOLD",
    2: "BUY"
}

REVERSE_LABEL_MAP = {
    "SELL": 0,
    "HOLD": 1,
    "BUY": 2
}

# Technical Indicator Parameters
INDICATOR_PARAMS = {
    "sma_fast": 10,
    "sma_slow": 50,
    "ema_fast": 12,
    "ema_slow": 26,
    "rsi_window": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_window": 20,
    "bb_std": 2.0,
    "atr_window": 14,
    "roc_window": 10,
    "volatility_window": 20,
    "volume_sma_window": 20
}

# Market Regime Detection Parameters
REGIME_CLUSTERS = 3
REGIME_NAMES = {
    0: "Sideways / Choppy",
    1: "Bullish (Low Volatility)",
    2: "Bearish (High Volatility)"
}

# Train/Val/Test Split Ratios (Chronological)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Transaction cost assumption for backtesting (0.1% per trade = brokerage + STT + slippage)
TRANSACTION_COST_PCT = 0.001
INITIAL_CAPITAL = 100000.0  # INR 1,00,000
