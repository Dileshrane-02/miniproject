"""
Technical Indicator Engineering Module.
Calculates trend, momentum, volatility, and volume indicators from OHLCV data
using pure vectorized pandas/numpy without lookahead bias.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Calculates Simple Moving Average."""
    return series.rolling(window=window).mean()


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculates Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculates Relative Strength Index (RSI) using Wilder's exponential smoothing method.
    RSI oscillates between 0 and 100 (>70: Overbought, <30: Oversold).
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Wilder's Smoothing
    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Dict[str, pd.Series]:
    """
    Calculates Moving Average Convergence Divergence (MACD).
    Returns MACD line, Signal line, and MACD Histogram.
    """
    fast_ema = compute_ema(close, fast_period)
    slow_ema = compute_ema(close, slow_period)
    macd_line = fast_ema - slow_ema
    macd_signal = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    
    return {
        "MACD_Line": macd_line,
        "MACD_Signal": macd_signal,
        "MACD_Hist": macd_hist
    }


def compute_bollinger_bands(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> Dict[str, pd.Series]:
    """
    Calculates Bollinger Bands: Upper, Middle, Lower bands, Bandwidth, and %B.
    """
    mid_band = compute_sma(close, window)
    std_dev = close.rolling(window=window).std()
    upper_band = mid_band + (std_dev * num_std)
    lower_band = mid_band - (std_dev * num_std)
    
    bandwidth = (upper_band - lower_band) / (mid_band + 1e-9)
    pct_b = (close - lower_band) / ((upper_band - lower_band) + 1e-9)
    
    return {
        "BB_Upper": upper_band,
        "BB_Mid": mid_band,
        "BB_Lower": lower_band,
        "BB_Bandwidth": bandwidth,
        "BB_PctB": pct_b
    }


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14
) -> Dict[str, pd.Series]:
    """
    Calculates Average True Range (ATR) measuring market volatility.
    Normalized ATR (NATR) provides scale-invariant percentage volatility.
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    natr = (atr / (close + 1e-9)) * 100.0  # Percentage ATR
    
    return {
        "ATR": atr,
        "NATR": natr
    }


def compute_roc(close: pd.Series, window: int = 10) -> pd.Series:
    """Calculates Rate of Change (ROC) momentum indicator (percentage change over N days)."""
    return ((close - close.shift(window)) / (close.shift(window) + 1e-9)) * 100.0


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculates On-Balance Volume (OBV) cumulative volume flow."""
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    return obv


def add_all_technical_indicators(df_input: pd.DataFrame, params: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Appends full suite of technical indicators to the input OHLCV DataFrame.
    """
    df = df_input.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    
    # 1. Moving Averages & Price Distance
    df["SMA_10"] = compute_sma(close, 10)
    df["SMA_50"] = compute_sma(close, 50)
    df["EMA_12"] = compute_ema(close, 12)
    df["EMA_26"] = compute_ema(close, 26)
    df["Dist_SMA_10"] = (close - df["SMA_10"]) / (df["SMA_10"] + 1e-9)
    df["Dist_SMA_50"] = (close - df["SMA_50"]) / (df["SMA_50"] + 1e-9)
    df["SMA_Cross"] = (df["SMA_10"] > df["SMA_50"]).astype(int)
    
    # 2. Momentum & Oscillators
    df["RSI_14"] = compute_rsi(close, 14)
    macd_dict = compute_macd(close, 12, 26, 9)
    for k, v in macd_dict.items():
        df[k] = v
        
    df["ROC_10"] = compute_roc(close, 10)
    df["ROC_5"] = compute_roc(close, 5)
    
    # 3. Volatility Indicators
    bb_dict = compute_bollinger_bands(close, 20, 2.0)
    for k, v in bb_dict.items():
        df[k] = v
        
    atr_dict = compute_atr(high, low, close, 14)
    for k, v in atr_dict.items():
        df[k] = v
        
    df["Daily_Return"] = close.pct_change()
    df["Return_5d"] = close.pct_change(periods=5)
    df["Historical_Vol_20"] = df["Daily_Return"].rolling(window=20).std() * np.sqrt(252)
    
    # 4. Volume Indicators
    df["Volume_SMA_20"] = compute_sma(volume, 20)
    df["Volume_Ratio"] = volume / (df["Volume_SMA_20"] + 1e-9)
    df["OBV"] = compute_obv(close, volume)
    df["OBV_EMA_10"] = compute_ema(df["OBV"], 10)
    
    # 5. Price Position (High/Low Range)
    rolling_max_20 = high.rolling(window=20).max()
    rolling_min_20 = low.rolling(window=20).min()
    df["Stoch_Oscillator"] = ((close - rolling_min_20) / ((rolling_max_20 - rolling_min_20) + 1e-9)) * 100.0

    return df


if __name__ == "__main__":
    from data_loader import fetch_stock_data
    df = fetch_stock_data("RELIANCE")
    df_feat = add_all_technical_indicators(df)
    print("Computed indicator columns:", [c for c in df_feat.columns if c not in df.columns])
    print(df_feat.tail(3)[["Date", "Close", "RSI_14", "MACD_Hist", "BB_PctB", "NATR"]])
