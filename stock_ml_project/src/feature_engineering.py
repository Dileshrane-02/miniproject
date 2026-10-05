"""
Feature Engineering Pipeline Module.
Integrates Stock Technical Indicators, Benchmark NIFTY 50 Movement,
Market Regime Probabilities, and Financial Sentiment into a unified ML matrix.
"""

import os
import sys
import logging
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from src.indicators import add_all_technical_indicators
from src.regime_detection import MarketRegimeDetector

logger = logging.getLogger(__name__)

# Standard ML Feature Column List
FEATURE_COLUMNS = [
    # Technical Momentum & Trend
    "RSI_14",
    "MACD_Hist",
    "MACD_Line",
    "Dist_SMA_10",
    "Dist_SMA_50",
    "SMA_Cross",
    "ROC_10",
    "ROC_5",
    "Stoch_Oscillator",
    # Volatility & Bands
    "BB_PctB",
    "BB_Bandwidth",
    "NATR",
    "Historical_Vol_20",
    # Volume Dynamics
    "Volume_Ratio",
    # Benchmark / Market Context
    "Nifty_Return",
    "Nifty_Vol_20",
    "Alpha_vs_Nifty",
    # Market Regime Detection
    "Regime_Code",
    "Regime_Prob_Bullish",
    "Regime_Prob_Bearish",
    # Sentiment Signals
    "Sentiment_Polarity"
]


def build_full_feature_dataset(
    df_stock: pd.DataFrame,
    df_nifty: pd.DataFrame,
    regime_detector: MarketRegimeDetector = None,
    fit_regime: bool = True
) -> Tuple[pd.DataFrame, MarketRegimeDetector, List[str]]:
    """
    Builds the complete feature set by merging stock indicators, benchmark returns,
    market regime classification, and sentiment scores.
    """
    # 1. Technical Indicators
    df_feat = add_all_technical_indicators(df_stock)
    
    # 2. Merge NIFTY 50 Benchmark Data
    df_feat = pd.merge(df_feat, df_nifty[["Date", "Nifty_Return", "Nifty_Vol_20"]], on="Date", how="left")
    df_feat["Nifty_Return"] = df_feat["Nifty_Return"].fillna(0.0)
    df_feat["Nifty_Vol_20"] = df_feat["Nifty_Vol_20"].bfill().fillna(0.15)
    
    # Calculate stock alpha/excess return over NIFTY
    df_feat["Alpha_vs_Nifty"] = df_feat["Daily_Return"] - df_feat["Nifty_Return"]
    
    # 3. Market Regime Detection
    if regime_detector is None:
        regime_detector = MarketRegimeDetector(n_regimes=config.REGIME_CLUSTERS, method="GMM")
        
    if fit_regime:
        # Fit on available data
        regime_detector.fit(df_feat)
        
    df_feat = regime_detector.predict_regimes(df_feat)
    
    # 4. Sentiment Signal (Base rolling proxy for historical records; live value applied during inference)
    # Price action momentum + volatility proxy for historical baseline sentiment
    if "Sentiment_Polarity" not in df_feat.columns:
        # Rolling sentiment proxy based on price drift and relative volume for historical backtesting
        rolling_momentum = df_feat["Daily_Return"].rolling(5).mean()
        df_feat["Sentiment_Polarity"] = np.tanh(rolling_momentum * 20.0).fillna(0.0)
        
    # Ensure all required columns exist
    for col in ["Regime_Prob_Bullish", "Regime_Prob_Bearish", "Regime_Prob_Sideways"]:
        if col not in df_feat.columns:
            df_feat[col] = 0.33
            
    # Drop warm-up rows containing NaNs from 50-day rolling indicators
    df_clean = df_feat.dropna(subset=["SMA_50", "NATR", "Historical_Vol_20", "RSI_14"]).copy().reset_index(drop=True)
    
    available_features = [col for col in FEATURE_COLUMNS if col in df_clean.columns]
    
    return df_clean, regime_detector, available_features


if __name__ == "__main__":
    from data_loader import fetch_stock_data, fetch_benchmark_data
    
    stock_df = fetch_stock_data("RELIANCE")
    nifty_df = fetch_benchmark_data()
    
    df_full, detector, features = build_full_feature_dataset(stock_df, nifty_df)
    print(f"Engineered dataset with {len(df_full)} rows and {len(features)} features:")
    print(features)
    print(df_full[features].tail(3))
