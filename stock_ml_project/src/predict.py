"""
Inference Engine Module.
Executes end-to-end prediction for next-day market direction (BUY/HOLD/SELL),
generates confidence score, regime detection, news sentiment, and SHAP-based explanations.
"""

import os
import sys
import logging
import joblib
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from src.data_loader import fetch_stock_data, fetch_benchmark_data, fetch_news_headlines
from src.sentiment import aggregate_sentiment
from src.regime_detection import MarketRegimeDetector
from src.feature_engineering import build_full_feature_dataset, FEATURE_COLUMNS
from src.train import train_and_compare_models
from src.explainability import PredictionExplainer

logger = logging.getLogger(__name__)


def predict_next_day_direction(
    stock_symbol: str = "RELIANCE",
    retrain_if_missing: bool = True
) -> Dict[str, Any]:
    """
    Complete real-time inference workflow:
    1. Fetches latest OHLCV stock and NIFTY 50 data.
    2. Fetches and scores latest financial news headlines.
    3. Loads or trains the model, scaler, and regime detector.
    4. Computes feature vector for the latest trading day.
    5. Outputs predicted direction, probability distribution, confidence score, and bullet explanations.
    """
    logger.info(f"Running next-day market direction inference for {stock_symbol}...")
    
    # 1. Fetch Data
    stock_info = config.SUPPORTED_STOCKS.get(stock_symbol, {"name": stock_symbol, "ticker": f"{stock_symbol}.NS", "sector": "Indian Equities"})
    df_stock = fetch_stock_data(stock_symbol)
    df_nifty = fetch_benchmark_data()
    
    # 2. News Sentiment
    headlines = fetch_news_headlines(stock_info["name"])
    sentiment_summary = aggregate_sentiment(headlines)
    
    # 3. Load or Train Model Artifacts
    model_path = config.MODELS_DIR / f"{stock_symbol}_best_model.joblib"
    scaler_path = config.MODELS_DIR / f"{stock_symbol}_scaler.joblib"
    regime_path = config.MODELS_DIR / f"{stock_symbol}_regime.joblib"
    
    train_metadata = None
    if not (model_path.exists() and scaler_path.exists() and regime_path.exists()):
        if retrain_if_missing:
            logger.info(f"Model artifacts not found for {stock_symbol}. Initiating training...")
            train_out = train_and_compare_models(stock_symbol)
            model = train_out["best_model"]
            scaler = train_out["scaler"]
            regime_detector = train_out["regime_detector"]
            best_model_name = train_out["best_model_name"]
            train_metadata = train_out["summary"]
        else:
            raise FileNotFoundError(f"Model artifacts missing for {stock_symbol}")
    else:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        regime_detector = MarketRegimeDetector.load(regime_path)
        best_model_name = type(model).__name__
        
    # 4. Feature Engineering on Full Series
    df_feats, _, feature_cols = build_full_feature_dataset(
        df_stock, df_nifty, regime_detector=regime_detector, fit_regime=False
    )
    
    # Inject live news sentiment into the latest row
    df_feats.loc[df_feats.index[-1], "Sentiment_Polarity"] = sentiment_summary["avg_polarity"]
    
    latest_row = df_feats.iloc[-1]
    latest_date = pd.to_datetime(latest_row["Date"]).strftime("%Y-%m-%d")
    latest_close = float(latest_row["Close"])
    
    # Extract feature vector
    X_latest_raw = df_feats[feature_cols].iloc[[-1]].values
    X_latest_scaled = scaler.transform(X_latest_raw)
    
    # 5. Model Inference
    pred_code = int(model.predict(X_latest_scaled)[0])
    pred_probs = model.predict_proba(X_latest_scaled)[0] if hasattr(model, "predict_proba") else [0.33, 0.34, 0.33]
    
    label_map = config.LABEL_MAP  # {0: "SELL", 1: "HOLD", 2: "BUY"}
    prediction_label = label_map.get(pred_code, "HOLD")
    
    prob_dict = {
        "SELL": round(float(pred_probs[0]) * 100.0, 1),
        "HOLD": round(float(pred_probs[1]) * 100.0, 1),
        "BUY": round(float(pred_probs[2]) * 100.0, 1)
    }
    confidence_score = prob_dict[prediction_label]
    
    # 6. Regime & Explanations
    current_regime = latest_row.get("Market_Regime", "Sideways / Choppy")
    raw_feature_dict = latest_row[feature_cols].to_dict()
    
    explainer = PredictionExplainer(model, feature_cols, model_name=best_model_name)
    reasons = explainer.explain_instance(
        raw_feature_dict=raw_feature_dict,
        prediction_label=prediction_label,
        regime_label=current_regime,
        sentiment_info=sentiment_summary
    )
    
    return {
        "stock_symbol": stock_symbol,
        "company_name": stock_info["name"],
        "sector": stock_info["sector"],
        "as_of_date": latest_date,
        "latest_close": latest_close,
        "prediction": prediction_label,
        "confidence_pct": confidence_score,
        "class_probabilities": prob_dict,
        "market_regime": current_regime,
        "news_sentiment": sentiment_summary["sentiment_label"],
        "sentiment_polarity": sentiment_summary["avg_polarity"],
        "reasons": reasons,
        "latest_indicators": {
            "RSI_14": round(float(latest_row["RSI_14"]), 2),
            "MACD_Hist": round(float(latest_row["MACD_Hist"]), 2),
            "SMA_10": round(float(latest_row["SMA_10"]), 2),
            "SMA_50": round(float(latest_row["SMA_50"]), 2),
            "NATR": round(float(latest_row["NATR"]), 2),
            "BB_PctB": round(float(latest_row["BB_PctB"]), 2),
            "Volume_Ratio": round(float(latest_row["Volume_Ratio"]), 2)
        },
        "news_headlines": headlines,
        "df_feats": df_feats,
        "train_metadata": train_metadata
    }


if __name__ == "__main__":
    result = predict_next_day_direction("RELIANCE")
    print(f"\n--- Prediction for {result['company_name']} ({result['as_of_date']}) ---")
    print(f"Latest Price: INR {result['latest_close']:.2f}")
    print(f"ML Direction Signal: {result['prediction']} (Confidence: {result['confidence_pct']}%)")
    print(f"Class Probabilities: {result['class_probabilities']}")
    print(f"Market Regime: {result['market_regime']}")
    print(f"News Sentiment: {result['news_sentiment']} ({result['sentiment_polarity']:+.2f})")
    print("\nKey Contributing Factors:")
    for i, r in enumerate(result["reasons"], 1):
        print(f"  {i}. {r}")
