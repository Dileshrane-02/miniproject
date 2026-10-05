"""
Model Training and Evaluation Pipeline.
Trains and compares Logistic Regression, Random Forest, and XGBoost classifiers
on strictly chronological time-series splits, computing Accuracy, Precision,
Recall, F1-score, Confusion Matrix, and ROC-AUC.
"""

import os
import sys
import json
import logging
import joblib
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from src.data_loader import fetch_stock_data, fetch_benchmark_data
from src.feature_engineering import build_full_feature_dataset
from src.preprocessing import create_classification_target, time_based_train_val_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_classifier(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Computes all standard classification metrics and multi-class ROC-AUC."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    # Multiclass ROC-AUC (One-vs-Rest)
    auc_score = 0.5
    if y_prob is not None:
        try:
            # Handle cases where not all classes are present in y_test
            present_classes = np.unique(y_test)
            if len(present_classes) > 1:
                auc_score = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"))
        except Exception as e:
            logger.warning(f"ROC-AUC computation skipped: {e}")
            auc_score = 0.5

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "roc_auc": round(auc_score, 4),
        "confusion_matrix": cm,
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist() if y_prob is not None else []
    }


def train_and_compare_models(
    stock_symbol: str = "RELIANCE",
    threshold: float = config.DEFAULT_RETURN_THRESHOLD
) -> Dict[str, Any]:
    """
    Complete pipeline: downloads data, computes indicators, regimes, targets,
    splits chronologically, trains Logistic Regression, Random Forest, XGBoost,
    evaluates them, and saves the best model artifacts.
    """
    logger.info(f"Starting training pipeline for {stock_symbol} with threshold {threshold}...")
    
    # 1. Acquire Data
    df_stock = fetch_stock_data(stock_symbol)
    df_nifty = fetch_benchmark_data()
    
    # 2. Build Full Features
    df_feats, regime_detector, feature_cols = build_full_feature_dataset(df_stock, df_nifty)
    
    # 3. Create Classification Target
    df_dataset = create_classification_target(df_feats, threshold=threshold)
    
    # 4. Strict Chronological Split
    split_data = time_based_train_val_test_split(
        df=df_dataset,
        feature_cols=feature_cols,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO
    )
    
    X_train, y_train = split_data["X_train"], split_data["y_train"]
    X_val, y_val = split_data["X_val"], split_data["y_val"]
    X_test, y_test = split_data["X_test"], split_data["y_test"]
    
    # 5. Initialize Candidate Models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=0.1,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss",
            n_jobs=-1
        )
    }
    
    results = {}
    best_model_name = None
    best_score = -1.0
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Test evaluation
        metrics = evaluate_classifier(model, X_test, y_test)
        results[name] = {
            "model": model,
            "metrics": metrics
        }
        logger.info(f"{name} Results -> Acc: {metrics['accuracy']:.4f}, F1-W: {metrics['f1_weighted']:.4f}, AUC: {metrics['roc_auc']:.4f}")
        
        # Selection criterion: weighted F1 + accuracy
        combined_score = metrics["f1_weighted"] + metrics["accuracy"]
        if combined_score > best_score:
            best_score = combined_score
            best_model_name = name

    logger.info(f"Best performing model selected: {best_model_name}")
    best_model = results[best_model_name]["model"]
    
    # 6. Save Artifacts for Inference and App
    model_save_path = config.MODELS_DIR / f"{stock_symbol}_best_model.joblib"
    scaler_save_path = config.MODELS_DIR / f"{stock_symbol}_scaler.joblib"
    regime_save_path = config.MODELS_DIR / f"{stock_symbol}_regime.joblib"
    
    joblib.dump(best_model, model_save_path)
    joblib.dump(split_data["scaler"], scaler_save_path)
    regime_detector.save(regime_save_path)
    
    # Save training summary JSON
    summary = {
        "stock": stock_symbol,
        "best_model_name": best_model_name,
        "feature_cols": feature_cols,
        "threshold": threshold,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "models_performance": {
            name: {k: v for k, v in res["metrics"].items() if k not in ("y_pred", "y_prob")}
            for name, res in results.items()
        }
    }
    
    with open(config.MODELS_DIR / f"{stock_symbol}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    return {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "scaler": split_data["scaler"],
        "regime_detector": regime_detector,
        "feature_cols": feature_cols,
        "results": results,
        "test_df": split_data["test_df"],
        "y_test": y_test,
        "X_test": X_test,
        "summary": summary
    }


if __name__ == "__main__":
    out = train_and_compare_models("RELIANCE")
    print("\nModel Comparison Table:")
    for m_name, m_data in out["summary"]["models_performance"].items():
        print(f"{m_name:20s} | Acc: {m_data['accuracy']:.2%} | F1: {m_data['f1_weighted']:.4f} | AUC: {m_data['roc_auc']:.4f}")
