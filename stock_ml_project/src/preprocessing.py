"""
Time-Series Data Preprocessing & Target Formulation Module.
Strictly prevents data leakage by:
1. Chronological (time-based) train/validation/test splitting without random shuffling.
2. Fitting scalers and statistics ONLY on the training split.
3. Aligning feature matrix X (day t) with target y (return of day t+1).
"""

import logging
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


def create_classification_target(
    df: pd.DataFrame,
    threshold: float = 0.005
) -> pd.DataFrame:
    """
    Constructs the 3-class classification target based on next-day return:
      Target = 2 (BUY)  if (Close_{t+1} - Close_t) / Close_t > +threshold
      Target = 0 (SELL) if (Close_{t+1} - Close_t) / Close_t < -threshold
      Target = 1 (HOLD) if -threshold <= (Close_{t+1} - Close_t) / Close_t <= +threshold
    """
    df_out = df.copy()
    
    # Next trading day return: Close(t+1) / Close(t) - 1
    df_out["Next_Day_Return"] = df_out["Close"].shift(-1) / df_out["Close"] - 1.0
    
    # Classify into 0 (SELL), 1 (HOLD), 2 (BUY)
    conditions = [
        df_out["Next_Day_Return"] > threshold,
        df_out["Next_Day_Return"] < -threshold
    ]
    choices = [2, 0]  # 2: BUY, 0: SELL
    
    df_out["Target"] = np.select(conditions, choices, default=1)  # 1: HOLD
    
    # Drop last row since Next_Day_Return is NaN
    df_out = df_out.dropna(subset=["Next_Day_Return"]).copy()
    df_out["Target"] = df_out["Target"].astype(int)
    
    return df_out


def time_based_train_val_test_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "Target",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Dict[str, Any]:
    """
    Splits the dataset strictly in chronological order into Train, Validation, and Test sets.
    Fits RobustScaler ONLY on the training partition to prevent lookahead data leakage.
    """
    total_len = len(df)
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    logger.info(
        f"Time-based split: Train={len(train_df)} ({train_df['Date'].min().date()} to {train_df['Date'].max().date()}), "
        f"Val={len(val_df)} ({val_df['Date'].min().date()} to {val_df['Date'].max().date()}), "
        f"Test={len(test_df)} ({test_df['Date'].min().date()} to {test_df['Date'].max().date()})"
    )
    
    X_train_raw = train_df[feature_cols].values
    y_train = train_df[target_col].values
    
    X_val_raw = val_df[feature_cols].values
    y_val = val_df[target_col].values
    
    X_test_raw = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    # Fit scaler strictly on X_train_raw
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    return {
        "scaler": scaler,
        "feature_names": feature_cols,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_val": X_val_scaled,
        "y_val": y_val,
        "X_test": X_test_scaled,
        "y_test": y_test
    }
