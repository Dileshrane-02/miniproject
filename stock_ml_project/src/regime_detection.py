"""
Market Regime Detection Module.
Uses Gaussian Mixture Models (GMM) and K-Means Clustering on rolling returns and volatility
to classify the market environment into:
1. Bullish (Positive Drift, Low-to-Moderate Volatility)
2. Bearish (Negative Drift, High Volatility)
3. Sideways / Choppy (Near-Zero Drift, Moderate Volatility)
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    def __init__(self, n_regimes: int = 3, method: str = "GMM", random_state: int = 42):
        self.n_regimes = n_regimes
        self.method = method.upper()
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.regime_map = {}  # Maps raw cluster ID to ordered label (0: Sideways, 1: Bullish, 2: Bearish)
        
    def _extract_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts rolling return and rolling annualized volatility for clustering."""
        feats = pd.DataFrame(index=df.index)
        
        # 20-day rolling return
        if "Daily_Return" in df.columns:
            ret = df["Daily_Return"]
        else:
            ret = df["Close"].pct_change()
            
        feats["Rolling_Ret_20"] = ret.rolling(window=20).mean() * 252  # Annualized drift
        feats["Rolling_Vol_20"] = ret.rolling(window=20).std() * np.sqrt(252)  # Annualized volatility
        
        # Relative price position to 50 SMA
        if "SMA_50" in df.columns:
            feats["SMA_Dist"] = (df["Close"] - df["SMA_50"]) / (df["SMA_50"] + 1e-9)
        else:
            sma_50 = df["Close"].rolling(window=50).mean()
            feats["SMA_Dist"] = (df["Close"] - sma_50) / (sma_50 + 1e-9)
            
        return feats.bfill().fillna(0)

    def fit(self, df: pd.DataFrame):
        """Fits GMM or KMeans on historical time-series features and establishes ordered regime mapping."""
        feats = self._extract_regime_features(df)
        X_scaled = self.scaler.fit_transform(feats)
        
        if self.method == "GMM":
            self.model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type="full",
                random_state=self.random_state,
                max_iter=200
            )
        else:
            self.model = KMeans(
                n_clusters=self.n_regimes,
                random_state=self.random_state,
                n_init=10
            )
            
        self.model.fit(X_scaled)
        cluster_labels = self.model.predict(X_scaled)
        
        # Analyze cluster centers to assign domain-consistent names
        cluster_stats = []
        for i in range(self.n_regimes):
            idx = np.where(cluster_labels == i)[0]
            if len(idx) > 0:
                mean_ret = feats["Rolling_Ret_20"].iloc[idx].mean()
                mean_vol = feats["Rolling_Vol_20"].iloc[idx].mean()
            else:
                mean_ret, mean_vol = 0.0, 0.0
            cluster_stats.append((i, mean_ret, mean_vol))
            
        # Sort clusters: Highest return -> Bullish (1), Lowest return / highest risk -> Bearish (2), Intermediate -> Sideways (0)
        # Sort by mean return ascending
        sorted_by_ret = sorted(cluster_stats, key=lambda x: x[1])
        bearish_id = sorted_by_ret[0][0]     # Lowest return
        bullish_id = sorted_by_ret[-1][0]    # Highest return
        remaining_ids = [x[0] for x in sorted_by_ret if x[0] not in (bearish_id, bullish_id)]
        sideways_id = remaining_ids[0] if remaining_ids else 0
        
        self.regime_map = {
            bullish_id: "Bullish",
            bearish_id: "Bearish",
            sideways_id: "Sideways / Choppy"
        }
        
        logger.info(f"Fitted {self.method} Regime Detector. Regime Mapping: {self.regime_map}")
        return self

    def predict_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns regime labels and probabilities to the entire dataframe.
        """
        feats = self._extract_regime_features(df)
        X_scaled = self.scaler.transform(feats)
        
        raw_clusters = self.model.predict(X_scaled)
        regime_names = [self.regime_map.get(c, "Sideways / Choppy") for c in raw_clusters]
        
        result = df.copy()
        result["Market_Regime"] = regime_names
        
        # Numerical encoding for ML feature: 0: Sideways, 1: Bullish, 2: Bearish
        regime_num_map = {"Sideways / Choppy": 0, "Bullish": 1, "Bearish": 2}
        result["Regime_Code"] = [regime_num_map.get(name, 0) for name in regime_names]
        
        if self.method == "GMM":
            probs = self.model.predict_proba(X_scaled)
            # Find cluster index for each name
            for raw_id, name in self.regime_map.items():
                safe_name = name.split(" ")[0]
                result[f"Regime_Prob_{safe_name}"] = probs[:, raw_id]
        else:
            result["Regime_Prob_Bullish"] = (result["Regime_Code"] == 1).astype(float)
            result["Regime_Prob_Bearish"] = (result["Regime_Code"] == 2).astype(float)
            result["Regime_Prob_Sideways"] = (result["Regime_Code"] == 0).astype(float)
            
        return result

    def save(self, filepath: str):
        joblib.dump(self, filepath)
        
    @staticmethod
    def load(filepath: str):
        return joblib.load(filepath)


if __name__ == "__main__":
    from data_loader import fetch_stock_data
    from indicators import add_all_technical_indicators
    
    df = fetch_stock_data("RELIANCE")
    df = add_all_technical_indicators(df)
    
    detector = MarketRegimeDetector(n_regimes=3, method="GMM")
    detector.fit(df)
    df_regime = detector.predict_regimes(df)
    
    print("Regime distribution:")
    print(df_regime["Market_Regime"].value_counts())
    print("\nRecent regime samples:")
    print(df_regime[["Date", "Close", "Market_Regime", "Regime_Code"]].tail(5))
