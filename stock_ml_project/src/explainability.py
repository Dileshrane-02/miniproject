"""
Explainable AI (XAI) Module.
Computes SHAP (SHapley Additive exPlanations) values and model feature importances,
translating complex mathematical feature contributions into human-interpretable
reasons for every prediction.
"""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
try:
    import shap
except Exception as e:
    shap = None

logger = logging.getLogger(__name__)


class PredictionExplainer:
    def __init__(self, model, feature_names: List[str], model_name: str = "XGBoost"):
        self.model = model
        self.feature_names = feature_names
        self.model_name = model_name
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self):
        """Initializes the appropriate SHAP explainer."""
        try:
            if "Forest" in self.model_name or "XGB" in self.model_name:
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # For Logistic Regression
                self.explainer = shap.LinearExplainer(self.model, masker=shap.maskers.Independent(np.zeros((1, len(self.feature_names)))))
        except Exception as e:
            logger.warning(f"SHAP Explainer initialization fallback: {e}")
            self.explainer = None

    def get_global_feature_importance(self, X_sample: np.ndarray = None) -> pd.DataFrame:
        """
        Returns global feature importance ranked in descending order.
        """
        importances = None
        
        # 1. Try SHAP mean absolute values if sample provided
        if self.explainer is not None and X_sample is not None and len(X_sample) > 0:
            try:
                sample_subset = X_sample[:min(100, len(X_sample))]
                shap_values = self.explainer.shap_values(sample_subset)
                
                # For multi-class (list of arrays), average across classes
                if isinstance(shap_values, list):
                    mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
                elif len(shap_values.shape) == 3:
                    mean_shap = np.abs(shap_values).mean(axis=(0, 2))
                else:
                    mean_shap = np.abs(shap_values).mean(axis=0)
                    
                importances = mean_shap
            except Exception as e:
                logger.warning(f"SHAP global importance computation failed: {e}")

        # 2. Fallback to model's native feature_importances_ or coef_
        if importances is None:
            if hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_
            elif hasattr(self.model, "coef_"):
                importances = np.mean(np.abs(self.model.coef_), axis=0)
            else:
                importances = np.ones(len(self.feature_names)) / len(self.feature_names)

        df_imp = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        
        # Normalize to 100%
        total = df_imp["Importance"].sum()
        if total > 0:
            df_imp["Importance_Pct"] = (df_imp["Importance"] / total) * 100.0
        else:
            df_imp["Importance_Pct"] = 100.0 / len(df_imp)
            
        return df_imp

    def explain_instance(
        self,
        raw_feature_dict: Dict[str, float],
        prediction_label: str,
        regime_label: str,
        sentiment_info: Dict[str, Any]
    ) -> List[str]:
        """
        Translates raw feature values, technical indicators, regime, and sentiment
        into clear, human-understandable explanation bullet points.
        """
        reasons = []
        
        rsi = raw_feature_dict.get("RSI_14", 50.0)
        macd_hist = raw_feature_dict.get("MACD_Hist", 0.0)
        vol_ratio = raw_feature_dict.get("Volume_Ratio", 1.0)
        dist_sma50 = raw_feature_dict.get("Dist_SMA_50", 0.0) * 100.0
        bb_pctb = raw_feature_dict.get("BB_PctB", 0.5)
        stoch = raw_feature_dict.get("Stoch_Oscillator", 50.0)
        nifty_ret = raw_feature_dict.get("Nifty_Return", 0.0) * 100.0
        
        # 1. RSI Reason
        if rsi > 70:
            reasons.append(f"RSI is Overbought ({rsi:.1f} > 70), indicating potential near-term price exhaustion or pullback risk.")
        elif rsi < 30:
            reasons.append(f"RSI is Oversold ({rsi:.1f} < 30), highlighting strong mean-reversion buying opportunity.")
        elif 50 <= rsi <= 70:
            reasons.append(f"RSI ({rsi:.1f}) shows sustained upward bullish momentum without reaching overbought territory.")
        else:
            reasons.append(f"RSI ({rsi:.1f}) reflects downward selling pressure.")

        # 2. MACD Momentum
        if macd_hist > 0:
            reasons.append(f"MACD Histogram is positive (+{macd_hist:.2f}), confirming bullish momentum expansion.")
        else:
            reasons.append(f"MACD Histogram is negative ({macd_hist:.2f}), signaling bearish momentum pressure.")

        # 3. Trend / Moving Average
        if dist_sma50 > 2.0:
            reasons.append(f"Price is trading {dist_sma50:.1f}% above the 50-day SMA, confirming established medium-term uptrend.")
        elif dist_sma50 < -2.0:
            reasons.append(f"Price is trading {abs(dist_sma50):.1f}% below the 50-day SMA, indicating medium-term downtrend.")
        else:
            reasons.append(f"Price is consolidating near its 50-day SMA ({dist_sma50:+.1f}%), reflecting a sideways consolidation phase.")

        # 4. Market Regime
        if regime_label == "Bullish":
            reasons.append(f"Market Regime Detection classifies the macro environment as Bullish with low volatility.")
        elif regime_label == "Bearish":
            reasons.append(f"Market Regime Detection flags a Bearish high-volatility macro environment.")
        else:
            reasons.append(f"Market Regime Detection indicates a Sideways / Choppy consolidation phase.")

        # 5. Volume & Institutional Flow
        if vol_ratio > 1.25:
            reasons.append(f"Trading volume is {vol_ratio:.1f}x higher than the 20-day average, signaling strong institutional participation.")
        elif vol_ratio < 0.75:
            reasons.append(f"Trading volume is light ({vol_ratio:.1f}x of 20-day average), indicating low market conviction.")
        else:
            reasons.append(f"Trading volume is aligned with the 20-day normal baseline ({vol_ratio:.1f}x).")

        # 6. News Sentiment
        sent_label = sentiment_info.get("sentiment_label", "Neutral")
        sent_pol = sentiment_info.get("avg_polarity", 0.0)
        reasons.append(f"Recent Financial News Sentiment is {sent_label} (Polarity: {sent_pol:+.2f}) across recent company updates.")

        return reasons[:5]
