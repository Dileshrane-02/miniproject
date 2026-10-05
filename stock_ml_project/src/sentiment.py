"""
Financial News Sentiment Analysis Module.
Provides NLP sentiment extraction using FinBERT (financial domain BERT model)
with an automatic graceful fallback to VADER (Valence Aware Dictionary and sEntiment Reasoner)
for fast, reliable execution on any machine.
"""

import re
import logging
from typing import List, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Try importing FinBERT / Transformers, otherwise prepare VADER fallback
_FINBERT_PIPELINE = None
_VADER_ANALYZER = None
_MODEL_BACKEND = "uninitialized"


def _clean_headline(text: str) -> str:
    """Preprocesses and normalizes news text."""
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Remove HTML entities
    text = re.sub(r"&\w+;", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _init_models():
    """Initializes FinBERT or VADER according to available hardware/libraries."""
    global _FINBERT_PIPELINE, _VADER_ANALYZER, _MODEL_BACKEND
    
    if _MODEL_BACKEND != "uninitialized":
        return

    # 1. Try FinBERT
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        logger.info("Attempting to load ProsusAI/finbert pipeline...")
        # Use a lightweight pipeline with strict timeout / CPU friendly settings
        _FINBERT_PIPELINE = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1  # CPU by default for stability
        )
        _MODEL_BACKEND = "FinBERT"
        logger.info("Successfully loaded FinBERT pipeline.")
        return
    except Exception as e:
        logger.info(f"FinBERT initialization skipped/failed ({e}). Falling back to VADER...")

    # 2. Try VADER from NLTK
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            
        _VADER_ANALYZER = SentimentIntensityAnalyzer()
        # Augment with financial domain terms
        financial_lexicon = {
            "surge": 2.0, "jump": 1.8, "rally": 2.2, "bullish": 2.5, "outperform": 2.0,
            "beat": 1.8, "profit": 1.5, "growth": 1.5, "breakout": 1.9, "record": 1.5,
            "plunge": -2.5, "slump": -2.2, "tumble": -2.3, "bearish": -2.5, "miss": -1.8,
            "loss": -1.9, "downgrade": -2.0, "fall": -1.5, "drop": -1.5, "fraud": -3.0
        }
        _VADER_ANALYZER.lexicon.update(financial_lexicon)
        _MODEL_BACKEND = "VADER"
        logger.info("Successfully initialized VADER financial sentiment analyzer.")
    except Exception as e:
        logger.warning(f"NLTK VADER failed ({e}). Using rule-based fallback lexicon.")
        _MODEL_BACKEND = "RuleBased"


def analyze_single_text(text: str) -> Dict[str, Any]:
    """
    Analyzes sentiment of a single headline.
    Returns:
      - label: "Positive", "Neutral", "Negative"
      - polarity: float [-1.0 to 1.0]
      - pos_score, neu_score, neg_score
      - backend: 'FinBERT' or 'VADER' or 'RuleBased'
    """
    _init_models()
    cleaned = _clean_headline(text)
    if not cleaned:
        return {"label": "Neutral", "polarity": 0.0, "pos_score": 0.33, "neu_score": 0.34, "neg_score": 0.33, "backend": _MODEL_BACKEND}

    if _MODEL_BACKEND == "FinBERT" and _FINBERT_PIPELINE is not None:
        try:
            result = _FINBERT_PIPELINE(cleaned[:512])[0]
            raw_label = result["label"].capitalize()  # Positive, Negative, Neutral
            score = float(result["score"])
            
            if raw_label == "Positive":
                polarity = score
                pos, neu, neg = score, (1 - score) * 0.5, (1 - score) * 0.5
            elif raw_label == "Negative":
                polarity = -score
                pos, neu, neg = (1 - score) * 0.5, (1 - score) * 0.5, score
            else:
                polarity = 0.0
                pos, neu, neg = (1 - score) * 0.5, score, (1 - score) * 0.5
                
            return {
                "label": raw_label,
                "polarity": round(polarity, 4),
                "pos_score": round(pos, 4),
                "neu_score": round(neu, 4),
                "neg_score": round(neg, 4),
                "backend": "FinBERT"
            }
        except Exception as e:
            logger.warning(f"FinBERT inference error: {e}. Falling back to lexicon.")

    if _VADER_ANALYZER is not None:
        scores = _VADER_ANALYZER.polarity_scores(cleaned)
        compound = scores["compound"]
        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
            
        return {
            "label": label,
            "polarity": round(compound, 4),
            "pos_score": round(scores["pos"], 4),
            "neu_score": round(scores["neu"], 4),
            "neg_score": round(scores["neg"], 4),
            "backend": "VADER"
        }

    # Rule-Based fallback
    pos_words = {"surge", "gain", "profit", "bull", "high", "rise", "jump", "growth", "strong", "rally"}
    neg_words = {"fall", "drop", "loss", "bear", "low", "slump", "miss", "plunge", "weak", "down"}
    words = set(cleaned.lower().split())
    pos_count = len(words & pos_words)
    neg_count = len(words & neg_words)
    
    if pos_count > neg_count:
        return {"label": "Positive", "polarity": 0.5, "pos_score": 0.7, "neu_score": 0.2, "neg_score": 0.1, "backend": "RuleBased"}
    elif neg_count > pos_count:
        return {"label": "Negative", "polarity": -0.5, "pos_score": 0.1, "neu_score": 0.2, "neg_score": 0.7, "backend": "RuleBased"}
    else:
        return {"label": "Neutral", "polarity": 0.0, "pos_score": 0.2, "neu_score": 0.6, "neg_score": 0.2, "backend": "RuleBased"}


def aggregate_sentiment(headlines: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Computes aggregated sentiment metrics over a list of news headlines.
    """
    if not headlines:
        return {
            "avg_polarity": 0.0,
            "sentiment_label": "Neutral",
            "pos_ratio": 0.33,
            "neu_ratio": 0.34,
            "neg_ratio": 0.33,
            "headline_count": 0,
            "detailed_results": []
        }

    detailed = []
    polarities = []
    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    
    for h in headlines:
        text = h.get("headline", "")
        res = analyze_single_text(text)
        detailed.append({
            "headline": text,
            "source": h.get("source", "News"),
            "published_at": h.get("published_at", ""),
            "label": res["label"],
            "polarity": res["polarity"]
        })
        polarities.append(res["polarity"])
        counts[res["label"]] += 1

    total = len(headlines)
    avg_pol = float(np.mean(polarities)) if polarities else 0.0
    
    if avg_pol > 0.10:
        overall_label = "Positive"
    elif avg_pol < -0.10:
        overall_label = "Negative"
    else:
        overall_label = "Neutral"

    return {
        "avg_polarity": round(avg_pol, 4),
        "sentiment_label": overall_label,
        "pos_ratio": round(counts["Positive"] / total, 3),
        "neu_ratio": round(counts["Neutral"] / total, 3),
        "neg_ratio": round(counts["Negative"] / total, 3),
        "headline_count": total,
        "detailed_results": detailed
    }


if __name__ == "__main__":
    sample_news = [
        {"headline": "Reliance Industries reports 15% surge in quarterly net profit beating analyst estimates.", "source": "Reuters"},
        {"headline": "TCS faces revenue slowdown amid US macroeconomic headwinds and IT budget cuts.", "source": "Bloomberg"},
        {"headline": "HDFC Bank holds steady interest rates following RBI monetary policy announcement.", "source": "Mint"}
    ]
    
    summary = aggregate_sentiment(sample_news)
    print(f"Aggregated Sentiment: {summary['sentiment_label']} (Avg Polarity: {summary['avg_polarity']})")
    for d in summary["detailed_results"]:
        print(f"[{d['label']}] ({d['polarity']}) - {d['headline']}")
