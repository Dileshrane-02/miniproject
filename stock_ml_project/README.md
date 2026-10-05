# AI-Based Stock Market Prediction & Analysis Using Technical Indicators, Market Regime Detection, and News Sentiment Analysis

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Market](https://img.shields.io/badge/Market-NSE%20India-orange.svg)](https://www.nseindia.com/)

A practical Machine Learning system for predicting next trading-day market direction (**BUY / HOLD / SELL**) for Indian equities listed on the **National Stock Exchange (NSE)**, benchmarked against the **NIFTY 50** index.

---

## 🚀 Key Features

1. **Multi-Source Data Acquisition**: Automated downloading of historical OHLCV data for Indian blue-chip stocks (RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN, ITC, LT) and the NIFTY 50 benchmark index via Yahoo Finance (`yfinance`), with local CSV caching.
2. **Comprehensive Feature Engineering**: Vectorized technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, ROC, OBV, Volatility, Stochastic Oscillator).
3. **Unsupervised Market Regime Detection**: Gaussian Mixture Model (GMM) clustering classifying market dynamics into **Bullish**, **Bearish**, and **Sideways / Choppy** states.
4. **Financial News Sentiment Analysis (NLP)**: Real-time RSS/API headline scraper with FinBERT / VADER sentiment scoring.
5. **Multi-Model ML Benchmarking**: Trains and compares **Logistic Regression**, **Random Forest**, and **XGBoost** on strictly chronological out-of-sample data.
6. **Strict Prevention of Data Leakage**: Time-based train/val/test split with scalers isolated to the training window.
7. **Explainable AI (XAI / SHAP)**: Plain-English explanations detailing the top factors driving every BUY/HOLD/SELL recommendation.
8. **Realistic Historical Backtesting**: Simulates trading strategy performance against passive Buy & Hold with transaction costs and slippage.
9. **Streamlit Web Dashboard**: User-friendly UI with interactive Plotly charts, model comparisons, and viva documentation.

---

## 📁 Project Architecture

```
stock_ml_project/
│
├── config.py                  # Global settings, stock tickers, indicator parameters, thresholds
├── requirements.txt           # Python dependencies
├── app.py                     # Streamlit web dashboard
├── README.md                  # Complete documentation and setup guide
│
├── data/                      # Cached historical CSV files for offline resilience
├── models/                    # Serialized best models, scalers, and metadata
├── notebooks/                 # Jupyter notebooks for experimentation
│
└── src/
    ├── data_loader.py         # Historical OHLCV, NIFTY 50, and news scrapers
    ├── indicators.py          # Vectorized technical indicator computations
    ├── regime_detection.py    # GMM / K-Means market regime clustering
    ├── sentiment.py           # FinBERT / VADER NLP sentiment engine
    ├── preprocessing.py       # Time-series splits and target formulation
    ├── feature_engineering.py # Consolidated feature matrix builder
    ├── train.py               # Model training, comparison, and evaluation
    ├── explainability.py      # SHAP feature importance & rationale generator
    ├── predict.py             # Real-time inference engine
    └── backtesting.py         # Portfolio simulation & performance analytics
```

---

## 🛠️ Quickstart Installation & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models and Test CLI Inference
```bash
python src/train.py
python src/predict.py
```

### 3. Launch Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Target Formulation & Return Threshold

The next trading-day return $R_{t+1}$ is formulated as:
$$R_{t+1} = \frac{\text{Close}_{t+1} - \text{Close}_t}{\text{Close}_t}$$

Classification Target $Y_t$:
- **BUY (2)**: $R_{t+1} > +0.5\%$ (or user threshold)
- **SELL (0)**: $R_{t+1} < -0.5\%$
- **HOLD (1)**: $-0.5\% \le R_{t+1} \le +0.5\%$

---

## ⚖️ Academic Disclaimer
*This project is developed solely for academic research, education, and college demonstration purposes. It does NOT constitute financial advice. Financial markets are subject to high volatility and risk.*
