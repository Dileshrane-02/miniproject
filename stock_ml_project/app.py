"""
Streamlit Web Application for Indian Stock Market AI Prediction & Analysis.
Features:
- Real-time stock data fetching (NSE)
- Technical Indicator Overlays & Candlestick Charts
- Market Regime Classification (GMM/K-Means)
- NLP Financial News Sentiment Analysis (FinBERT / VADER)
- Next-Day Market Direction Prediction (BUY / HOLD / SELL)
- XAI / SHAP Factor Explanations
- Model Performance Comparison
- Historical Strategy Backtesting vs. Buy & Hold Benchmark
"""

import os
import sys
import logging
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Indian Stock Market AI Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import config
from src.data_loader import fetch_stock_data, fetch_benchmark_data, fetch_news_headlines
from src.sentiment import aggregate_sentiment
from src.indicators import add_all_technical_indicators
from src.regime_detection import MarketRegimeDetector
from src.feature_engineering import build_full_feature_dataset, FEATURE_COLUMNS
from src.train import train_and_compare_models
from src.explainability import PredictionExplainer
from src.backtesting import run_backtest

# Custom CSS for modern polished styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .signal-buy {
        background-color: #DCFCE7;
        color: #166534;
        font-size: 1.8rem;
        font-weight: 800;
        padding: 8px 16px;
        border-radius: 8px;
        text-align: center;
        border: 2px solid #22C55E;
    }
    .signal-hold {
        background-color: #FEF9C3;
        color: #854D0E;
        font-size: 1.8rem;
        font-weight: 800;
        padding: 8px 16px;
        border-radius: 8px;
        text-align: center;
        border: 2px solid #EAB308;
    }
    .signal-sell {
        background-color: #FEE2E2;
        color: #991B1B;
        font-size: 1.8rem;
        font-weight: 800;
        padding: 8px 16px;
        border-radius: 8px;
        text-align: center;
        border: 2px solid #EF4444;
    }
    .reason-box {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
        padding: 12px 18px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Sidebar Configuration
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/bullish.png", width=64)
st.sidebar.title("Configuration")

selected_stock = st.sidebar.selectbox(
    "Select Indian Stock (NSE):",
    options=list(config.SUPPORTED_STOCKS.keys()),
    index=0,
    format_func=lambda s: f"{s} - {config.SUPPORTED_STOCKS[s]['name']}"
)

stock_info = config.SUPPORTED_STOCKS[selected_stock]

threshold_pct = st.sidebar.slider(
    "Classification Threshold (±%):",
    min_value=0.2,
    max_value=1.5,
    value=float(config.DEFAULT_RETURN_THRESHOLD * 100.0),
    step=0.1,
    help="Daily return threshold required to trigger a BUY or SELL signal instead of HOLD."
) / 100.0

retrain_btn = st.sidebar.button("🔄 Retrain Models & Refresh Data", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Company:** {stock_info['name']}")
st.sidebar.markdown(f"**NSE Ticker:** `{stock_info['ticker']}`")
st.sidebar.markdown(f"**Sector:** {stock_info['sector']}")
st.sidebar.markdown(f"**Benchmark:** NIFTY 50 (`^NSEI`)")
st.sidebar.caption("⚠️ For Academic & Educational Research Only. Not Financial Advice.")


# -------------------------------------------------------------
# Caching & Pipeline Execution
# -------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_all_data(stock_symbol: str, threshold: float):
    # Acquire stock and index data
    df_stock = fetch_stock_data(stock_symbol)
    df_nifty = fetch_benchmark_data()
    
    # Train/compare models
    train_results = train_and_compare_models(stock_symbol, threshold=threshold)
    
    # Fetch news headlines
    headlines = fetch_news_headlines(config.SUPPORTED_STOCKS[stock_symbol]["name"])
    sentiment_data = aggregate_sentiment(headlines)
    
    return df_stock, df_nifty, train_results, headlines, sentiment_data

with st.spinner(f"Fetching market data, computing indicators, and evaluating ML models for {selected_stock}..."):
    if retrain_btn:
        st.cache_data.clear()
    df_stock, df_nifty, train_res, news_headlines, sentiment_summary = load_all_data(selected_stock, threshold_pct)


# Extract pipeline artifacts
best_model = train_res["best_model"]
best_model_name = train_res["best_model_name"]
scaler = train_res["scaler"]
regime_detector = train_res["regime_detector"]
feature_cols = train_res["feature_cols"]
test_df = train_res["test_df"]
X_test = train_res["X_test"]
y_test = train_res["y_test"]

# Build full features for latest inference
df_feats, _, _ = build_full_feature_dataset(
    df_stock, df_nifty, regime_detector=regime_detector, fit_regime=False
)
df_feats.loc[df_feats.index[-1], "Sentiment_Polarity"] = sentiment_summary["avg_polarity"]

# Inference on latest row
latest_row = df_feats.iloc[-1]
latest_date = pd.to_datetime(latest_row["Date"]).strftime("%d %b %Y")
latest_close = float(latest_row["Close"])
prev_close = float(df_feats.iloc[-2]["Close"])
price_change = latest_close - prev_close
price_change_pct = (price_change / prev_close) * 100.0

X_latest_scaled = scaler.transform(df_feats[feature_cols].iloc[[-1]].values)
pred_code = int(best_model.predict(X_latest_scaled)[0])
pred_probs = best_model.predict_proba(X_latest_scaled)[0] if hasattr(best_model, "predict_proba") else [0.33, 0.34, 0.33]

pred_label = config.LABEL_MAP.get(pred_code, "HOLD")
confidence = float(pred_probs[pred_code]) * 100.0
current_regime = latest_row.get("Market_Regime", "Sideways / Choppy")

# Generate SHAP & Factor Explanations
explainer = PredictionExplainer(best_model, feature_cols, model_name=best_model_name)
reasons = explainer.explain_instance(
    raw_feature_dict=latest_row[feature_cols].to_dict(),
    prediction_label=pred_label,
    regime_label=current_regime,
    sentiment_info=sentiment_summary
)


# -------------------------------------------------------------
# Main Header & Top Metric Banners
# -------------------------------------------------------------
st.markdown(f'<div class="main-header">📈 {stock_info["name"]} ({selected_stock})</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">AI-Based Market Direction Classifier | As of <b>{latest_date}</b> | Benchmark: <b>NIFTY 50</b></div>', unsafe_allow_html=True)

# Top 5 KPI Metrics
col1, col2, col3, col4, col5 = st.columns([1.2, 1.3, 1.2, 1.2, 1.1])

with col1:
    st.metric(
        label="Latest Close (NSE)",
        value=f"₹{latest_close:,.2f}",
        delta=f"{price_change:+,.2f} ({price_change_pct:+.2f}%)"
    )

with col2:
    st.markdown("**Next-Day Signal**")
    if pred_label == "BUY":
        st.markdown('<div class="signal-buy">🟢 BUY</div>', unsafe_allow_html=True)
    elif pred_label == "SELL":
        st.markdown('<div class="signal-sell">🔴 SELL</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="signal-hold">🟡 HOLD</div>', unsafe_allow_html=True)

with col3:
    st.metric(
        label="Signal Confidence",
        value=f"{confidence:.1f}%",
        delta=f"Prob: {pred_label}"
    )

with col4:
    regime_icon = "🟢" if current_regime == "Bullish" else ("🔴" if current_regime == "Bearish" else "🟡")
    st.metric(
        label="Market Regime",
        value=f"{regime_icon} {current_regime.split(' ')[0]}",
        delta=f"GMM State"
    )

with col5:
    sent_label = sentiment_summary["sentiment_label"]
    sent_icon = "🟢" if sent_label == "Positive" else ("🔴" if sent_label == "Negative" else "🟡")
    st.metric(
        label="News Sentiment",
        value=f"{sent_icon} {sent_label}",
        delta=f"Pol: {sentiment_summary['avg_polarity']:+.2f}"
    )

st.markdown("---")

# -------------------------------------------------------------
# Dashboard Navigation Tabs
# -------------------------------------------------------------
tabs = st.tabs([
    "🎯 Prediction & Rationale",
    "📊 Technical Charts & Overlays",
    "🌐 Market Regime Detection",
    "📰 Financial News NLP",
    "🏆 Model Evaluation & Comparison",
    "📈 Historical Backtest",
    "📚 Project & Viva Documentation"
])


# =============================================================
# TAB 1: Prediction & Rationale
# =============================================================
with tabs[0]:
    st.subheader("🤖 AI Market Direction Prediction Rationale")
    
    col_left, col_right = st.columns([1.4, 1.0])
    
    with col_left:
        st.markdown("#### 🔍 Why did the AI predict this direction?")
        for idx, reason in enumerate(reasons, 1):
            st.markdown(f'<div class="reason-box"><b>{idx}.</b> {reason}</div>', unsafe_allow_html=True)
            
        st.caption("Explanations synthesized from SHAP feature contributions, oscillator states, macro regime detector, and NLP news sentiment.")

    with col_right:
        st.markdown("#### 🎯 Class Probability Distribution")
        prob_df = pd.DataFrame({
            "Signal": ["SELL", "HOLD", "BUY"],
            "Probability (%)": [pred_probs[0]*100, pred_probs[1]*100, pred_probs[2]*100],
            "Color": ["#EF4444", "#EAB308", "#22C55E"]
        })
        
        fig_prob = go.Figure(go.Bar(
            x=prob_df["Probability (%)"],
            y=prob_df["Signal"],
            orientation="h",
            marker_color=prob_df["Color"],
            text=[f"{p:.1f}%" for p in prob_df["Probability (%)"]],
            textposition="auto"
        ))
        fig_prob.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(range=[0, 100], title="Probability (%)"),
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_prob, use_container_width=True)
        
        st.info(f"**Best Model:** `{best_model_name}` | **Return Threshold:** `±{threshold_pct*100:.1f}%`")


# =============================================================
# TAB 2: Technical Charts & Overlays
# =============================================================
with tabs[1]:
    st.subheader("📊 Interactive Technical Analysis & Indicators")
    
    # Range selector
    chart_lookback = st.slider("Historical Lookback Window (Trading Days):", min_value=60, max_value=len(df_feats), value=250, step=30)
    df_plot = df_feats.tail(chart_lookback).copy()
    
    # 4-Subplot Interactive Plotly Chart
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.18, 0.17],
        subplot_titles=("Price (OHLC) + Bollinger Bands + Moving Averages", "Volume & Volume SMA", "RSI (14)", "MACD (12, 26, 9)")
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_plot["Date"],
        open=df_plot["Open"],
        high=df_plot["High"],
        low=df_plot["Low"],
        close=df_plot["Close"],
        name="OHLC"
    ), row=1, col=1)
    
    # SMA & Bollinger Bands
    fig.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot["SMA_10"], line=dict(color="#3B82F6", width=1.2), name="SMA 10"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot["SMA_50"], line=dict(color="#F97316", width=1.5), name="SMA 50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot["BB_Upper"], line=dict(color="#9CA3AF", dash="dot", width=1), name="BB Upper"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot["BB_Lower"], line=dict(color="#9CA3AF", dash="dot", width=1), fill="tonexty", fillcolor="rgba(156,163,175,0.1)", name="BB Lower"), row=1, col=1)
    
    # Volume
    colors = ['#22C55E' if row['Close'] >= row['Open'] else '#EF4444' for _, row in df_plot.iterrows()]
    fig.add_trace(go.Bar(x=df_plot["Date"], y=df_plot["Volume"], marker_color=colors, name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot["Volume_SMA_20"], line=dict(color="#6366F1", width=1.2), name="Vol SMA 20"), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot["RSI_14"], line=dict(color="#8B5CF6", width=1.5), name="RSI (14)"), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot["MACD_Line"], line=dict(color="#2563EB", width=1.2), name="MACD Line"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot["MACD_Signal"], line=dict(color="#DC2626", width=1.2), name="MACD Signal"), row=4, col=1)
    hist_colors = ['#22C55E' if v >= 0 else '#EF4444' for v in df_plot["MACD_Hist"]]
    fig.add_trace(go.Bar(x=df_plot["Date"], y=df_plot["MACD_Hist"], marker_color=hist_colors, name="MACD Hist"), row=4, col=1)
    
    fig.update_layout(
        height=750,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================
# TAB 3: Market Regime Detection
# =============================================================
with tabs[2]:
    st.subheader("🌐 Unsupervised Market Regime Detection")
    st.markdown("""
    Market Regime Detection classifies market states into **Bullish**, **Bearish**, and **Sideways / Choppy**
    using a **Gaussian Mixture Model (GMM)** fitted on rolling annualized return drift and rolling annualized volatility.
    """)
    
    col_r1, col_r2 = st.columns([1.2, 1.0])
    
    with col_r1:
        # Regime timeline chart
        fig_regime = go.Figure()
        regime_palette = {"Bullish": "#22C55E", "Bearish": "#EF4444", "Sideways / Choppy": "#EAB308"}
        
        for r_name, r_color in regime_palette.items():
            sub = df_feats[df_feats["Market_Regime"] == r_name]
            fig_regime.add_trace(go.Scatter(
                x=sub["Date"],
                y=sub["Close"],
                mode="markers",
                marker=dict(size=4, color=r_color),
                name=r_name
            ))
            
        fig_regime.update_layout(
            title="Historical Stock Price Segmented by Detected Market Regime",
            height=380,
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis_title="Close Price (INR)"
        )
        st.plotly_chart(fig_regime, use_container_width=True)

    with col_r2:
        # Regime Distribution Pie
        regime_counts = df_feats["Market_Regime"].value_counts().reset_index()
        regime_counts.columns = ["Regime", "Count"]
        
        fig_pie = go.Figure(go.Pie(
            labels=regime_counts["Regime"],
            values=regime_counts["Count"],
            marker=dict(colors=[regime_palette.get(r, "#9CA3AF") for r in regime_counts["Regime"]]),
            hole=0.45
        ))
        fig_pie.update_layout(
            title="Historical Regime Proportion",
            height=380,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_pie, use_container_width=True)


# =============================================================
# TAB 4: News Sentiment Analysis
# =============================================================
with tabs[3]:
    st.subheader("📰 Real-Time Financial News & NLP Sentiment Analysis")
    
    col_n1, col_n2 = st.columns([1.2, 1.0])
    
    with col_n1:
        st.markdown(f"#### Live News Feed for {stock_info['name']}")
        for item in sentiment_summary["detailed_results"]:
            badge_color = "#22C55E" if item["label"] == "Positive" else ("#EF4444" if item["label"] == "Negative" else "#EAB308")
            st.markdown(f"""
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-size: 0.8rem; color: #64748B;"><b>{item['source']}</b> | {item['published_at']}</span>
                    <span style="background-color: {badge_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;">{item['label']} ({item['polarity']:+.2f})</span>
                </div>
                <div style="font-size: 0.95rem; font-weight: 500; color: #1E293B;">{item['headline']}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_n2:
        st.markdown("#### Sentiment Aggregation Breakdown")
        sent_pie_df = pd.DataFrame({
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Ratio": [sentiment_summary["pos_ratio"], sentiment_summary["neu_ratio"], sentiment_summary["neg_ratio"]],
            "Color": ["#22C55E", "#EAB308", "#EF4444"]
        })
        fig_sent_pie = go.Figure(go.Pie(
            labels=sent_pie_df["Sentiment"],
            values=sent_pie_df["Ratio"],
            marker=dict(colors=sent_pie_df["Color"]),
            hole=0.4
        ))
        fig_sent_pie.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_sent_pie, use_container_width=True)
        
        st.metric("Aggregated Polarity Score", f"{sentiment_summary['avg_polarity']:+.3f}", delta=sentiment_summary['sentiment_label'])


# =============================================================
# TAB 5: Model Evaluation & Comparison
# =============================================================
with tabs[4]:
    st.subheader("🏆 Multi-Model Benchmarking & Explainable AI (SHAP)")
    
    # 1. Comparison Table
    st.markdown("#### 1. Out-of-Sample Performance Comparison (Strict Chronological Test Split)")
    comp_rows = []
    for m_name, m_data in train_res["results"].items():
        met = m_data["metrics"]
        comp_rows.append({
            "Model": m_name,
            "Accuracy": f"{met['accuracy']*100:.2f}%",
            "Precision (Macro)": f"{met['precision_macro']*100:.2f}%",
            "Recall (Macro)": f"{met['recall_macro']*100:.2f}%",
            "F1-Score (Weighted)": f"{met['f1_weighted']*100:.2f}%",
            "ROC-AUC": f"{met['roc_auc']:.3f}",
            "Status": "⭐ Selected Best" if m_name == best_model_name else "Candidate"
        })
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)
    
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        # Confusion Matrix Heatmap
        st.markdown(f"#### 2. Confusion Matrix ({best_model_name})")
        cm = train_res["results"][best_model_name]["metrics"]["confusion_matrix"]
        labels = ["SELL (0)", "HOLD (1)", "BUY (2)"]
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 14}
        ))
        fig_cm.update_layout(
            xaxis_title="Predicted Direction",
            yaxis_title="True Direction",
            height=320,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_e2:
        # Global Feature Importance / SHAP
        st.markdown("#### 3. Global Feature Importance (Top Predictors)")
        df_imp = explainer.get_global_feature_importance(X_test)
        top_10 = df_imp.head(10).sort_values("Importance_Pct", ascending=True)
        
        fig_imp = go.Figure(go.Bar(
            x=top_10["Importance_Pct"],
            y=top_10["Feature"],
            orientation="h",
            marker_color="#3B82F6",
            text=[f"{v:.1f}%" for v in top_10["Importance_Pct"]],
            textposition="auto"
        ))
        fig_imp.update_layout(
            xaxis_title="Contribution (%)",
            height=320,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_imp, use_container_width=True)


# =============================================================
# TAB 6: Historical Strategy Backtesting
# =============================================================
with tabs[5]:
    st.subheader("📈 Historical Strategy Backtesting vs. Buy & Hold Benchmark")
    st.caption("Evaluated on the unseen out-of-sample Test period with 0.1% transaction cost and slippage per trade.")
    
    # Run backtest on best model test predictions
    y_test_preds = best_model.predict(X_test)
    bt_res = run_backtest(test_df, y_test_preds)
    
    col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns(5)
    col_b1.metric("Strategy Return", f"{bt_res['strategy_total_return_pct']:+.2f}%", delta=f"CAGR: {bt_res['strategy_cagr_pct']:+.1f}%")
    col_b2.metric("Buy & Hold Return", f"{bt_res['buyhold_total_return_pct']:+.2f}%", delta=f"CAGR: {bt_res['buyhold_cagr_pct']:+.1f}%")
    col_b3.metric("Max Drawdown", f"{bt_res['strategy_max_drawdown_pct']:.2f}%", delta=f"B&H: {bt_res['buyhold_max_drawdown_pct']:.2f}%", delta_color="inverse")
    col_b4.metric("Sharpe Ratio", f"{bt_res['strategy_sharpe_ratio']:.2f}", delta=f"B&H: {bt_res['buyhold_sharpe_ratio']:.2f}")
    col_b5.metric("Trades / Win Rate", f"{bt_res['total_trades']} Trades", delta=f"{bt_res['win_rate_pct']:.1f}% Win")
    
    # Equity Curve Plotly
    eq_df = bt_res["equity_curve_df"]
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(
        x=eq_df["Date"], y=eq_df["Strategy_Equity"],
        line=dict(color="#2563EB", width=2.2),
        name="AI ML Strategy"
    ))
    fig_bt.add_trace(go.Scatter(
        x=eq_df["Date"], y=eq_df["BuyHold_Equity"],
        line=dict(color="#9CA3AF", dash="dash", width=1.8),
        name="Buy & Hold Benchmark"
    ))
    
    fig_bt.update_layout(
        title=f"Portfolio Equity Curve (Initial Capital: ₹{config.INITIAL_CAPITAL:,.0f}) | {bt_res['test_period_start']} to {bt_res['test_period_end']}",
        yaxis_title="Portfolio Value (INR)",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_bt, use_container_width=True)


# =============================================================
# TAB 7: Academic Project & Viva Documentation
# =============================================================
with tabs[6]:
    st.subheader("📚 Academic Project Report & Viva Defense Guide")
    st.markdown("""
    ### 1. Project Title
    **“AI-Based Stock Market Prediction and Analysis Using Technical Indicators, Market Regime Detection, and News Sentiment Analysis”**

    ### 2. Abstract
    Financial market direction forecasting remains a core challenge in quantitative finance. This project proposes an end-to-end Machine Learning and Natural Language Processing framework for predicting the next-day price movement direction (BUY, HOLD, SELL) of major Indian equities listed on the National Stock Exchange (NSE). The framework integrates multi-period technical indicators, Gaussian Mixture Model (GMM) market regime clustering, and NLP sentiment analysis from financial news headlines. Strict chronological time-series splitting prevents lookahead data leakage.

    ### 3. Mathematical Formulations
    1. **Next-Day Classification Target:**
       $$\\text{Return}_{t+1} = \\frac{\\text{Close}_{t+1} - \\text{Close}_t}{\\text{Close}_t}$$
       $$Y_t = \\begin{cases} 2 \\text{ (BUY)} & \\text{if } \\text{Return}_{t+1} > +\\tau \\\\ 0 \\text{ (SELL)} & \\text{if } \\text{Return}_{t+1} < -\\tau \\\\ 1 \\text{ (HOLD)} & \\text{otherwise} \\end{cases}$$
       where $\\tau = 0.5\\%$ is the daily noise buffer.

    2. **Relative Strength Index (RSI):**
       $$\\text{RSI} = 100 - \\left( \\frac{100}{1 + \\frac{\\text{EMA}(\\text{Gain}, 14)}{\\text{EMA}(\\text{Loss}, 14)}} \\right)$$

    3. **Market Regime Detection (Gaussian Mixture Model):**
       $$p(x) = \\sum_{k=1}^{K} \\pi_k \\mathcal{N}(x | \\mu_k, \\Sigma_k)$$
       where $x = [\\text{Rolling Return}_{20}, \\text{Rolling Volatility}_{20}]^T$.

    ### 4. Data Leakage Prevention Guarantees
    * **No Random Shuffling:** Data split strictly chronologically ($70\\%$ Train $\\rightarrow 15\\%$ Validation $\\rightarrow 15\\%$ Test).
    * **Scaler Isolation:** Feature scalers (`RobustScaler`) fitted solely on training data $X_{\\text{train}}$.
    * **Feature Alignment:** Features at day $t$ use data $\\le t$ to predict target at $t+1$.

    ### 5. Viva Defense Q&A Cheatsheet
    * **Q1: Why formulate this as classification instead of regression (price prediction)?**
      * *Answer:* Stock prices follow non-stationary stochastic processes (martingales/random walk). Predicting exact rupee values produces misleading $R^2$ scores driven by price levels rather than trend. Classifying actionable direction (BUY/HOLD/SELL) aligns directly with algorithmic execution and risk management.
    * **Q2: Why is GMM better than simple thresholding for market regimes?**
      * *Answer:* GMM is a probabilistic clustering model that accounts for covariance between return drift and volatility, capturing latent market transitions (e.g. low-vol bull vs high-vol bear) without rigid hardcoded cutoffs.
    * **Q3: How does news sentiment get incorporated into the ML pipeline?**
      * *Answer:* Financial headlines are parsed using transformer-based FinBERT/VADER, extracting normalized polarity scores $[-1.0, 1.0]$ which serve as quantitative feature inputs alongside technical indicators.
    """)
