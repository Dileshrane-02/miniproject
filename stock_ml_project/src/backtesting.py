"""
Historical Backtesting Engine Module.
Simulates ML-guided trading strategy vs. passive Buy-and-Hold benchmark over the unseen test period,
incorporating transaction fees, slippage, equity curves, maximum drawdown, and Sharpe ratio.
"""

import os
import sys
import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

logger = logging.getLogger(__name__)


def run_backtest(
    df_test: pd.DataFrame,
    predictions: np.ndarray,
    initial_capital: float = config.INITIAL_CAPITAL,
    transaction_cost_pct: float = config.TRANSACTION_COST_PCT,
    risk_free_rate: float = 0.065  # 6.5% Indian Repo / T-Bill rate
) -> Dict[str, Any]:
    """
    Executes a realistic long-only/cash trading simulation on the test set.
    
    Trading Rules:
      - Signal = 2 (BUY): Take / maintain Long position (Position = 1.0)
      - Signal = 0 (SELL): Exit to Cash (Position = 0.0)
      - Signal = 1 (HOLD): Maintain previous position
    """
    df = df_test.copy().reset_index(drop=True)
    df["Predicted_Signal"] = predictions
    
    # Generate Target Position: 1.0 (Invested) or 0.0 (Cash)
    positions = []
    current_pos = 0.0
    
    for sig in df["Predicted_Signal"]:
        if sig == 2:    # BUY
            current_pos = 1.0
        elif sig == 0:  # SELL
            current_pos = 0.0
        # If HOLD (1), keep current_pos unchanged
        positions.append(current_pos)
        
    df["Position"] = positions
    
    # Calculate position changes (trades) to deduct transaction fees
    # Trade occurs when position changes: from 0->1 (entry) or 1->0 (exit)
    df["Position_Change"] = df["Position"].diff().abs().fillna(df["Position"].iloc[0])
    df["Trade_Cost"] = df["Position_Change"] * transaction_cost_pct
    
    # Daily stock return
    df["Stock_Return"] = df["Close"].pct_change().fillna(0.0)
    
    # Strategy Return = Position * Stock Return - Transaction Cost
    # Position chosen at close of t-1 applies to return of day t
    df["Strategy_Return"] = (df["Position"].shift(1).fillna(0.0) * df["Stock_Return"]) - df["Trade_Cost"]
    
    # Cumulative Equity Curves
    df["Strategy_Equity"] = initial_capital * (1.0 + df["Strategy_Return"]).cumprod()
    df["BuyHold_Equity"] = initial_capital * (1.0 + df["Stock_Return"]).cumprod()
    
    # Performance Metrics
    total_days = len(df)
    years = max(total_days / 252.0, 0.05)
    
    strat_final = df["Strategy_Equity"].iloc[-1]
    buyhold_final = df["BuyHold_Equity"].iloc[-1]
    
    strat_total_ret = (strat_final - initial_capital) / initial_capital
    buyhold_total_ret = (buyhold_final - initial_capital) / initial_capital
    
    strat_cagr = (strat_final / initial_capital) ** (1.0 / years) - 1.0 if strat_final > 0 else -1.0
    buyhold_cagr = (buyhold_final / initial_capital) ** (1.0 / years) - 1.0 if buyhold_final > 0 else -1.0
    
    # Maximum Drawdown
    strat_peak = df["Strategy_Equity"].cummax()
    strat_dd = (df["Strategy_Equity"] - strat_peak) / strat_peak
    strat_max_dd = float(strat_dd.min())
    
    buyhold_peak = df["BuyHold_Equity"].cummax()
    buyhold_dd = (df["BuyHold_Equity"] - buyhold_peak) / buyhold_peak
    buyhold_max_dd = float(buyhold_dd.min())
    
    # Sharpe Ratio
    daily_rf = (1.0 + risk_free_rate) ** (1/252) - 1.0
    strat_excess_ret = df["Strategy_Return"] - daily_rf
    strat_vol = df["Strategy_Return"].std() * np.sqrt(252)
    strat_sharpe = float((df["Strategy_Return"].mean() - daily_rf) * 252 / (strat_vol + 1e-9)) if strat_vol > 0 else 0.0
    
    buyhold_vol = df["Stock_Return"].std() * np.sqrt(252)
    buyhold_sharpe = float((df["Stock_Return"].mean() - daily_rf) * 252 / (buyhold_vol + 1e-9)) if buyhold_vol > 0 else 0.0
    
    # Trade statistics
    trade_events = df[df["Position_Change"] > 0]
    num_trades = int(df["Position_Change"].sum())
    
    # Win rate on active holding days
    active_days = df[df["Position"].shift(1) == 1.0]
    win_days = active_days[active_days["Stock_Return"] > 0]
    win_rate = (len(win_days) / len(active_days) * 100.0) if len(active_days) > 0 else 0.0

    return {
        "initial_capital": initial_capital,
        "strategy_final_capital": round(strat_final, 2),
        "buyhold_final_capital": round(buyhold_final, 2),
        "strategy_total_return_pct": round(strat_total_ret * 100.0, 2),
        "buyhold_total_return_pct": round(buyhold_total_ret * 100.0, 2),
        "strategy_cagr_pct": round(strat_cagr * 100.0, 2),
        "buyhold_cagr_pct": round(buyhold_cagr * 100.0, 2),
        "strategy_max_drawdown_pct": round(strat_max_dd * 100.0, 2),
        "buyhold_max_drawdown_pct": round(buyhold_max_dd * 100.0, 2),
        "strategy_sharpe_ratio": round(strat_sharpe, 2),
        "buyhold_sharpe_ratio": round(buyhold_sharpe, 2),
        "total_trades": num_trades,
        "win_rate_pct": round(win_rate, 2),
        "test_period_start": str(df["Date"].iloc[0].date()),
        "test_period_end": str(df["Date"].iloc[-1].date()),
        "equity_curve_df": df[["Date", "Close", "Strategy_Equity", "BuyHold_Equity", "Position", "Predicted_Signal"]]
    }


if __name__ == "__main__":
    from train import train_and_compare_models
    
    train_res = train_and_compare_models("RELIANCE")
    test_df = train_res["test_df"]
    best_model = train_res["best_model"]
    X_test = train_res["X_test"]
    
    y_preds = best_model.predict(X_test)
    bt_res = run_backtest(test_df, y_preds)
    
    print("\n--- Backtest Results Summary ---")
    print(f"Period: {bt_res['test_period_start']} to {bt_res['test_period_end']}")
    print(f"ML Strategy Return: {bt_res['strategy_total_return_pct']}% (CAGR: {bt_res['strategy_cagr_pct']}%)")
    print(f"Buy & Hold Return:  {bt_res['buyhold_total_return_pct']}% (CAGR: {bt_res['buyhold_cagr_pct']}%)")
    print(f"ML Max Drawdown:    {bt_res['strategy_max_drawdown_pct']}% vs B&H: {bt_res['buyhold_max_drawdown_pct']}%")
    print(f"Sharpe Ratio:       ML: {bt_res['strategy_sharpe_ratio']} | B&H: {bt_res['buyhold_sharpe_ratio']}")
    print(f"Total Trades:       {bt_res['total_trades']} (Win Rate: {bt_res['win_rate_pct']}%)")
