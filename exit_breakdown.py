# exit_breakdown.py
# Run this to see WHERE the P&L is actually leaking: TP hits, SL hits, or
# timeout exits. This tells us whether the fix belongs in signal_engine.py
# (exit logic / TP-SL distance) or train_model.py (the model itself).
#
# Usage: python exit_breakdown.py

import config
import data_feed
from backtester import run_institutional_backtest_with_slippage, calculate_backtest_metrics

print(f"Fetching {config.DEFAULT_SYMBOL} historical data for backtest...")
df = data_feed.fetch_btc_historical_data(config.DEFAULT_SYMBOL, config.TRADE_TIMEFRAME, days=7)
df = df.rename(columns={c: c.title() for c in df.columns})
print(f"Got {len(df)} bars.\n")

result = run_institutional_backtest_with_slippage(df, symbol=config.DEFAULT_SYMBOL)
trades = result["trades"]

if trades.empty:
    print("No trades fired in this window.")
else:
    print("=== OVERALL METRICS ===")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}")

    print("\n=== BREAKDOWN BY EXIT REASON ===")
    for reason in ["TP", "SL", "TIMEOUT"]:
        sub = trades[trades["Exit_Reason"] == reason]
        if len(sub) == 0:
            continue
        count = len(sub)
        pct = count / len(trades) * 100
        total_pnl = sub["Net_PnL"].sum()
        avg_pnl = sub["Net_PnL"].mean()
        avg_bars = sub["Bars_Held"].mean()
        print(f"  {reason:8s}: {count:3d} trades ({pct:5.1f}%)  "
              f"total_PnL={total_pnl:10.2f}  avg_PnL={avg_pnl:8.2f}  avg_bars_held={avg_bars:.1f}")

    print("\n=== WHAT THIS TELLS YOU ===")
    timeout_pnl = trades[trades["Exit_Reason"] == "TIMEOUT"]["Net_PnL"].sum()
    sl_pnl = trades[trades["Exit_Reason"] == "SL"]["Net_PnL"].sum()
    if timeout_pnl < 0 and abs(timeout_pnl) > abs(sl_pnl) * 0.3:
        print("  -> TIMEOUT exits are a major loss source. The model's calls are")
        print("     often 'not wrong direction, just not decisive enough' before")
        print("     max_hold_bars runs out. Consider shortening max_hold_bars or")
        print("     tightening tp_atr_mult so targets are reachable faster.")
    total_fee = trades["Friction_Cost"].sum()
    gross = trades["Gross_PnL"].sum()
    if gross != 0:
        print(f"  -> Fees consumed {total_fee:.2f} against gross P&L of {gross:.2f} "
              f"({(total_fee/abs(gross)*100 if gross!=0 else 0):.1f}% of gross magnitude).")
        print("     If this is large, you may be overtrading relative to your edge size.")
        