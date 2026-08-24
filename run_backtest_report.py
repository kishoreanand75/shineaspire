# ================================================================================
# ANTONY QUANT AI TERMINAL - RUN THIS TO GENERATE backtest_report.json
# ================================================================================
# WHY THIS FILE EXISTS:
# backtester.py had working metrics code but was never actually run against
# real historical data anywhere in this project — so there was zero evidence
# behind any win-rate/drawdown claim. This script runs it for real and saves
# the result to backtest_report.json, which broker_integrator.py's
# check_live_trading_readiness() reads before it will unlock REAL execution mode.
#
# HOW TO RUN (locally, NOT inside a network-restricted sandbox):
#   python run_backtest_report.py
#
# Requires internet access to Yahoo Finance (query1/query2.finance.yahoo.com).
# If you're behind a firewall/proxy that blocks Yahoo Finance, this will fail
# with a clear error — it will NOT silently write fake numbers.
# ================================================================================

import json
import sys
import argparse
import backtester
import config
import data_feed


def main():
    parser = argparse.ArgumentParser(description="Run a BTC timeframe backtest with walk-forward validation.")
    parser.add_argument("--timeframe", choices=config.SUPPORTED_TIMEFRAMES, default=config.TRADE_TIMEFRAME)
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()

    validation_days = max(1, args.days)
    config.TRADE_TIMEFRAME = args.timeframe
    config.TIMEFRAME = args.timeframe
    print(f"Fetching {config.DEFAULT_SYMBOL} historical data ({args.timeframe}, {validation_days} days)...")
    df = data_feed.fetch_btc_historical_data(config.DEFAULT_SYMBOL, config.TRADE_TIMEFRAME, days=validation_days)

    if df.empty:
        print("ERROR: Could not fetch Bitcoin data. Check Binance connectivity.")
        sys.exit(1)

    df = df.rename(columns={column: column.title() for column in df.columns})
    required_columns = {"Open", "High", "Low", "Close", "Volume"}
    if not required_columns.issubset(df.columns):
        missing = ", ".join(sorted(required_columns - set(df.columns)))
        print(f"ERROR: Binance returned incomplete candle data; missing: {missing}")
        sys.exit(1)

    # backtester.run_institutional_backtest_with_slippage expects capitalized
    # OHLC column names and an 'ATR' column for target/stop sizing.
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df.dropna(inplace=True)

    print(f"Got {len(df)} bars. Running one-week backtest with realistic slippage + Binance taker fees...")
    result = backtester.run_institutional_backtest_with_slippage(
        df, initial_capital=100000.0, symbol=config.DEFAULT_SYMBOL,
        risk_per_trade_pct=config.RISK_PER_TRADE_PCT,
    )

    if result is None:
        print("ERROR: Not enough data bars to run a meaningful backtest (need 30+).")
        sys.exit(1)

    print("\n=== BACKTEST RESULTS (single period) ===")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}")
    print(f"  signals_in_week: {result['metrics'].get('total_trades', 0)}")

    print("\nRunning walk-forward validation (5 folds) to check for overfitting...")
    wf = backtester.walk_forward_validate(
        df, backtester.run_institutional_backtest_with_slippage, n_splits=5,
        initial_capital=100000.0, symbol=config.DEFAULT_SYMBOL,
        risk_per_trade_pct=config.RISK_PER_TRADE_PCT,
    )
    print("\n=== WALK-FORWARD SUMMARY ===")
    print(json.dumps(wf.get("summary", {}), indent=2))

    report = {
        "generated_from": config.DEFAULT_SYMBOL,
        "timeframe": args.timeframe,
        "validation_days": validation_days,
        "bars_used": len(df),
        "signals_in_week": result["metrics"].get("total_trades", 0),
        "metrics": result["metrics"],
        "walk_forward_summary": wf.get("summary", {}),
        "walk_forward_folds": wf.get("folds", []),
    }

    with open("backtest_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n✅ Saved backtest_report.json")
    print(
        f"Live-trading gate needs: win_rate >= {config.MIN_BACKTEST_WIN_RATE}%, "
        f"max_drawdown_pct <= {config.MAX_BACKTEST_DRAWDOWN_PCT}%"
    )


if __name__ == "__main__":
    main()