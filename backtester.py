# backtester.py
import os
import pandas as pd
import numpy as np
import ta
import config
from xgboost import XGBClassifier
from signal_engine import decide_from_row, build_features
import data_feed

# Load the SAME model file the live bot loads, so backtest and live use
# identical signal logic (see signal_engine.py header for why this matters).
_MODEL_FILE = "xgboost_model.json"
_backtest_model = None
if os.path.exists(_MODEL_FILE):
    _backtest_model = XGBClassifier()
    _backtest_model.load_model(_MODEL_FILE)

def calculate_backtest_metrics(trades_df: pd.DataFrame, equity_curve: list, initial_capital: float, periods_per_year: int = 252 * 75):
    """
    Computes standard backtest performance metrics from a trades dataframe + equity curve.
    periods_per_year: approx number of 5-min bars in a trading year (used for Sharpe annualization).
    Adjust this if your bar interval or trading calendar differs.
    """
    if trades_df is None or len(trades_df) == 0:
        return {
            "total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "max_drawdown_pct": 0.0, "max_drawdown_amount": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0, "final_capital": initial_capital,
            "total_return_pct": 0.0
        }

    pnl_col = "Net_PnL" if "Net_PnL" in trades_df.columns else "PnL"
    pnls = trades_df[pnl_col].astype(float)

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    total_trades = len(pnls)
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0

    # Sharpe ratio: based on per-trade returns (as % of capital at the time), annualized.
    equity_arr = np.array(equity_curve, dtype=float)
    if len(equity_arr) > 1:
        valid_denominators = equity_arr[:-1] > 0
        returns = np.diff(equity_arr)[valid_denominators] / equity_arr[:-1][valid_denominators]
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)
        else:
            sharpe = 0.0
    else:
        returns = np.array([])
        sharpe = 0.0

    # Sortino ratio: like Sharpe, but only penalizes DOWNSIDE volatility (returns < 0).
    # A strategy with big upside swings and small, controlled downside swings should
    # score better here than on Sharpe alone -- which is exactly the risk profile we want.
    if len(returns) > 0:
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0
        if downside_std > 0:
            sortino = (returns.mean() / downside_std) * np.sqrt(periods_per_year)
        elif returns.mean() > 0:
            sortino = float('inf')  # no downside volatility at all and net-positive: undefined upper bound
        else:
            sortino = 0.0
    else:
        sortino = 0.0

    # Max drawdown from equity curve
    running_max = np.maximum.accumulate(equity_arr) if len(equity_arr) > 0 else np.array([initial_capital])
    drawdowns = (equity_arr - running_max)
    max_dd_amount = drawdowns.min() if len(drawdowns) > 0 else 0.0
    max_dd_pct = (max_dd_amount / running_max[np.argmin(drawdowns)] * 100) if len(drawdowns) > 0 and running_max[np.argmin(drawdowns)] != 0 else 0.0

    final_capital = equity_arr[-1] if len(equity_arr) > 0 else initial_capital
    total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100

    return {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "inf (no losses)",
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2) if sortino != float('inf') else "inf (no downside volatility)",
        "max_drawdown_pct": round(max_dd_pct, 2),
        "max_drawdown_amount": round(max_dd_amount, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "final_capital": round(final_capital, 2),
        "total_return_pct": round(total_return_pct, 2)
    }


def walk_forward_validate(df: pd.DataFrame, backtest_fn, n_splits: int = 5, initial_capital: float = 100000.0, **backtest_kwargs):
    """
    Splits df into n_splits sequential chunks. Runs backtest_fn on each chunk independently
    (out-of-sample style: each fold is a fresh, non-overlapping period), so a strategy that
    only works in one lucky period shows up as inconsistent across folds instead of hiding
    inside one aggregate number.

    backtest_fn must accept (df_slice, initial_capital=..., **kwargs) and return a dict with
    "trades" (DataFrame) and "equity" (list), matching run_institutional_backtest_with_slippage.
    """
    if df is None or len(df) < n_splits * 30:
        return {"error": f"Not enough data for {n_splits} folds (need at least {n_splits * 30} bars, got {len(df) if df is not None else 0})"}

    fold_size = len(df) // n_splits
    fold_results = []

    for fold_idx in range(n_splits):
        start = fold_idx * fold_size
        end = start + fold_size if fold_idx < n_splits - 1 else len(df)
        df_fold = df.iloc[start:end].copy()

        result = backtest_fn(df_fold, initial_capital=initial_capital, **backtest_kwargs)
        if result is None:
            fold_results.append({"fold": fold_idx + 1, "error": "Insufficient data in this fold"})
            continue

        metrics = calculate_backtest_metrics(result.get("trades"), result.get("equity", [initial_capital]), initial_capital)
        metrics["fold"] = fold_idx + 1
        metrics["period_start"] = str(df_fold.index[0]) if len(df_fold) > 0 else None
        metrics["period_end"] = str(df_fold.index[-1]) if len(df_fold) > 0 else None
        fold_results.append(metrics)

    # Consistency check: how much do win-rate / Sharpe swing across folds?
    valid_folds = [f for f in fold_results if "error" not in f]
    if valid_folds:
        win_rates = [f["win_rate"] for f in valid_folds]
        sharpes = [f["sharpe_ratio"] for f in valid_folds]
        summary = {
            "num_valid_folds": len(valid_folds),
            "win_rate_mean": round(np.mean(win_rates), 2),
            "win_rate_std": round(np.std(win_rates), 2),
            "sharpe_mean": round(np.mean(sharpes), 2),
            "sharpe_std": round(np.std(sharpes), 2),
            "consistency_warning": "HIGH VARIANCE across folds - possible overfitting" if np.std(win_rates) > 20 else "Reasonably consistent"
        }
    else:
        summary = {"error": "No valid folds produced results"}

    return {"folds": fold_results, "summary": summary}

def run_historical_backtest(df, initial_capital=100000, target_pct=0.06, sl_pct=0.03, friction=45.0):
    if df is None or len(df) < 30:
        return None
    
    df = df.copy()
    # Body Range Ratio
    df['Body_Ratio'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-6)
    
    trades = []
    capital = initial_capital
    equity = [capital]
    
    for i in range(20, len(df) - 1):
        row = df.iloc[i]
        # Core Entry Rules Check
        # BUG FIX: Body_Ratio uses abs(), so it was firing on strong bearish
        # (red) candles too, buying right after a downward move about half
        # the time. A long-only strategy must require a bullish candle body.
        if row['Body_Ratio'] >= 0.60 and row['Close'] > row['Open']:
            entry = df.iloc[i+1]['Open']
            tp = entry * (1 + target_pct)
            sl = entry * (1 - sl_pct)
            
            next_candle = df.iloc[i+1]
            if next_candle['High'] >= tp:
                exit_p = tp
                res = "TARGET (+6%)"
            elif next_candle['Low'] <= sl:
                exit_p = sl
                res = "STOP LOSS (-3%)"
            else:
                exit_p = next_candle['Close']
                res = "5-MIN TIMEOUT"
                
            pnl = ((exit_p - entry) / entry) * capital - friction
            capital += pnl
            equity.append(capital)
            
            trades.append({"Time": df.index[i+1], "Type": "BUY", "Entry": entry, "Exit": exit_p, "PnL": pnl, "Result": res})
            
    tdf = pd.DataFrame(trades)
    wins = len(tdf[tdf['PnL'] > 0]) if len(tdf) > 0 else 0
    total = len(tdf) if len(tdf) > 0 else 1
    win_rate = (wins / total) * 100
    
    return {
        "trades": tdf,
        "equity": equity,
        "win_rate": round(win_rate, 2),
        "total_trades": total,
        "final_capital": round(capital, 2),
        "total_profit": round(capital - initial_capital, 2)
    }

def compute_htf_trend_series(df: pd.DataFrame, htf: str = "1h", ema_window: int = 50) -> pd.Series:
    """
    Point-in-time higher-timeframe trend for backtesting.

    LEAK THIS REPLACES: run_institutional_backtest_with_slippage() used to call
    data_feed.fetch_htf_trend(symbol, htf="1h") ONCE before the backtest loop
    started, then reused that single value for every historical bar tested.
    fetch_htf_trend() always hits the LIVE Binance API, so every bar in a
    multi-day backtest was being filtered using TODAY's 1h trend, not the 1h
    trend as it actually existed at that point in history -- a real
    look-ahead leak that inflates backtest performance.

    Fix: resample the SAME historical df already being backtested up to
    `htf`, compute EMA(ema_window) on those HTF closes, label each HTF bar
    BULLISH/BEARISH/NEUTRAL, then shift by one HTF bar so a trade-timeframe
    bar can only ever see the most recently *closed* HTF candle -- never the
    one still forming at that moment. No network call, no future data.
    """
    resample_rule = {"1m": "1min", "5m": "5min", "15m": "15min",
                      "1h": "1h", "4h": "4h", "1d": "1D"}.get(htf, "1h")

    htf_df = df[['Open', 'High', 'Low', 'Close']].resample(resample_rule).agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()

    if len(htf_df) < ema_window + 1:
        return pd.Series("UNKNOWN", index=df.index)

    ema = ta.trend.ema_indicator(htf_df['Close'], window=ema_window)

    htf_trend = pd.Series("UNKNOWN", index=htf_df.index)
    valid = ema.notna()
    htf_trend.loc[valid & (htf_df['Close'] > ema)] = "BULLISH"
    htf_trend.loc[valid & (htf_df['Close'] < ema)] = "BEARISH"
    htf_trend.loc[valid & (htf_df['Close'] == ema)] = "NEUTRAL"

    # Shift by one HTF bar: the trend label for an HTF candle isn't knowable
    # until that candle has fully closed.
    htf_trend = htf_trend.shift(1).fillna("UNKNOWN")

    # Point-in-time join: for every trade-timeframe timestamp, look back to
    # the most recent already-closed HTF bar. merge_asof(direction='backward')
    # guarantees no future HTF bar is ever visible to an earlier trade bar.
    left = pd.DataFrame({"ts": df.index}).sort_values("ts")
    right = pd.DataFrame({"ts": htf_trend.index, "htf_trend": htf_trend.values}).sort_values("ts")
    merged = pd.merge_asof(left, right, on="ts", direction="backward")
    merged["htf_trend"] = merged["htf_trend"].fillna("UNKNOWN")

    result = pd.Series(merged["htf_trend"].values, index=df.index)
    return result


def run_institutional_backtest_with_slippage(df: pd.DataFrame, initial_capital: float = 100000.0,
                                              slippage_pct: float = 0.0008, taker_fee_pct: float = 0.00075,
                                              max_hold_bars: int = 20, tp_atr_mult: float = None, sl_atr_mult: float = None,
                                              min_lookback: int = 45, symbol: str = "BTCUSDT",
                                              risk_per_trade_pct: float = 0.01):
    """
    Vectorized Backtest Engine with Realistic Slippage (0.08%) & Spread Cost.

    FIX (previous): The original version only looked at the SINGLE next candle
    to decide TP/SL/exit while asking for a 3x-ATR move -- fixed by holding up
    to max_hold_bars and using a realistic reward:risk ratio.

    FIX (previous): This backtester used its own standalone entry rule which
    was NOT the same logic the live bot actually trades -- now calls the SAME
    evaluate_smart_breakout_signals function the live bot uses, so the
    reported win rate reflects the strategy that would actually be deployed.

    FIX (this pass, 3 changes at user's request):
      1. evaluate_smart_breakout_signals now skips the 11:00-13:00 lunch-chop
         window (was computed elsewhere in the codebase but never enforced).
      2. tp_atr_mult/sl_atr_mult changed from 1.5:1 to 1:1 -- the 1.5:1
         target was rarely reached before the 1x stop, given how weak the
         entry signal actually is; a 1:1 target is more reachable, trading
         win-rate for a smaller required edge per trade.
      3. max_hold_bars raised from 10 to 20 -- trades were timing out into a
         mediocre/negative close before having room to reach target.
    FIX (this pass, crypto fee model): friction now uses Binance's standard
    spot taker fee (0.075% / 0.00075) charged on BOTH the entry and exit
    notional -- replacing the old equity-market fixed brokerage (flat ₹40) +
    STT/GST (0.15%) model, which doesn't apply to a crypto exchange at all.
    None of these are guaranteed to produce a profitable strategy -- they are
    testable adjustments, not a fix with a known outcome.
    """
    if df is None or len(df) < 30:
        return None

    tp_atr_mult = config.ATR_TP_MULTIPLIER if tp_atr_mult is None else tp_atr_mult
    sl_atr_mult = config.ATR_SL_MULTIPLIER if sl_atr_mult is None else sl_atr_mult

    df = df.copy()
    df_feat = build_features(df)  # computed ONCE for the whole dataset -- O(n), not O(n^2)

    # Point-in-time HTF trend per bar (see compute_htf_trend_series docstring
    # for why the old single live-fetch-and-reuse approach was a leak).
    htf_trend_series = compute_htf_trend_series(df, htf="1h")

    trades = []
    capital = initial_capital
    equity_curve = [capital]

    i = min_lookback
    n = len(df)
    while i < n - 1:
        if capital <= 0:
            break

        sig_result = decide_from_row(df_feat, i, asset_symbol=symbol, model=_backtest_model, htf_trend=htf_trend_series.iloc[i])
        signal = sig_result.get("signal", "HOLD")

        if signal in ("BUY_CALL", "BUY_PUT"):
            row = df.iloc[i]
            direction = 1 if signal == "BUY_CALL" else -1
            raw_entry = df.iloc[i + 1]['Open']

            # Buy-side slippage always works against entry (pay more to buy
            # a call side, or get a worse fill entering a bearish/put side).
            entry_price = raw_entry * (1 + slippage_pct) if direction == 1 else raw_entry * (1 - slippage_pct)

            atr_val = df_feat.iloc[i].get('ATR', raw_entry * 0.01)
            if direction == 1:
                tp = entry_price + (atr_val * tp_atr_mult)
                sl = entry_price - (atr_val * sl_atr_mult)
            else:
                tp = entry_price - (atr_val * tp_atr_mult)
                sl = entry_price + (atr_val * sl_atr_mult)

            risk_per_unit = abs(entry_price - sl)
            if risk_per_unit <= 0 or not np.isfinite(risk_per_unit):
                i += 1
                continue

            # Match the live broker: size from the stop distance instead of
            # treating the whole account as the trade's notional value.
            risk_amount = capital * risk_per_trade_pct
            quantity = risk_amount / risk_per_unit

            raw_exit = None
            exit_reason = None
            bars_held = None
            exit_bar_idx = min(i + max_hold_bars, n - 1)
            for j in range(i + 1, exit_bar_idx + 1):
                bar = df.iloc[j]
                if direction == 1:
                    hit_tp = bar['High'] >= tp
                    hit_sl = bar['Low'] <= sl
                else:
                    hit_tp = bar['Low'] <= tp
                    hit_sl = bar['High'] >= sl
                # Conservative assumption: if both TP and SL fall inside the
                # same candle's range, assume SL hit first (worst case), so
                # results aren't optimistically biased.
                if hit_sl:
                    raw_exit = sl
                    exit_reason = "SL"
                    bars_held = j - i
                    i = j
                    break
                elif hit_tp:
                    raw_exit = tp
                    exit_reason = "TP"
                    bars_held = j - i
                    i = j
                    break

            if raw_exit is None:
                raw_exit = df.iloc[exit_bar_idx]['Close']
                exit_reason = "TIMEOUT"
                bars_held = exit_bar_idx - i
                i = exit_bar_idx

            exit_price = raw_exit * (1 - slippage_pct) if direction == 1 else raw_exit * (1 + slippage_pct)

            gross_pnl = (exit_price - entry_price) * quantity * direction
            # Binance spot taker fee (0.075%) applied to both the entry and
            # exit notional value -- replaces the equity-market flat brokerage
            # + STT/GST friction model, which does not apply to crypto.
            entry_notional = entry_price * quantity
            exit_notional = exit_price * quantity
            taker_fees = (entry_notional + exit_notional) * taker_fee_pct
            total_friction = taker_fees
            net_pnl = gross_pnl - total_friction

            capital = max(0.0, capital + net_pnl)
            equity_curve.append(capital)

            trades.append({
                "Direction": "CALL" if direction == 1 else "PUT",
                "Entry": round(entry_price, 2),
                "Exit": round(exit_price, 2),
                "Quantity": round(quantity, 6),
                "Gross_PnL": round(gross_pnl, 2),
                "Friction_Cost": round(total_friction, 2),
                "Exit_Reason": exit_reason,
                "Bars_Held": bars_held,
                "Net_PnL": round(net_pnl, 2)
            })
            i += 1
        else:
            i += 1

    tdf = pd.DataFrame(trades)
    metrics = calculate_backtest_metrics(tdf, equity_curve, initial_capital)
    return {
        "trades": tdf,
        "equity": equity_curve,
        "final_capital": round(capital, 2),
        "metrics": metrics
    }