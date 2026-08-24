# regime_detector.py - Phase 3: Volatility Regime Detection & Multi-Asset Correlation
import pandas as pd
import numpy as np
import ta


def classify_volatility_regime(df: pd.DataFrame, atr_window: int = 14, lookback: int = 100) -> dict:
    """
    Classifies current market regime as TRENDING, SIDEWAYS, or HIGH_VOLATILITY_CHOP
    using ATR percentile + ADX, so a strategy can adapt (or stand down) per regime
    instead of using one fixed rule set for every market condition.

    Returns dict with regime label, ATR percentile rank, ADX value, and a
    recommended action (TRADE_NORMAL / TRADE_REDUCED_SIZE / STAND_ASIDE).
    """
    if df is None or len(df) < max(atr_window, 20) + 5:
        return {"regime": "UNKNOWN", "reason": "Insufficient data", "action": "STAND_ASIDE"}

    df = df.copy()
    atr = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=atr_window)
    adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()

    recent_atr = atr.tail(lookback)
    current_atr = atr.iloc[-1]
    current_adx = adx.iloc[-1]

    # Percentile rank of current ATR vs its own recent history -> relative, not absolute,
    # volatility measure (works across assets with very different price scales).
    atr_percentile = (recent_atr < current_atr).mean() * 100 if len(recent_atr) > 0 else 50.0

    if current_adx >= 25 and atr_percentile <= 80:
        regime = "TRENDING"
        action = "TRADE_NORMAL"
        reason = f"Strong trend (ADX {current_adx:.1f}) with controlled volatility (ATR pct {atr_percentile:.0f})"
    elif current_adx < 20 and atr_percentile <= 50:
        regime = "SIDEWAYS"
        action = "STAND_ASIDE"
        reason = f"Weak trend (ADX {current_adx:.1f}), low volatility - chop risk, breakout strategies underperform here"
    elif atr_percentile >= 85:
        regime = "HIGH_VOLATILITY_CHOP"
        action = "TRADE_REDUCED_SIZE"
        reason = f"Volatility spike (ATR pct {atr_percentile:.0f}) - wider stops needed, whipsaw risk elevated"
    else:
        regime = "TRANSITIONAL"
        action = "TRADE_REDUCED_SIZE"
        reason = f"Mixed signals (ADX {current_adx:.1f}, ATR pct {atr_percentile:.0f}) - reduce conviction"

    return {
        "regime": regime,
        "action": action,
        "reason": reason,
        "adx": round(float(current_adx), 2),
        "atr": round(float(current_atr), 2),
        "atr_percentile": round(float(atr_percentile), 1)
    }


def compute_volume_features(df: pd.DataFrame, window: int = 20) -> dict:
    """
    Volume-based features beyond the simple vol_ratio already used in multi_strategy.py:
    - relative volume percentile (how unusual is current volume vs recent history)
    - volume trend (is volume expanding or contracting over last few bars)
    - price-volume divergence (price making new highs/lows without matching volume = weak move)
    """
    if df is None or len(df) < window + 5:
        return {"error": "Insufficient data"}

    df = df.copy()
    vol = df['Volume']
    vol_ma = vol.rolling(window).mean()
    current_vol = vol.iloc[-1]
    current_vol_ma = vol_ma.iloc[-1]

    recent_vol = vol.tail(window * 2)
    vol_percentile = (recent_vol < current_vol).mean() * 100 if len(recent_vol) > 0 else 50.0

    # Volume trend: compare last 5-bar avg vs prior 5-bar avg
    last5_avg = vol.tail(5).mean()
    prior5_avg = vol.iloc[-10:-5].mean() if len(vol) >= 10 else last5_avg
    vol_trend = "EXPANDING" if last5_avg > prior5_avg * 1.1 else ("CONTRACTING" if last5_avg < prior5_avg * 0.9 else "STABLE")

    # Simple price-volume divergence check on last 5 bars
    price_change_5 = (df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5] if len(df) >= 5 else 0.0
    divergence = "NONE"
    if abs(price_change_5) > 0.005 and vol_trend == "CONTRACTING":
        divergence = "WEAK_MOVE_LOW_VOLUME"  # price moved but volume didn't confirm -> lower conviction

    return {
        "current_volume_ratio": round(float(current_vol / current_vol_ma), 2) if current_vol_ma > 0 else 0.0,
        "volume_percentile": round(float(vol_percentile), 1),
        "volume_trend": vol_trend,
        "divergence_flag": divergence
    }


def build_correlation_matrix(price_data: dict, window: int = 50) -> pd.DataFrame:
    """
    Builds a correlation matrix across multiple assets' returns, so the bot can avoid
    stacking several "different" positions that are actually all the same directional bet
    (e.g. BTC + ETH + SOL are usually highly correlated - holding all three isn't real
    diversification, it's the same trade sized up 3x).

    price_data: dict of {asset_name: pd.DataFrame with 'Close' column}
    Returns a correlation matrix DataFrame of recent returns.
    """
    returns_dict = {}
    for name, df in price_data.items():
        if df is None or len(df) < window:
            continue
        returns_dict[name] = df['Close'].pct_change().tail(window).reset_index(drop=True)

    if len(returns_dict) < 2:
        return pd.DataFrame()

    returns_df = pd.DataFrame(returns_dict)
    return returns_df.corr()


def check_correlation_before_entry(open_positions: dict, candidate_symbol: str, price_data: dict,
                                     correlation_threshold: float = 0.75, window: int = 50) -> dict:
    """
    Before opening a new position, checks its correlation against currently open positions.
    If highly correlated with an existing position, flags it (still allows the trade, but
    the caller can choose to reduce size or skip - real diversification benefit is limited
    when correlation is this high).
    """
    if not open_positions or candidate_symbol not in price_data:
        return {"is_correlated": False, "max_correlation": 0.0, "correlated_with": None}

    candidate_returns = price_data[candidate_symbol]['Close'].pct_change().tail(window)

    max_corr = 0.0
    correlated_with = None
    for open_symbol in open_positions.keys():
        base_name = open_symbol.split("_")[0]  # strip _OPT_CALL/_PUT suffix
        if base_name not in price_data or base_name == candidate_symbol:
            continue
        open_returns = price_data[base_name]['Close'].pct_change().tail(window)
        if len(candidate_returns) < 10 or len(open_returns) < 10:
            continue
        aligned_len = min(len(candidate_returns), len(open_returns))
        corr = candidate_returns.tail(aligned_len).reset_index(drop=True).corr(
            open_returns.tail(aligned_len).reset_index(drop=True)
        )
        if abs(corr) > abs(max_corr):
            max_corr = corr
            correlated_with = base_name

    return {
        "is_correlated": abs(max_corr) >= correlation_threshold,
        "max_correlation": round(float(max_corr), 2) if not np.isnan(max_corr) else 0.0,
        "correlated_with": correlated_with
    }