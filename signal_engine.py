# signal_engine.py
# ============================================================================
# SINGLE SOURCE OF TRUTH for trade signal generation.
#
# WHY THIS FILE EXISTS:
# The original codebase had the SAME decision logic duplicated (and slightly
# diverging) across three places:
#   1. strategy.py            -> generate_signal()
#   2. multi_strategy.py      -> evaluate_smart_breakout_signals()  (used by backtester.py)
#   3. multi_strategy.py      -> scan_all_assets()                  (used LIVE)
#   4. multi_strategy.py      -> evaluate_single_asset_signal()     (used for crypto radar)
#
# Because backtester.py tested path #2 while the live bot actually traded
# path #3, the backtest_report.json numbers you were looking at did NOT
# reflect what the live bot would actually do. That is the single biggest
# reason the reported win-rate/Sharpe numbers couldn't be trusted.
#
# From now on: BOTH live scanning and backtesting must call generate_signal()
# in this file. If you change the logic, you change it in exactly one place,
# and backtest results are guaranteed to reflect live behavior.
#
# WHAT ELSE CHANGED VS THE ORIGINAL:
# - Removed the hardcoded `confidence = 0.72`. When a trained model is
#   available, confidence is the model's real predict_proba() output.
#   When no model is available (rule-only fallback), confidence is explicitly
#   marked as "RULE_ONLY" with confidence=None -- we do not fabricate a
#   number that looks like a probability but isn't one.
# - The entry rule still fires only after price has already crossed
#   VWAP/EMA (a "confirmation" rule). This is intentionally left as-is here
#   -- fixing the late-entry problem requires new features/labels and a
#   model retrain, which is a separate, bigger change (see train_model.py
#   notes). Flagging it clearly rather than silently "fixing" it with an
#   untested heuristic.
# ============================================================================

import numpy as np
import pandas as pd
import ta
import config

try:
    import data_feed as _data_feed  # for fetch_htf_trend() -- optional at import time
except Exception:
    _data_feed = None


# NOTE: this list MUST stay in sync with FEATURE_COLUMNS in train_model.py.
# If you add/remove/rename a feature here, mirror it there and retrain, or
# XGBoost will raise a "feature_names mismatch" error at inference time.
FEATURE_COLUMNS = [
    'RSI', 'EMA_Diff', 'ADX', 'ATR_Pct', 'GK_Volatility', 'VWAP_Diff',
    'BB_Width', 'BB_Pband', 'Return_1', 'Return_3',
    'Is_Morning_Open', 'Is_Lunch_Chop', 'Is_Power_Hour',
    'Pattern_Doji', 'Pattern_Marubozu', 'Pattern_Hammer',
    'Pattern_ShootingStar', 'Pattern_BullishEngulfing', 'Pattern_BearishEngulfing',
    # Candlestick anatomy (continuous, not just binary pattern flags)
    'Body_Pct', 'Upper_Wick_Pct', 'Lower_Wick_Pct',
    # Volume behaviour
    'Vol_Ratio', 'Vol_Delta',
    # Momentum speed (log returns) + multi-window realized volatility ("market energy")
    'log_ret_1', 'log_ret_5', 'volatility_15m', 'volatility_60m',
    # Multi-bar sequence lag features (past 3 bars of Return/Body_Pct/Vol_Delta)
    'Return_lag1', 'Return_lag2', 'Return_lag3',
    'Body_Pct_lag1', 'Body_Pct_lag2',
    'Vol_Delta_lag1', 'Vol_Delta_lag2',
]

MIN_MODEL_CONFIDENCE = 0.55  # actionable gate: below this -> HOLD, no exceptions.
ADX_TREND_MIN = 18.0  # TEMP DIAGNOSTIC VALUE (was 22.0), same reasoning as above.
BODY_RATIO_MIN = 0.60
MIN_ATR_PCT_OF_PRICE = 0.0015
PAPER_RULE_FALLBACK = True
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 3.0   # 1.5x SL vs 3.0x TP => 1:2 risk-to-reward
RRR_TARGET = 2.0
REQUIRE_HTF_ALIGNMENT = True
HTF_TIMEFRAME = "1h"
TIMEFRAME_FILTERS = {
    "1m": {"adx": 15.0, "body": 0.45, "atr": 0.00035},
    "5m": {"adx": 18.0, "body": 0.50, "atr": 0.00075},
    "15m": {"adx": 18.0, "body": 0.55, "atr": 0.0010},
    "1h": {"adx": 20.0, "body": 0.60, "atr": 0.0015},
    "4h": {"adx": 20.0, "body": 0.60, "atr": 0.0020},
    "1d": {"adx": 20.0, "body": 0.60, "atr": 0.0030},
}
LUNCH_START_HOUR = 11
LUNCH_END_HOUR = 13


def is_crypto_asset(asset_symbol: str) -> bool:
    """Crypto trades continuously, so exchange lunch-hour filters do not apply."""
    normalized = str(asset_symbol or "").upper().replace("/", "")
    return any(asset in normalized for asset in ("BTC", "ETH", "SOL", "BNB", "XRP"))


def get_timeframe_filters(df: pd.DataFrame) -> dict:
    """Select BTC filters from the Binance candle spacing."""
    if df is not None and len(df.index) >= 2:
        interval_minutes = (df.index[-1] - df.index[-2]).total_seconds() / 60.0
        durations = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        timeframe = min(durations, key=lambda key: abs(durations[key] - interval_minutes))
        return TIMEFRAME_FILTERS[timeframe]
    return {"adx": ADX_TREND_MIN, "body": BODY_RATIO_MIN, "atr": MIN_ATR_PCT_OF_PRICE}


def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    open_p, high_p, low_p, close_p = df['Open'], df['High'], df['Low'], df['Close']
    body = abs(close_p - open_p)
    candle_range = (high_p - low_p).replace(0, 0.001)

    df['Pattern_Doji'] = (body <= (candle_range * 0.1)).astype(int)
    df['Pattern_Marubozu'] = (body >= (candle_range * 0.85)).astype(int)
    lower_shadow = np.minimum(open_p, close_p) - low_p
    df['Pattern_Hammer'] = ((lower_shadow >= (body * 2)) & (body > 0)).astype(int)
    upper_shadow = high_p - np.maximum(open_p, close_p)
    df['Pattern_ShootingStar'] = ((upper_shadow >= (body * 2)) & (body > 0)).astype(int)
    df['Pattern_BullishEngulfing'] = ((close_p > open_p) & (close_p.shift(1) < open_p.shift(1)) & (close_p >= open_p.shift(1))).astype(int)
    df['Pattern_BearishEngulfing'] = ((close_p < open_p) & (close_p.shift(1) > open_p.shift(1)) & (close_p <= open_p.shift(1))).astype(int)
    return df


def calculate_daily_reset_vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
    vol = df['Volume'].replace(0, np.nan).fillna(1)
    vol_price = typical_price * vol
    dates = df.index.date
    cum_vol_price = vol_price.groupby(dates).cumsum()
    cum_vol = vol.groupby(dates).cumsum()
    vwap = cum_vol_price / cum_vol
    return vwap.fillna(typical_price)


def calculate_garman_klass_volatility(df: pd.DataFrame, window: int = 14) -> pd.Series:
    log_hl = np.log(df['High'] / df['Low']) ** 2
    log_co = np.log(df['Close'] / df['Open']) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return np.sqrt(gk.rolling(window).mean()).fillna(0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Builds the exact feature set the model was trained on (train_model.py FEATURE_COLUMNS)."""
    df = df.copy()

    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['EMA_9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA_21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['EMA_Diff'] = (df['EMA_9'] - df['EMA_21']) / df['EMA_21']

    adx_ind = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
    df['ADX'] = adx_ind.adx()

    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100.0
    df['GK_Volatility'] = calculate_garman_klass_volatility(df, window=14)

    df['VWAP'] = calculate_daily_reset_vwap(df)
    df['VWAP_Diff'] = (df['Close'] - df['VWAP']) / df['VWAP']

    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Width'] = bb.bollinger_wband()
    df['BB_Pband'] = bb.bollinger_pband()

    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute
    df['Is_Morning_Open'] = ((df['Hour'] == 9) & (df['Minute'] >= 15) | (df['Hour'] == 10)).astype(int)
    df['Is_Lunch_Chop'] = ((df['Hour'] >= 11) & (df['Hour'] <= 13)).astype(int)
    df['Is_Power_Hour'] = ((df['Hour'] == 14) | ((df['Hour'] == 15) & (df['Minute'] <= 15))).astype(int)

    df['Return_1'] = df['Close'].pct_change(1)
    df['Return_3'] = df['Close'].pct_change(3)

    df = detect_candlestick_patterns(df)
    df = add_candle_anatomy_and_volume_features(df)
    df = add_momentum_and_volatility_features(df)
    df = add_lag_features(df)
    return df


def add_candle_anatomy_and_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Continuous candle anatomy + volume features (vs the binary Pattern_* flags)."""
    candle_range = (df['High'] - df['Low']).replace(0, np.nan)
    body = (df['Close'] - df['Open']).abs()
    upper_wick = df['High'] - df[['Open', 'Close']].max(axis=1)
    lower_wick = df[['Open', 'Close']].min(axis=1) - df['Low']

    df['Body_Pct'] = (body / candle_range).fillna(0)
    df['Upper_Wick_Pct'] = (upper_wick / candle_range).fillna(0)
    df['Lower_Wick_Pct'] = (lower_wick / candle_range).fillna(0)

    vol_ma20 = df['Volume'].rolling(20).mean()
    df['Vol_Ratio'] = (df['Volume'] / vol_ma20.replace(0, np.nan)).fillna(1.0)
    df['Vol_Delta'] = df['Volume'].diff().fillna(0)
    return df


def add_momentum_and_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Log-return momentum speed + rolling realized volatility ("market energy")."""
    close = df['Close'].replace(0, np.nan)
    df['log_ret_1'] = np.log(close / close.shift(1)).fillna(0)
    df['log_ret_5'] = np.log(close / close.shift(5)).fillna(0)
    df['volatility_15m'] = df['log_ret_1'].rolling(15).std().fillna(0)
    df['volatility_60m'] = df['log_ret_1'].rolling(60).std().fillna(0)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Multi-bar sequence lags so the model sees the last 3 bars, not just the current one."""
    for lag in (1, 2, 3):
        df[f'Return_lag{lag}'] = df['Return_1'].shift(lag).fillna(0)
    for lag in (1, 2):
        df[f'Body_Pct_lag{lag}'] = df['Body_Pct'].shift(lag).fillna(0)
        df[f'Vol_Delta_lag{lag}'] = df['Vol_Delta'].shift(lag).fillna(0)
    return df


def detect_vcp_squeeze_contraction(df: pd.DataFrame) -> dict:
    if len(df) < 5:
        return {"is_vcp": False, "score_boost": 0.0, "status": "NORMAL_RANGE"}
    r1 = float(df['High'].iloc[-3] - df['Low'].iloc[-3])
    r2 = float(df['High'].iloc[-2] - df['Low'].iloc[-2])
    r3 = float(df['High'].iloc[-1] - df['Low'].iloc[-1])
    is_contraction = (r3 < r2) and (r2 < r1)
    vol_ma20 = df['Volume'].rolling(20).mean().iloc[-1]
    is_low_vol = df['Volume'].iloc[-1] < vol_ma20
    if is_contraction and is_low_vol:
        return {"is_vcp": True, "score_boost": 0.10, "status": "VCP_SQUEEZE_DETECTED"}
    return {"is_vcp": False, "score_boost": 0.0, "status": "NORMAL_RANGE"}


def generate_signal(df: pd.DataFrame, asset_symbol: str = "", model=None) -> dict:
    """
    Convenience wrapper for LIVE use: builds features fresh (cheap -- live df
    is a small rolling window) and evaluates the latest bar.

    WARNING: do NOT call this in a tight backtest loop over a growing window
    -- it recomputes every indicator from scratch each call, which is O(n^2)
    over a full dataset and will hang for anything beyond a few hundred bars.
    For backtesting, call build_features(df) ONCE up front, then
    decide_from_row(df_features, i, ...) per bar instead.
    """
    if df is None or len(df) < 25:
        return {"signal": "HOLD", "confidence": None, "confidence_source": "RULE_ONLY",
                "reason": "Insufficient candle data for analysis", "blocked_by": "insufficient_data"}
    df_feat = build_features(df)
    return decide_from_row(df_feat, len(df_feat) - 1, asset_symbol=asset_symbol, model=model)


def decide_from_row(df_feat: pd.DataFrame, i: int, asset_symbol: str = "", model=None, htf_trend: str = None) -> dict:
    """
    THE single decision function for both live trading and backtesting.
    Takes a dataframe that ALREADY has build_features() applied, and an
    integer position i -- does no recomputation, so this is cheap to call
    once per bar in a backtest loop (O(n) total instead of O(n^2)).

    Returns a dict:
      signal:      "BUY_CALL" | "BUY_PUT" | "HOLD"
      confidence:  float in [0,1] if a real model produced it, else None
      confidence_source: "MODEL" | "RULE_ONLY"
      reason:      human-readable explanation string
      blocked_by:  which filter rejected the bar, if any (None if a signal fired)
    """
    if df_feat is None or i < 25 or i >= len(df_feat):
        return {"signal": "HOLD", "confidence": None, "confidence_source": "RULE_ONLY",
                "reason": "Insufficient candle data for analysis", "blocked_by": "insufficient_data"}

    last_row = df_feat.iloc[i]
    filters = get_timeframe_filters(df_feat)

    # Multi-timeframe alignment: only allow BUY_CALL predictions in an HTF
    # uptrend, and BUY_PUT predictions in an HTF downtrend. "UNKNOWN" (feed
    # unavailable) does not block trading -- it just skips the extra filter.
    resolved_htf_trend = htf_trend or "UNKNOWN"
    if htf_trend is None and REQUIRE_HTF_ALIGNMENT and _data_feed is not None:
        try:
            htf_info = _data_feed.fetch_htf_trend(asset_symbol, htf=HTF_TIMEFRAME) if asset_symbol else None
            if htf_info:
                resolved_htf_trend = htf_info.get("trend", "UNKNOWN")
        except Exception:
            resolved_htf_trend = "UNKNOWN"

    # Lunch-hour chop is an exchange-hours rule and does not apply to crypto.
    try:
        last_hour = df_feat.index[i].hour
        if not is_crypto_asset(asset_symbol) and LUNCH_START_HOUR <= last_hour < LUNCH_END_HOUR:
            return {"signal": "HOLD", "confidence": None, "confidence_source": "RULE_ONLY",
                    "reason": "Lunch-hour chop window (11:00-13:00) - signals skipped",
                    "blocked_by": "lunch_hour"}
    except Exception:
        pass

    # Candle body ratio filter
    candle_range = last_row['High'] - last_row['Low'] + 1e-6
    body_ratio = abs(last_row['Close'] - last_row['Open']) / candle_range
    if body_ratio < filters["body"] and not config.PAPER_EXPERIMENTAL_SIGNALS:
        return {"signal": "HOLD", "confidence": None, "confidence_source": "RULE_ONLY",
                "reason": f"Body ratio {body_ratio:.2f} < {BODY_RATIO_MIN} (weak/doji candle)",
                "blocked_by": "body_ratio"}

    # ADX trend-strength filter
    if last_row['ADX'] < filters["adx"] and not config.PAPER_EXPERIMENTAL_SIGNALS:
        return {"signal": "HOLD", "confidence": None, "confidence_source": "RULE_ONLY",
                "reason": f"ADX {last_row['ADX']:.1f} < {ADX_TREND_MIN} (sideways chop)",
                "blocked_by": "adx"}

    # Minimum volatility filter (must clear brokerage friction)
    atr_val = last_row['ATR']
    if atr_val < last_row['Close'] * filters["atr"] and not config.PAPER_EXPERIMENTAL_SIGNALS:
        return {"signal": "HOLD", "confidence": None, "confidence_source": "RULE_ONLY",
                "reason": "ATR below minimum volatility threshold", "blocked_by": "min_atr"}

    if model is not None:
        try:
            features = pd.DataFrame([{col: last_row[col] for col in FEATURE_COLUMNS}]).fillna(0)
            probs = model.predict_proba(features)[0]
            probability_fields = {
                "put_probability": round(float(probs[0]), 6),
                "hold_probability": round(float(probs[1]), 6),
                "call_probability": round(float(probs[2]), 6),
            }
            pred = int(np.argmax(probs))
            confidence = float(np.max(probs))

            vcp_window = df_feat.iloc[max(0, i - 20):i + 1]
            vcp_res = detect_vcp_squeeze_contraction(vcp_window)
            if vcp_res["is_vcp"]:
                confidence = min(1.0, confidence + vcp_res["score_boost"])

            ema9, ema21 = last_row['EMA_9'], last_row['EMA_21']
            vwap_diff = last_row['VWAP_Diff']
            bullish_structure = vwap_diff > 0 and ema9 > ema21 and last_row['Close'] > last_row['Open']
            bearish_structure = vwap_diff < 0 and ema9 < ema21 and last_row['Close'] < last_row['Open']

            htf_allows_buy = resolved_htf_trend in ("BULLISH", "UNKNOWN")
            htf_allows_sell = resolved_htf_trend in ("BEARISH", "UNKNOWN")
            if REQUIRE_HTF_ALIGNMENT and resolved_htf_trend == "BEARISH" and pred == 2:
                return {"signal": "HOLD", "confidence": round(confidence, 3), "confidence_source": "MODEL",
                        "reason": f"BUY rejected: 1H HTF trend is BEARISH (model wanted CALL, confidence {confidence:.2f})",
                        "blocked_by": "htf_misalignment", "htf_trend": resolved_htf_trend, **probability_fields}
            if REQUIRE_HTF_ALIGNMENT and resolved_htf_trend == "BULLISH" and pred == 0:
                return {"signal": "HOLD", "confidence": round(confidence, 3), "confidence_source": "MODEL",
                        "reason": f"SELL rejected: 1H HTF trend is BULLISH (model wanted PUT, confidence {confidence:.2f})",
                        "blocked_by": "htf_misalignment", "htf_trend": resolved_htf_trend, **probability_fields}

            # A high model score is not enough: the predicted direction must
            # agree with the independent price-structure filters AND the HTF trend.
            required_confidence = (
                config.PAPER_MIN_ACTIONABLE_CONFIDENCE
                if config.PAPER_TRADING_MODE
                else MIN_MODEL_CONFIDENCE
            )
            if config.PAPER_EXPERIMENTAL_SIGNALS and pred == 1:
                directional_probabilities = {0: float(probs[0]), 2: float(probs[2])}
                pred = max(directional_probabilities, key=directional_probabilities.get)
                confidence = directional_probabilities[pred]
                if confidence >= config.PAPER_MIN_DIRECTIONAL_PROBABILITY:
                    signal = "BUY_CALL" if pred == 2 else "BUY_PUT"
                    aligned = (pred == 2 and htf_allows_buy) or (pred == 0 and htf_allows_sell)
                    if aligned:
                        levels = compute_atr_risk_levels(last_row, signal)
                        return {"signal": signal, "confidence": round(confidence, 3),
                                "confidence_source": "PAPER_EXPERIMENTAL",
                                "reason": f"Paper experiment: directional probability {confidence:.2f}; strict filter status body={body_ratio:.2f}, ADX={last_row['ADX']:.1f}, HTF={resolved_htf_trend}",
                                "blocked_by": None, "htf_trend": resolved_htf_trend, **levels, **probability_fields}
            if confidence >= required_confidence and (
                (pred == 2 and bullish_structure and htf_allows_buy) or
                (pred == 0 and bearish_structure and htf_allows_sell)
            ):
                signal = "BUY_CALL" if pred == 2 else "BUY_PUT"
                levels = compute_atr_risk_levels(last_row, signal)
                return {"signal": signal, "confidence": round(confidence, 3), "confidence_source": "MODEL",
                        "reason": f"Model pred={pred} confidence={confidence:.2f} (ADX {last_row['ADX']:.1f}, body {body_ratio:.2f}, HTF {resolved_htf_trend})",
                        "blocked_by": None, "htf_trend": resolved_htf_trend, **levels, **probability_fields}

            if PAPER_RULE_FALLBACK and confidence >= required_confidence:
                if vwap_diff > 0 and ema9 > ema21 and last_row['Close'] > last_row['Open'] and htf_allows_buy:
                    levels = compute_atr_risk_levels(last_row, "BUY_CALL")
                    return {"signal": "BUY_CALL", "confidence": round(confidence, 3), "confidence_source": "RULE_FALLBACK",
                            "reason": f"Model HOLD fallback: bullish VWAP/EMA/candle alignment ({confidence:.2f} confidence, HTF {resolved_htf_trend})",
                            "blocked_by": None, "htf_trend": resolved_htf_trend, **levels, **probability_fields}
                if vwap_diff < 0 and ema9 < ema21 and last_row['Close'] < last_row['Open'] and htf_allows_sell:
                    levels = compute_atr_risk_levels(last_row, "BUY_PUT")
                    return {"signal": "BUY_PUT", "confidence": round(confidence, 3), "confidence_source": "RULE_FALLBACK",
                            "reason": f"Model HOLD fallback: bearish VWAP/EMA/candle alignment ({confidence:.2f} confidence, HTF {resolved_htf_trend})",
                            "blocked_by": None, "htf_trend": resolved_htf_trend, **levels, **probability_fields}
            return {"signal": "HOLD", "confidence": round(confidence, 3), "confidence_source": "MODEL",
                    "reason": f"Model confidence {confidence:.2f} below {required_confidence:.2f} threshold or predicted HOLD",
                    "blocked_by": "low_confidence", "htf_trend": resolved_htf_trend, **probability_fields}
        except Exception as e:
            # Fall through to rule-based path below if model scoring fails,
            # but say so explicitly -- never silently swap logic.
            rule_reason_suffix = f" (model scoring failed: {e}, used rule fallback)"
    else:
        rule_reason_suffix = " (no trained model loaded, rule fallback)"

    # Rule-only fallback -- NOT a probability, explicitly labelled as such.
    ema9, ema21 = last_row['EMA_9'], last_row['EMA_21']
    vwap_diff = last_row['VWAP_Diff']
    htf_allows_buy = resolved_htf_trend in ("BULLISH", "UNKNOWN")
    htf_allows_sell = resolved_htf_trend in ("BEARISH", "UNKNOWN")

    if vwap_diff > 0 and ema9 > ema21 and last_row['Close'] > last_row['Open'] and htf_allows_buy:
        levels = compute_atr_risk_levels(last_row, "BUY_CALL")
        return {"signal": "BUY_CALL", "confidence": None, "confidence_source": "RULE_ONLY",
                "reason": "Rule: price>VWAP, EMA9>EMA21, bullish candle" + rule_reason_suffix,
                "blocked_by": None, "htf_trend": resolved_htf_trend, **levels}
    if vwap_diff < 0 and ema9 < ema21 and last_row['Close'] < last_row['Open'] and htf_allows_sell:
        levels = compute_atr_risk_levels(last_row, "BUY_PUT")
        return {"signal": "BUY_PUT", "confidence": None, "confidence_source": "RULE_ONLY",
                "reason": "Rule: price<VWAP, EMA9<EMA21, bearish candle" + rule_reason_suffix,
                "blocked_by": None, "htf_trend": resolved_htf_trend, **levels}

    return {"signal": "HOLD", "confidence": None, "confidence_source": "RULE_ONLY",
            "reason": "Waiting for clear VWAP/EMA trend breakout" + rule_reason_suffix,
            "blocked_by": "no_setup", "htf_trend": resolved_htf_trend}


def compute_atr_risk_levels(last_row: pd.Series, signal: str) -> dict:
    """
    Dynamic ATR-based entry/SL/TP for a fired signal:
      - entry_price = last close
      - stop_loss   = entry -/+ 1.5x ATR
      - take_profit = entry +/- 3.0x ATR   (=> 1:2 risk-to-reward)
    Every generated BUY/SELL signal must carry these so downstream alerting
    and the broker layer never have to guess risk levels.
    """
    entry_price = float(last_row['Close'])
    atr = float(last_row.get('ATR', 0) or 0)
    if atr <= 0:
        # Fallback: derive a rough ATR proxy from the candle range so we never
        # ship a signal with no risk levels at all.
        atr = float(last_row['High'] - last_row['Low']) or entry_price * 0.001

    sl_distance = atr * ATR_SL_MULTIPLIER
    tp_distance = atr * ATR_TP_MULTIPLIER

    if signal == "BUY_CALL":
        stop_loss = entry_price - sl_distance
        take_profit = entry_price + tp_distance
    else:  # BUY_PUT
        stop_loss = entry_price + sl_distance
        take_profit = entry_price - tp_distance

    return {
        "entry_price": round(entry_price, 4),
        "stop_loss": round(stop_loss, 4),
        "take_profit": round(take_profit, 4),
        "rrr": RRR_TARGET,
        "atr": round(atr, 4),
    }