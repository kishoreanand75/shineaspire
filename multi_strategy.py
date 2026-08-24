# multi_strategy.py - With Direct Telegram Mobile Alerts & Hurst Engine
import os
import logging
import streamlit as st
import datetime
import yfinance as yf
import pandas as pd
import ta
import numpy as np
import config
import data_feed
import trade_logger

logger = logging.getLogger(__name__)
from xgboost import XGBClassifier
from notifier import send_telegram_alert, send_formatted_signal_alert
from signal_engine import compute_atr_risk_levels, generate_signal as _engine_generate_signal

MODEL_FILE = "xgboost_model.json"
model = None
model_mtime = None

def reload_model_if_updated():
    """Reload the trained model only after the background trainer replaces it."""
    global model, model_mtime
    if not os.path.exists(MODEL_FILE):
        return False
    current_mtime = os.path.getmtime(MODEL_FILE)
    if model is not None and model_mtime == current_mtime:
        return False
    try:
        candidate = XGBClassifier()
        candidate.load_model(MODEL_FILE)
        model = candidate
        model_mtime = current_mtime
        print(f"[MODEL] Loaded model updated at {datetime.datetime.fromtimestamp(current_mtime).isoformat()}")
        return True
    except Exception as exc:
        print(f"[MODEL] Reload skipped: {exc}")
        return False


reload_model_if_updated()

def detect_vcp_squeeze_contraction(df: pd.DataFrame) -> dict:
    """Detects Volatility Contraction Pattern (VCP) - Shrinking Range Before Explosive Breakout"""
    if len(df) < 5:
        return {"is_vcp": False, "score_boost": 0.0, "status": "NORMAL_RANGE"}

    # Calculate Candle Ranges for last 3 bars
    r1 = float(df['High'].iloc[-3] - df['Low'].iloc[-3])
    r2 = float(df['High'].iloc[-2] - df['Low'].iloc[-2])
    r3 = float(df['High'].iloc[-1] - df['Low'].iloc[-1])

    # VCP Rule: Volatility is shrinking (r3 < r2 < r1)
    is_contraction = (r3 < r2) and (r2 < r1)
    
    # Volume Contraction Check
    vol_ma20 = df['Volume'].rolling(20).mean().iloc[-1]
    is_low_vol = df['Volume'].iloc[-1] < vol_ma20

    # If VCP Squeeze is detected, boost AI Confidence score by +10%
    if is_contraction and is_low_vol:
        return {"is_vcp": True, "score_boost": 0.10, "status": "🎯 VCP SQUEEZE DETECTED (+10% AI Confidence Boost)"}
        
    return {"is_vcp": False, "score_boost": 0.0, "status": "NORMAL_RANGE"}

def detect_liquidity_sweep_trap(df: pd.DataFrame, pdh: float, pdl: float) -> dict:
    """Detects Institutional Liquidity Sweeps above PDH or below PDL for High-Probability Reversals"""
    if df is None or len(df) < 2:
        return {"signal": "NONE", "confidence_boost": 0.0, "status": "NORMAL"}

    last_candle = df.iloc[-1]
    candle_range = last_candle['High'] - last_candle['Low'] + 1e-6

    # 1. Bearish Liquidity Sweep (Spiked above PDH but closed inside range with long upper wick)
    if last_candle['High'] > pdh and last_candle['Close'] < pdh:
        upper_wick = last_candle['High'] - max(last_candle['Open'], last_candle['Close'])
        if (upper_wick / candle_range) >= 0.40: # Long Upper Wick Trap
            return {
                "signal": "BUY_PUT",
                "confidence_boost": 0.15,
                "status": "🚨 BEARISH LIQUIDITY SWEEP TRAP AT PDH (+15% AI Boost)"
            }

    # 2. Bullish Liquidity Sweep (Spiked below PDL but closed inside range with long lower wick)
    if last_candle['Low'] < pdl and last_candle['Close'] > pdl:
        lower_wick = min(last_candle['Open'], last_candle['Close']) - last_candle['Low']
        if (lower_wick / candle_range) >= 0.40: # Long Lower Wick Trap
            return {
                "signal": "BUY_CALL",
                "confidence_boost": 0.15,
                "status": "🚀 BULLISH LIQUIDITY SWEEP TRAP AT PDL (+15% AI Boost)"
            }

    return {"signal": "NONE", "confidence_boost": 0.0, "status": "NORMAL"}


def _rejection_wick_sweep(last_candle, level_high: float, level_low: float, side: str) -> dict:
    """Shared wick-rejection check used against ANY institutional liquidity level
    (Previous-Day High/Low or Asian-Session High/Low). >=40% rejection wick required."""
    candle_range = last_candle['High'] - last_candle['Low'] + 1e-6
    if side == "HIGH" and last_candle['High'] > level_high and last_candle['Close'] < level_high:
        upper_wick = last_candle['High'] - max(last_candle['Open'], last_candle['Close'])
        if (upper_wick / candle_range) >= 0.40:
            return {"swept": True, "signal": "BUY_PUT"}
    if side == "LOW" and last_candle['Low'] < level_low and last_candle['Close'] > level_low:
        lower_wick = min(last_candle['Open'], last_candle['Close']) - last_candle['Low']
        if (lower_wick / candle_range) >= 0.40:
            return {"swept": True, "signal": "BUY_CALL"}
    return {"swept": False, "signal": "NONE"}


def get_asian_session_high_low(df: pd.DataFrame) -> dict:
    """
    Asian Session = 00:00-08:00 UTC (~5:30 AM - 1:30 PM IST). Returns the High/Low
    of the most recently COMPLETED Asian session found in df's index (UTC-based).
    Returns {"high": None, "low": None} if no complete session is available yet.
    """
    if df is None or df.empty:
        return {"high": None, "low": None}
    idx = df.index
    dates = pd.Series(idx.date, index=idx)
    hours = pd.Series(idx.hour, index=idx)
    is_asian = (hours >= 0) & (hours < 8)

    today = idx[-1].date()
    candidate_dates = sorted({d for d in dates[is_asian].unique() if d <= today}, reverse=True)
    for session_date in candidate_dates:
        # A session counts as "completed" once we have data past 08:00 UTC that date,
        # OR it's a prior day (always complete by definition).
        mask = is_asian & (dates == session_date)
        if session_date < today:
            session_slice = df.loc[mask]
            if not session_slice.empty:
                return {"high": float(session_slice['High'].max()), "low": float(session_slice['Low'].min())}
        else:
            # Today's Asian session: only use it once we're past 08:00 UTC.
            if idx[-1].hour >= 8:
                session_slice = df.loc[mask]
                if not session_slice.empty:
                    return {"high": float(session_slice['High'].max()), "low": float(session_slice['Low'].min())}
    return {"high": None, "low": None}


def detect_session_liquidity_sweeps(df: pd.DataFrame, pdh: float, pdl: float) -> dict:
    """
    Institutional liquidity sweep detector covering BOTH:
      1. Real UTC Previous Day High/Low (PDH/PDL)
      2. Asian Session (00:00-08:00 UTC) High/Low -- the level London/NY sessions
         commonly hunt with a fake breakout before reversing.

    A valid sweep requires the wick beyond the level to be >= 40% of the candle's
    range (a genuine rejection, not a clean breakout). Keeps the same +0.15
    (+15%) AI confidence boost as the legacy PDH/PDL-only detector.
    """
    if df is None or len(df) < 2:
        return {"signal": "NONE", "confidence_boost": 0.0, "status": "NORMAL", "swept_level": None}

    last_candle = df.iloc[-1]
    asian = get_asian_session_high_low(df.iloc[:-1])  # exclude the forming/last candle from the session calc

    checks = [
        ("PDH", pdh, None, "HIGH", "🚨 BEARISH LIQUIDITY SWEEP TRAP AT PDH (+15% AI Boost)"),
        ("PDL", None, pdl, "LOW", "🚀 BULLISH LIQUIDITY SWEEP TRAP AT PDL (+15% AI Boost)"),
    ]
    if asian["high"] is not None:
        checks.append(("ASIA_HIGH", asian["high"], None, "HIGH",
                        "🚨 BEARISH LIQUIDITY SWEEP TRAP AT ASIAN SESSION HIGH (+15% AI Boost)"))
    if asian["low"] is not None:
        checks.append(("ASIA_LOW", None, asian["low"], "LOW",
                        "🚀 BULLISH LIQUIDITY SWEEP TRAP AT ASIAN SESSION LOW (+15% AI Boost)"))

    for level_name, level_high, level_low, side, status_msg in checks:
        res = _rejection_wick_sweep(last_candle, level_high if level_high is not None else float('inf'),
                                     level_low if level_low is not None else float('-inf'), side)
        if res["swept"]:
            return {"signal": res["signal"], "confidence_boost": 0.15, "status": status_msg, "swept_level": level_name}

    return {"signal": "NONE", "confidence_boost": 0.0, "status": "NORMAL", "swept_level": None}


def get_previous_utc_day_high_low(df: pd.DataFrame) -> tuple:
    """Return the completed UTC day's high/low, excluding today's candles."""
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return None, None
    index = df.index
    if index.tz is not None:
        utc_index = index.tz_convert("UTC")
    else:
        utc_index = index
    latest_day = utc_index[-1].date()
    previous_day = latest_day - datetime.timedelta(days=1)
    mask = utc_index.date == previous_day
    previous = df.loc[mask]
    if previous.empty:
        return None, None
    return float(previous["High"].max()), float(previous["Low"].min())


def evaluate_pyramiding_scaling(current_gain_pct: float, vcp_active: bool) -> dict:
    """Zero-Risk Pyramiding Position Scaling Logic"""
    # Trigger Pyramiding only if Target 1 (+6%) is hit and VCP Momentum is present
    if current_gain_pct >= 0.06 and vcp_active:
        return {
            "allow_pyramiding": True,
            "additional_qty_pct": 0.50, # Add 50% additional scaling lot
            "sl_action": "SHIFT_TO_BREAKEVEN",
            "status": "🔥 PYRAMIDING SCALING ACTIVE (Zero Risk Mode)"
        }
    return {"allow_pyramiding": False, "additional_qty_pct": 0.0, "sl_action": "NORMAL", "status": "NORMAL"}

def check_kill_switch_status(consecutive_losses: int) -> dict:
    """Checks Kill-Switch status and respects Extended Testing Mode Override"""
    
    # Check if user enabled Extended Testing Mode Toggle in Sidebar
    is_testing_override = False
    try:
        if hasattr(st, 'session_state'):
            is_testing_override = st.session_state.get('allow_extended_trades', False)
    except Exception:
        pass
    
    if consecutive_losses >= 2:
        if is_testing_override:
            # TESTING OVERRIDE: Allow scanning, but raise AI Confidence threshold to 75%
            return {
                "is_locked": False,
                "min_confidence": 0.75, # High Bar for testing after 2 losses
                "status_msg": "🧪 TESTING OVERRIDE: 2 Losses Bypassed for Market Analysis (Requires 75%+ AI Confidence)"
            }
        else:
            # Production Mode: Hard Shut-off for safety
            return {
                "is_locked": True,
                "min_confidence": 0.70,
                "status_msg": "🛑 CONSECUTIVE LOSS KILL-SWITCH: Locked for today to protect capital."
            }
            
    return {"is_locked": False, "min_confidence": 0.70, "status_msg": "NORMAL"}

def evaluate_smart_breakout_signals(df: pd.DataFrame, asset_symbol: str) -> dict:
    """Smart ATR Volatility Expansion & Friction Filter Strategy Engine"""
    if df is None or len(df) < 20:
        return {"signal": "HOLD", "confidence": 0.50, "reason": "Insufficient candle data for analysis"}

    df = df.copy()
    last_row = df.iloc[-1]

    # Crypto trades continuously; no stock-market lunch-hour block applies.
    # 1. Ezekiel Chew Candle Body Ratio Filter (>= 60%)
    candle_range = last_row['High'] - last_row['Low'] + 1e-6
    body_size = abs(last_row['Close'] - last_row['Open'])
    body_ratio = body_size / candle_range

    if body_ratio < 0.60:
        return {"signal": "HOLD", "confidence": 0.50, "reason": f"⚠️ Body ratio {body_ratio:.2f} < 0.60 (Weak Wick / Doji Candle Rejected)"}

    # 2. ADX Trend Strength Filter (> 22.0) - Rejects Dead Sideways Chop
    adx_ind = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    adx_val = adx_ind.adx().iloc[-1]
    if adx_val < 22.0:
        return {"signal": "HOLD", "confidence": 0.50, "reason": f"⚠️ ADX {adx_val:.1f} < 22.0 (Dead Sideways Chop - Signal Rejected)"}

    # 3. Volume Spike Filter (>= 1.20x 20-MA)
    # Pure indices (^NSEI, ^NSEBANK, etc.) have no tradable volume of their
    # own -- Yahoo Finance always reports Volume=0 for them, which made this
    # filter reject every single bar unconditionally for NIFTY/BANKNIFTY.
    # Only apply the volume check to symbols that actually have real volume
    # (stocks, crypto). An index ticker is identified by its "^" prefix.
    is_index_symbol = str(asset_symbol).startswith("^") or "NIFTY" in str(asset_symbol).upper() or "BANKNIFTY" in str(asset_symbol).upper()
    vol_ratio = 0.0
    if not is_index_symbol:
        vol_ma20 = df['Volume'].rolling(20).mean().iloc[-1]
        vol_ratio = last_row['Volume'] / (vol_ma20 + 1e-6)
        if vol_ratio < 1.20:
            return {"signal": "HOLD", "confidence": 0.50, "reason": f"⚠️ Volume {vol_ratio:.2f}x < 1.20x Average (Low Institutional Volume)"}

    # 4. Minimum Volatility Threshold (Expected Move > Brokerage Friction ₹50)
    atr_val = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14).iloc[-1]
    min_required_atr = last_row['Close'] * 0.0015 # Minimum 0.15% Move Requirement
    if atr_val < min_required_atr:
        return {"signal": "HOLD", "confidence": 0.50, "reason": f"⚠️ ATR {atr_val:.2f} < Volatility Threshold (Potential Brokerage Fee Trap)"}

    # 5. EMA 9/21 Trend & VWAP Direction Alignment
    ema9 = ta.trend.ema_indicator(df['Close'], window=9).iloc[-1]
    ema21 = ta.trend.ema_indicator(df['Close'], window=21).iloc[-1]
    # Volume-weighted VWAP breaks (0/0 = NaN) when Volume is always 0, which
    # is exactly the case for pure indices like ^NSEI/^NSEBANK -- that NaN
    # made every Close>vwap and Close<vwap comparison silently False forever,
    # blocking every signal. Fall back to volume=1 (equal-weighted / simple
    # typical-price average) when real volume isn't available, same pattern
    # already used in calculate_daily_reset_vwap() above.
    safe_volume = df['Volume'].replace(0, np.nan).fillna(1)
    vwap_series = (safe_volume * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / safe_volume.cumsum()
    vwap_curr = vwap_series.iloc[-1]

    # BUY CALL Condition
    if last_row['Close'] > vwap_curr and ema9 > ema21 and last_row['Close'] > last_row['Open']:
        return {
            "signal": "BUY_CALL",
            "confidence": 0.72,
            "reason": f"🚀 High-Conviction Bullish Breakout: Body {body_ratio:.2f} | ADX {adx_val:.1f} | Vol {vol_ratio:.2f}x | ATR {atr_val:.2f}"
        }

    # BUY PUT Condition
    if last_row['Close'] < vwap_curr and ema9 < ema21 and last_row['Close'] < last_row['Open']:
        return {
            "signal": "BUY_PUT",
            "confidence": 0.72,
            "reason": f"🚨 High-Conviction Bearish Breakdown: Body {body_ratio:.2f} | ADX {adx_val:.1f} | Vol {vol_ratio:.2f}x | ATR {atr_val:.2f}"
        }

    return {"signal": "HOLD", "confidence": 0.50, "reason": "⏸️ Waiting for clear VWAP/EMA Trend Breakout"}

WATCHLIST = {"BITCOIN": config.DEFAULT_SYMBOL}

last_notified_signal = {}

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

def scan_all_assets():
    reload_model_if_updated()
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    
    best_opportunity = None
    scanned_results = []

    for name, symbol in WATCHLIST.items():
        try:
            df = data_feed.fetch_btc_live_data(symbol, timeframe=config.TRADE_TIMEFRAME)
            df = df.rename(columns={col: col.title() for col in df.columns})

            # This used to "continue" silently, making a dead data feed look
            # identical to "no signal this bar" in every downstream log/UI.
            # data_feed already logs the root cause (CRITICAL if truly no
            # data at all); this makes the scanner-level consequence visible
            # too, so a stuck feed doesn't masquerade as a calm market.
            if df.empty or len(df) < 25:
                logger.warning(
                    "Skipping %s this scan: got %d bars (need >=25). "
                    "Check data_feed logs -- this may be a dead feed, not a quiet market.",
                    symbol, len(df) if df is not None else 0
                )
                continue

            # Calculate real completed UTC previous-day liquidity levels.
            pdh_val, pdl_val = get_previous_utc_day_high_low(df.iloc[:-1])
            pdh_val = pdh_val if pdh_val is not None else float("inf")
            pdl_val = pdl_val if pdl_val is not None else float("-inf")

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

            latest = df.iloc[-2]

            # SINGLE SOURCE OF TRUTH: same generate_signal() the backtester
            # calls. Do not re-implement scoring logic here -- see
            # signal_engine.py header for why that caused backtest/live
            # divergence before.
            engine_result = _engine_generate_signal(df.iloc[:-1], asset_symbol=symbol, model=model)
            signal = engine_result["signal"]

            # Liquidity Sweep Boost stays here (asset-specific PDH/PDL context
            # that generate_signal() doesn't have) -- can upgrade a HOLD-level
            # confidence read into a signal only when the engine already saw
            # a tradeable setup; it must not override an engine HOLD outright.
            sweep_res = detect_session_liquidity_sweeps(df, pdh_val, pdl_val)
            if signal == "HOLD" and sweep_res["signal"] != "NONE" and engine_result.get("confidence") is not None \
                    and engine_result["confidence"] + sweep_res["confidence_boost"] >= config.MIN_ACTIONABLE_CONFIDENCE:
                signal = sweep_res["signal"]
                engine_result = {
                    **engine_result,
                    "signal": signal,
                    "confidence": round(min(1.0, engine_result["confidence"] + sweep_res["confidence_boost"]), 3),
                    "confidence_source": "MODEL_PLUS_LIQUIDITY_SWEEP",
                    "reason": f"{sweep_res['status']}; base reason: {engine_result.get('reason', '')}",
                    "sweep_level": sweep_res.get("swept_level"),
                    **compute_atr_risk_levels(df.iloc[-2], signal),
                }

            gemini_reason = ""
            body_ratio = float(abs(latest['Close'] - latest['Open']) / max(0.001, latest['High'] - latest['Low']))
            if signal != "HOLD":
                from ai_analyst import ask_gemini_trade_validation
                opt_type = "CALL" if signal == "BUY_CALL" else "PUT"
                vwap_dist = float(latest['VWAP_Diff'] * 100.0)
                body = abs(latest['Close'] - latest['Open'])
                candle_range = max(0.001, (latest['High'] - latest['Low']))
                body_ratio = float(body / candle_range)
                
                gemini_res = ask_gemini_trade_validation(name, opt_type, float(latest['RSI']), vwap_dist, body_ratio)
                
                if gemini_res.get("decision") == "APPROVED":
                    gemini_reason = gemini_res.get("reason", "Approved by Gemini AI.")
                    print(f"✅ Gemini Approved {signal} for {name}: {gemini_reason}")
                else:
                    gemini_reason = gemini_res.get("reason", "Rejected by Gemini AI.")
                    print(f"⚠️ Gemini Rejected {signal} for {name}: {gemini_reason}")
                    signal = "HOLD"

            if signal != "HOLD" and last_notified_signal.get(name) != signal:
                last_notified_signal[name] = signal
                # Prefer the fully-formatted alert (Entry/SL/TP/Confidence/HTF trend)
                # whenever the final signal still matches what signal_engine.py
                # produced -- it carries the ATR risk levels. If a liquidity-sweep
                # upgrade changed the signal, those levels don't apply, so fall
                # back to the plain alert rather than showing stale risk levels.
                sent_formatted = False
                if engine_result.get("signal") == signal and engine_result.get("entry_price") is not None:
                    sent_formatted = send_formatted_signal_alert(symbol, engine_result, name=name)
                if not sent_formatted:
                    alert_msg = f"🚨 <b>AI TRADE SIGNAL DETECTED!</b>\n\n<b>Asset:</b> {name}\n<b>Signal:</b> {signal}\n<b>Live Price:</b> {latest['Close']:,.2f}\n<b>RSI:</b> {latest['RSI']:.1f}\n<b>Gemini Validation:</b> {gemini_reason}\n<b>Time:</b> {now_dt.strftime('%H:%M:%S IST')}"
                    send_telegram_alert(alert_msg)

            trade_logger.record_signal_event(
                symbol, df.index[-2], signal, engine_result.get("confidence"),
                engine_result.get("reason", gemini_reason), engine_result.get("htf_trend", "UNKNOWN")
            )

            scanned_results.append({
                "Name": name, "Symbol": symbol, "Price": latest['Close'], "RSI": latest['RSI'], "Signal": signal,
                "Timeframe": config.TRADE_TIMEFRAME, "Candle_Time": str(df.index[-2]),
                "Entry_Window": "1-4 minutes after candle close",
                "Confidence": engine_result.get("confidence"),
                "Confidence_Source": engine_result.get("confidence_source", ""),
                "Entry_Price": engine_result.get("entry_price"),
                "Stop_Loss": engine_result.get("stop_loss"),
                "Take_Profit": engine_result.get("take_profit"),
                "HTF_Trend": engine_result.get("htf_trend", "UNKNOWN"),
                "Candle_Open": float(latest["Open"]), "Candle_High": float(latest["High"]),
                "Candle_Low": float(latest["Low"]), "Candle_Close": float(latest["Close"]),
                "Signal_Reason": engine_result.get("reason", gemini_reason),
                "Market_Context": {
                    "rsi": float(latest["RSI"]),
                    "vwap_diff": float(latest["VWAP_Diff"]),
                    "body_ratio": float(body_ratio),
                },
            })

            if signal != "HOLD" and best_opportunity is None:
                best_opportunity = {
                    "Name": name, "Symbol": symbol, "Price": latest['Close'], "Signal": signal,
                    "Candle_Time": str(df.index[-2]),
                    "Confidence": engine_result.get("confidence"),
                    "Confidence_Source": engine_result.get("confidence_source", ""),
                    "Entry_Price": engine_result.get("entry_price"),
                    "Stop_Loss": engine_result.get("stop_loss"),
                    "Take_Profit": engine_result.get("take_profit"),
                    "HTF_Trend": engine_result.get("htf_trend", "UNKNOWN"),
                    "Signal_Reason": engine_result.get("reason", gemini_reason),
                    "Market_Context": scanned_results[-1]["Market_Context"],
                }

        except Exception as e:
            print(f"\n[SCAN ERROR - {name}] {type(e).__name__}: {e}")
            scanned_results.append({
                "Name": name, "Symbol": symbol, "Price": None, "RSI": None,
                "Signal": "ERROR", "Confidence": None,
                "Signal_Reason": f"Scanner error: {type(e).__name__}: {e}",
                "Timeframe": config.TRADE_TIMEFRAME,
            })
            continue

    return best_opportunity, scanned_results

def is_daily_limit_reached(completed_trades_count: int) -> bool:
    """Check if 3 trades daily limit is reached (Bypassed if Extended Testing Mode is ON)"""
    
    # Check if user enabled the temporary testing toggle in Dashboard
    is_testing_mode_on = st.session_state.get('allow_extended_trades', False)
    
    if is_testing_mode_on:
        # Testing Mode is ON -> Allow scanning beyond 3 trades
        return False
        
    # Default Safe Mode -> Hard Lock at 3 completed trades
    return completed_trades_count >= 3

def is_safe_entry_window_in_candle() -> bool:
    """Ensure entry happens only between 2nd and 4th minute of the 5-min candle (60s to 240s)"""
    current_second = datetime.datetime.now().second + (datetime.datetime.now().minute % 5) * 60
    
    # 60 seconds <= current_second <= 240 seconds (Safest Entry Zone)
    if 60 <= current_second <= 240:
        return True
        
    return False # Rejects entry in 1st minute and last 1 minute of the candle

def calculate_dynamic_atr_levels(df: pd.DataFrame, entry_price: float, signal_type: str, atr_multiplier_sl: float = 1.5, atr_multiplier_tp1: float = 1.5, atr_multiplier_tp2: float = 3.0):
    """Dynamic ATR-Based Stop Loss & Target Calculation"""
    df = df.copy()
    atr = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14).iloc[-1]
    
    if signal_type == "BUY_CALL":
        sl = entry_price - (atr * atr_multiplier_sl)
        tp1 = entry_price + (atr * atr_multiplier_tp1)
        tp2 = entry_price + (atr * atr_multiplier_tp2)
    else: # BUY_PUT
        sl = entry_price + (atr * atr_multiplier_sl)
        tp1 = entry_price - (atr * atr_multiplier_tp1)
        tp2 = entry_price - (atr * atr_multiplier_tp2)
        
    return round(sl, 2), round(tp1, 2), round(tp2, 2), round(atr, 2)

def evaluate_soft_kill_switch_position_scaling(consecutive_losses: int):
    """Soft Kill-Switch: Position Sizing & Confidence Threshold Adjustment"""
    if consecutive_losses >= 2:
        # Soft Lock: Scale Position Size to 50% & Require 75% AI Confidence
        return {
            "position_scale_factor": 0.50, # 50% Position Size
            "min_ai_confidence": 0.75,     # Higher Bar for 3rd Trade (75%)
            "required_adx": 28.0,          # Strong Trend Only
            "status": "SOFT_KILL_SWITCH_ACTIVE"
        }
    return {
        "position_scale_factor": 1.00,
        "min_ai_confidence": 0.70,
        "required_adx": 25.0,
        "status": "NORMAL"
    }

def is_safe_mid_candle_window() -> bool:
    """Rule #4: Safest Entry Window (2nd to 4th minute inside 5-min candle: 60s to 240s)"""
    now = datetime.datetime.now()
    second_in_candle = now.second + (now.minute % 5) * 60
    return 60 <= second_in_candle <= 240


# =============================================================
# MULTI-COIN CRYPTO RADAR SCANNER (BTC, ETH, SOL, BNB, XRP)
# =============================================================

CRYPTO_RADAR_PAIRS = ["BTC/USDT"]

def evaluate_single_asset_signal(df: pd.DataFrame, symbol: str, ai_model_tuple=None):
    """Evaluates strategy indicators & ML probability for a single symbol"""
    if df is None or len(df) < 20:
        return "HOLD", 0.50, "Insufficient Data", {}

    df = df.copy()
    df_cols = {col.lower(): col for col in df.columns}
    c_col = df_cols.get('close', 'Close')
    h_col = df_cols.get('high', 'High')
    l_col = df_cols.get('low', 'Low')
    o_col = df_cols.get('open', 'Open')
    v_col = df_cols.get('volume', 'Volume')

    # Basic Indicators
    df['EMA_9'] = ta.trend.ema_indicator(df[c_col], window=9)
    df['EMA_21'] = ta.trend.ema_indicator(df[c_col], window=21)
    df['RSI'] = ta.momentum.rsi(df[c_col], window=14)
    
    latest = df.iloc[-1]
    ema9, ema21, rsi = float(latest['EMA_9']), float(latest['EMA_21']), float(latest['RSI'])
    
    signal = "HOLD"
    ai_score = 0.50
    reason = "Neutral indicator zone"
    
    if ema9 > ema21 and rsi > 58:
        signal = "BUY_CALL"
        ai_score = 0.72
        reason = f"Bullish EMA Breakout + RSI {rsi:.1f}"
    elif ema9 < ema21 and rsi < 42:
        signal = "BUY_PUT"
        ai_score = 0.72
        reason = f"Bearish EMA Breakdown + RSI {rsi:.1f}"

    # VCP boost
    vcp = detect_vcp_squeeze_contraction(df)
    if vcp.get("is_vcp"):
        ai_score = min(1.0, ai_score + vcp.get("score_boost", 0.10))

    if ai_model_tuple is not None:
        try:
            from train_model_institutional import predict_calibrated_win_probability

            # BB_Width and Vol_Ratio must be computed from real data here --
            # these previously were hardcoded placeholders (0.05 / 1.2) fed to
            # the model on every single call regardless of actual market
            # conditions. Formulas mirror compute_advanced_institutional_features()
            # in train_model_institutional.py so live inference matches training.
            bb = ta.volatility.BollingerBands(df[c_col], window=20, window_dev=2)
            bb_width = float(
                (bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1])
                / bb.bollinger_mavg().iloc[-1]
            )
            vol_ma20 = df[v_col].rolling(20).mean().iloc[-1]
            vol_ratio = float(latest[v_col] / (vol_ma20 + 1e-6))

            feat_df = pd.DataFrame([{
                'ADX': float(ta.trend.adx(df[h_col], df[l_col], df[c_col], window=14).iloc[-1]),
                'DMI_Plus': float(ta.trend.adx_pos(df[h_col], df[l_col], df[c_col], window=14).iloc[-1]),
                'DMI_Minus': float(ta.trend.adx_neg(df[h_col], df[l_col], df[c_col], window=14).iloc[-1]),
                'ATR_Ratio': float(ta.volatility.average_true_range(df[h_col], df[l_col], df[c_col], window=14).iloc[-1] / latest[c_col]),
                'BB_Width': bb_width,
                'Body_Ratio': float(abs(latest[c_col] - latest[o_col]) / (latest[h_col] - latest[l_col] + 1e-6)),
                'Vol_Ratio': vol_ratio,
                'EMA_Slope': float((ema9 - ema21) / ema21)
            }])
            calib_p = predict_calibrated_win_probability(ai_model_tuple, feat_df)
            if calib_p > 0:
                ai_score = calib_p
        except Exception:
            # Previously silent (bare "except Exception: pass"), which meant
            # a broken model call would fall back to the hardcoded 0.72
            # rule-based score with zero visibility that the model didn't
            # actually run. Now logged so failures are diagnosable instead
            # of masquerading as a normal rule-based signal.
            logger.warning(
                "Institutional model scoring failed for %s; falling back to "
                "rule-based ai_score=%.2f", symbol, ai_score, exc_info=True
            )

    metrics = {
        "price": float(latest[c_col]),
        "rsi": rsi,
        "ema9": ema9,
        "ema21": ema21
    }

    return signal, ai_score, reason, metrics


def scan_all_crypto_radar_pairs(get_live_df_func, ai_model_tuple):
    """
    Scans Top 5 Crypto Radar Pairs simultaneously on every 5-minute bar close.
    Returns the HIGHEST CONFIDENCE signal pair!
    """
    best_pair = None
    best_ai_score = 0.0
    best_signal_data = None

    for symbol in CRYPTO_RADAR_PAIRS:
        try:
            df = get_live_df_func(symbol, timeframe="5m", limit=100)
            if df is None or len(df) < 20:
                continue
                
            # Evaluate Strategy Rules
            signal, ai_score, reason, extra_metrics = evaluate_single_asset_signal(df, symbol, ai_model_tuple)
            
            # Catch high confidence breakout signals!
            if signal in ['BUY_CALL', 'BUY_PUT'] and ai_score >= 0.70:
                if ai_score > best_ai_score:
                    best_ai_score = ai_score
                    best_pair = symbol
                    best_signal_data = {
                        'symbol': symbol,
                        'signal': signal,
                        'ai_score': ai_score,
                        'reason': reason,
                        'metrics': extra_metrics
                    }
        except Exception:
            continue

    return best_signal_data