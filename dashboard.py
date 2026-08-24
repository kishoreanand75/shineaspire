# ================================================================================
# ANTONY QUANT AI TERMINAL - DASHBOARD (BITCOIN-ONLY EDITION)
#
# By user request: NIFTY and FOREX removed from the UI entirely. The NIFTY
# signal-engine code (signal_engine.py, backtester.py, multi_strategy.py etc.)
# still exists in this repo untouched -- it's just not wired into this page
# anymore. If NIFTY needs to come back later, it's a re-wiring job, not a
# rewrite, since none of that logic was deleted.
#
# HONEST STATE: same as before -- there is still no real trading
# strategy/signal engine for Bitcoin in this codebase. This page shows a real
# live price ticker and a real 15-minute candle countdown (both confirmed
# working, ported from the old dashboard), plus a live BTC candlestick chart.
# It does not generate buy/sell signals or place any trades.
# ================================================================================
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import json
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timezone, timedelta
import requests

import config
import data_feed
import trade_logger
import trade_memory
import prediction_tracker
prediction_tracker.ensure_csv()
import broker_integrator
import multi_strategy
from signal_engine import build_features, compute_atr_risk_levels, decide_from_row, generate_signal


def load_active_positions():
    """Read paper positions without allowing a damaged state file to break UI."""
    try:
        with open("active_trade.json", encoding="utf-8") as handle:
            state = json.load(handle)
        return state.get("positions", {}) if state.get("status") == "ACTIVE" else {}
    except (OSError, TypeError, ValueError):
        return {}


def normalize_trade_rows(trades):
    """Repair rows written by the older date-inclusive CSV schema."""
    if trades.empty:
        return trades
    trades = trades.astype(object)
    malformed = (
        trades.get("Entry_Candle_Time", pd.Series(index=trades.index, dtype="object"))
        .astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$")
        & trades.get("Exit_Time", pd.Series(index=trades.index, dtype="object"))
        .astype(str).str.match(r"^\d+(\.\d+)?$")
        & trades.get("Exit_Candle_Time", pd.Series(index=trades.index, dtype="object"))
        .astype(str).str.startswith("BTCUSDT_OPT_")
    )
    for index in trades.index[malformed]:
        row = trades.loc[index].to_dict()
        normalized = {
            "Trade_ID": row.get("Trade_ID", ""),
            "Entry_Time": row.get("Entry_Time", ""),
            "Entry_Candle_Time": "",
            "Exit_Time": "",
            "Exit_Candle_Time": "",
            "Duration_Minutes": row.get("Exit_Time", ""),
            "Symbol": row.get("Exit_Candle_Time", ""),
            "Direction": row.get("Duration_Minutes", ""),
            "Entry_Price": row.get("Symbol", ""),
            "Exit_Price": row.get("Direction", ""),
            "Stop_Loss": row.get("Entry_Price", ""),
            "Take_Profit": row.get("Exit_Price", ""),
            "Premium_Entry_Price": row.get("Stop_Loss", ""),
            "Premium_Exit_Price": row.get("Take_Profit", ""),
            "Premium_Stop_Loss": row.get("Premium_Entry_Price", ""),
            "Premium_Take_Profit": row.get("Premium_Exit_Price", ""),
            "Quantity": row.get("Premium_Stop_Loss", ""),
            "Exit_Reason": row.get("Premium_Take_Profit", ""),
            "Outcome": row.get("Quantity", ""),
            "Gross_PnL": row.get("Exit_Reason", ""),
            "Brokerage_Taxes": row.get("Outcome", ""),
            "Net_PnL": row.get("Gross_PnL", ""),
            "Capital_Balance": row.get("Brokerage_Taxes", ""),
            "AI_Confidence": row.get("Net_PnL", ""),
            "Signal_Reason": row.get("Capital_Balance", ""),
            "Post_Mortem": row.get("AI_Confidence", ""),
            "Market_Context": row.get("Signal_Reason", ""),
        }
        for column, value in normalized.items():
            trades.at[index, column] = str(value) if value is not None else ""
    return trades


def display_candle_time(value, fallback_time=""):
    """Show candle timestamps in the same IST timezone as trade timestamps."""
    candidate = value
    if candidate in (None, "", "nan", "NaT") and fallback_time not in (None, "", "nan", "NaT"):
        local_time = pd.to_datetime(fallback_time, errors="coerce")
        if pd.notna(local_time):
            candidate = local_time.floor("5min") - pd.Timedelta(minutes=5)
            return candidate.strftime("%Y-%m-%d %H:%M:%S IST")
    parsed = pd.to_datetime(candidate, errors="coerce", utc=True)
    if pd.isna(parsed):
        return "--"
    return parsed.tz_convert("Asia/Kolkata").strftime("%Y-%m-%d %H:%M:%S IST")


def display_time(value):
    """Convert an exchange timestamp to the dashboard's IST display timezone."""
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return "--"
    return parsed.tz_convert("Asia/Kolkata").strftime("%d-%b %H:%M IST")

# ================================================================================
# PAGE SETUP
# ================================================================================
st.set_page_config(
    page_title="ANTONY Quant AI - Bitcoin Terminal",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================================
# AUTO-REFRESH -- rerun shortly after the next Binance 5-minute boundary. This
# keeps the full page stable between candles and avoids the dimmed overlay that
# frequent full Streamlit reruns cause.
# ================================================================================
_refresh_now = datetime.now(timezone.utc)
_seconds_to_next_candle = 300 - (_refresh_now.minute % 5) * 60 - _refresh_now.second
if _seconds_to_next_candle <= 0:
    _seconds_to_next_candle = 300
st_autorefresh(interval=5000, key="live_data_autorefresh")

# ================================================================================
# TRADING TERMINAL THEME (CSS) -- unchanged from the NIFTY version
# ================================================================================
st.markdown("""
<style>
    .stApp { background-color: #0B0E11; }
    /* Keep the live terminal visually stable while Streamlit reruns. */
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main,
    [data-testid="stAppViewContainer"] [data-testid="stMainBlockContainer"] {
        opacity: 1 !important;
        filter: none !important;
    }
    [data-testid="stStatusWidget"] { display: none !important; }
    section[data-testid="stSidebar"] { background-color: #10141A; border-right: 1px solid #1E2530; }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

    .term-bar {
        display: flex; justify-content: space-between; align-items: center;
        background: linear-gradient(90deg, #10141A 0%, #151b24 100%);
        border: 1px solid #1E2530; border-radius: 10px;
        padding: 14px 22px; margin-bottom: 14px;
    }
    .term-title { font-size: 20px; font-weight: 700; color: #E6EAF0; letter-spacing: 0.3px; }
    .term-sub { font-size: 12px; color: #6B7684; margin-top: 2px; }

    .pill-open {
        display:inline-block; background-color: #0F2B22; border: 1px solid #17C964;
        color: #17C964; padding: 5px 14px; border-radius: 20px; font-size: 12px; font-weight: 700;
        letter-spacing: 0.5px;
    }

    .ticker-strip { display: flex; gap: 12px; margin-bottom: 16px; }
    .ticker-chip {
        flex: 1; background-color: #10141A; border: 1px solid #1E2530; border-radius: 10px;
        padding: 12px 16px;
    }
    .ticker-label { font-size: 11px; color: #6B7684; text-transform: uppercase; letter-spacing: 0.6px; }
    .ticker-value { font-size: 20px; font-weight: 700; color: #E6EAF0; margin-top: 2px; }
    .ticker-delta-up { color: #17C964; font-size: 13px; font-weight: 600; }
    .ticker-delta-down { color: #F5455C; font-size: 13px; font-weight: 600; }

    .sec-label {
        font-size: 13px; font-weight: 700; color: #6B7684; text-transform: uppercase;
        letter-spacing: 1px; margin: 22px 0 8px 0; border-bottom: 1px solid #1E2530; padding-bottom: 8px;
    }

    div[data-testid="stMetric"] {
        background-color: #10141A; border: 1px solid #1E2530; border-radius: 10px; padding: 10px 14px;
    }
    div[data-testid="stMetricLabel"] { color: #6B7684; }
    div[data-testid="stDataFrame"] { border: 1px solid #1E2530; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ================================================================================
# TOP TERMINAL BAR -- Bitcoin is 24/7, always "live"
# ================================================================================
st.markdown("""
<div class="term-bar">
    <div>
        <div class="term-title">🎯 ANTONY QUANT AI — BITCOIN (BTC/USDT) TERMINAL</div>
        <div class="term-sub">Live Price Ticker &nbsp;·&nbsp; Selected Candle Countdown</div>
    </div>
    <div><span class="pill-open">🟢 BITCOIN 24/7 MARKET LIVE</span></div>
</div>
""", unsafe_allow_html=True)

_, download_col = st.columns([7, 1])
with download_col:
    if Path(prediction_tracker.CSV_FILE).exists():
        st.download_button(
            "⬇️ Download CSV", data=Path(prediction_tracker.CSV_FILE).read_bytes(),
            file_name="antony_prediction_outcomes.csv", mime="text/csv",
            width="stretch", key="top_trade_csv_download",
        )

# ================================================================================
# LIVE CLOCK + 15M CANDLE COUNTDOWN + BTC LIVE TICKER
# Pure JavaScript (setInterval + Binance WebSocket) -- ticks on its own every
# second in the browser, independent of Streamlit reruns. Ported verbatim
# from the old dashboard (confirmed working there).
# ================================================================================
CHART_TIMEFRAMES = list(config.SUPPORTED_TIMEFRAMES)
selected_tf = st.radio(
    "Timeframe", CHART_TIMEFRAMES, index=CHART_TIMEFRAMES.index(config.TIMEFRAME),
    horizontal=True, key="btc_chart_tf", label_visibility="collapsed",
)

st.components.v1.html("""
<div style="background-color: #111827; border: 1px solid #374151; padding: 12px; border-radius: 10px; text-align: center; font-family: monospace; color: #F3F4F6;">
    <span id="live-date" style="color: #60A5FA; font-size: 14px; font-weight: bold;"></span> &nbsp;|&nbsp;
    <span id="live-clock" style="color: #FBBF24; font-size: 16px; font-weight: bold;"></span><br>
    <span id="candle-timer" style="color: #FFD54F; font-size: 16px; font-weight: bold;">⏳ 15M CANDLE: Loading...</span> &nbsp;|&nbsp;
    <span style="color:#00E676; font-weight:bold;">⚡ BTC LIVE TICKER: </span>
    <span id="btc-ticker-price" style="color: #00E676; font-size: 20px; font-weight: bold;">$Loading...</span>
</div>
<script>
const selectedTimeframe = '""" + selected_tf + """';
const timeframeMs = {'1m': 60000, '5m': 300000, '15m': 900000, '1h': 3600000, '4h': 14400000, '1d': 86400000}[selectedTimeframe];
let serverClockOffsetMs = 0;

async function syncBinanceClock() {
    try {
        const response = await fetch('https://api.binance.com/api/v3/time', {cache: 'no-store'});
        const payload = await response.json();
        serverClockOffsetMs = Number(payload.serverTime) - Date.now();
    } catch (error) {
        serverClockOffsetMs = 0;
    }
}

function updateClockAndCandleTimer() {
    const now = new Date(Date.now() + serverClockOffsetMs);
    document.getElementById('live-date').innerText = '📅 ' + now.toLocaleDateString('en-IN', { timeZone: 'Asia/Kolkata', weekday: 'short', day: '2-digit', month: 'short', year: 'numeric' });
    document.getElementById('live-clock').innerText = '⏰ ' + now.toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true }) + ' IST';

    const remSec = Math.max(0, Math.ceil((timeframeMs - (now.getTime() % timeframeMs)) / 1000));
    const remMin = Math.floor(remSec / 60);
    const remS = remSec % 60;

    const minStr = String(remMin).padStart(2, '0');
    const secStr = String(remS).padStart(2, '0');

    const timerElem = document.getElementById('candle-timer');
    if (remSec <= 60) {
        timerElem.style.color = '#FF5252';
        timerElem.innerText = '⚠️ GET READY FOR NEXT CANDLE ENTRY (' + minStr + ':' + secStr + ' REMAINING)';
    } else {
        timerElem.style.color = '#FFD54F';
        timerElem.innerText = '⏳ ' + selectedTimeframe.toUpperCase() + ' CANDLE COUNTDOWN: ' + minStr + ':' + secStr + ' REMAINING';
    }
}
setInterval(updateClockAndCandleTimer, 1000); updateClockAndCandleTimer();
syncBinanceClock(); setInterval(syncBinanceClock, 60000);

const ws = new WebSocket('wss://stream.binance.com:9443/ws/btcusdt@ticker');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    const price = parseFloat(data.c).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
    document.getElementById('btc-ticker-price').innerText = '$' + price;
};
</script>
""", height=85)

# ================================================================================
# FETCH BTC LIVE DATA (server-side, for the spot metric + candlestick chart)
# ================================================================================
df_btc = data_feed.fetch_btc_live_data(config.DEFAULT_SYMBOL, selected_tf)
trade_df_btc = (
    df_btc
    if selected_tf == config.TRADE_TIMEFRAME
    else data_feed.fetch_btc_live_data(config.DEFAULT_SYMBOL, config.TRADE_TIMEFRAME)
)
chart_df_source = df_btc

if df_btc.empty or len(df_btc) < 5 or trade_df_btc.empty or len(trade_df_btc) < 30:
    st.warning("⏳ Connecting to Bitcoin Live Feed... Please wait a few seconds and it will retry on the next auto-refresh.")
else:
    spot_price = float(df_btc.iloc[-1]['close'])
    prev_close = float(df_btc.iloc[-2]['close'])
    spot_change = spot_price - prev_close
    spot_change_pct = (spot_change / prev_close * 100) if prev_close else 0.0
    spot_delta_cls = "ticker-delta-up" if spot_change >= 0 else "ticker-delta-down"

    st.markdown(f"""
<div class="ticker-strip">
    <div class="ticker-chip">
        <div class="ticker-label">BTC/USDT Spot</div>
        <div class="ticker-value">${spot_price:,.2f}</div>
        <div class="{spot_delta_cls}">{spot_change:+,.2f} ({spot_change_pct:+.2f}%)</div>
    </div>
</div>
""", unsafe_allow_html=True)

    signal_df = trade_df_btc.rename(columns={column: column.title() for column in trade_df_btc.columns})
    trade_now = datetime.now(timezone.utc)
    # Use the server's UTC clock for the boundary countdown. A delayed API
    # response must not freeze the countdown on an already-closed candle.
    current_candle_start = pd.Timestamp.now(tz="UTC").floor("5min").to_pydatetime()
    current_candle_end = current_candle_start + timedelta(minutes=5)
    seconds_to_close = max(0, int((current_candle_end - trade_now).total_seconds()))

    # Always forecast the next candle from completed candles only. The current
    # forming candle is displayed on the chart but must not move the forecast.
    signal_input = signal_df.iloc[:-1]
    signal_result = generate_signal(signal_input, asset_symbol=config.DEFAULT_SYMBOL, model=multi_strategy.model)
    signal_name = signal_result.get("signal", "HOLD")
    signal_candle_time = display_time(signal_df.index[-2])
    signal_entry_price = float(signal_df.iloc[-2]["Close"])
    confidence = signal_result.get("confidence")
    high_confidence_forecast = (
        signal_name in ("BUY_CALL", "BUY_PUT")
        and confidence is not None
        and float(confidence) >= (
            config.PAPER_MIN_ACTIONABLE_CONFIDENCE
            if config.PAPER_TRADING_MODE else config.MIN_ACTIONABLE_CONFIDENCE
        )
    )
    probability_text = ""
    class_probabilities = {0: None, 1: None, 2: None}
    if multi_strategy.model is not None:
        try:
            model_features = build_features(signal_input).iloc[-1]
            feature_values = pd.DataFrame([{column: model_features[column] for column in multi_strategy.model.feature_names_in_}]).fillna(0)
            probabilities = multi_strategy.model.predict_proba(feature_values)[0]
            class_probabilities = {int(label): float(probability) for label, probability in zip(multi_strategy.model.classes_, probabilities)}
            probability_text = (
                f"Next-candle probabilities: PUT {class_probabilities.get(0, 0.0):.1%} · "
                f"HOLD {class_probabilities.get(1, 0.0):.1%} · CALL {class_probabilities.get(2, 0.0):.1%}"
            )
        except Exception:
            probability_text = "Next-candle probabilities unavailable"
    if confidence is None and probability_text and any(value is not None for value in class_probabilities.values()):
        confidence = max(value for value in class_probabilities.values() if value is not None)
    if not high_confidence_forecast:
        signal_result = dict(signal_result)
        signal_result["signal"] = "HOLD"
        if signal_name in ("BUY_CALL", "BUY_PUT") and confidence is not None:
            signal_result["reason"] = f"Directional confidence {float(confidence):.1%} is below the {config.MIN_ACTIONABLE_CONFIDENCE:.0%} alert threshold"
        else:
            display_threshold = config.PAPER_MIN_ACTIONABLE_CONFIDENCE if config.PAPER_TRADING_MODE else config.MIN_ACTIONABLE_CONFIDENCE
            signal_result["reason"] = f"No {display_threshold:.0%}+ directional forecast for the next candle"
        signal_name = "HOLD"
    last_confirmed = None
    if signal_name == "HOLD":
        signal_features = build_features(signal_df)
        historical_htf_trend = signal_result.get("htf_trend", "UNKNOWN")
        for candle_index in range(len(signal_features) - 2, max(24, len(signal_features) - 51), -1):
            historical_result = decide_from_row(
                signal_features, candle_index, asset_symbol=config.DEFAULT_SYMBOL,
                model=multi_strategy.model, htf_trend=historical_htf_trend,
            )
            if historical_result.get("signal") in ("BUY_CALL", "BUY_PUT"):
                last_confirmed = (signal_features.index[candle_index], historical_result)
                break
    signal_features = build_features(signal_input)
    signal_row = signal_features.iloc[-1]
    atr_value = float(signal_row.get("ATR", 0) or 0)
    if not pd.notna(atr_value) or atr_value <= 0:
        atr_value = float(signal_df["High"].sub(signal_df["Low"]).rolling(14).mean().iloc[-2])
    if not pd.notna(atr_value) or atr_value <= 0:
        atr_value = spot_price * 0.01

    if signal_name == "BUY_CALL":
        signal_label, signal_color = "LONG / BUY BTC", "#17C964"
        risk_levels = compute_atr_risk_levels(signal_row, "BUY_CALL")
        stop_price = risk_levels["stop_loss"]
        target_price = risk_levels["take_profit"]
    elif signal_name == "BUY_PUT":
        signal_label, signal_color = "SHORT / SELL BTC", "#F5455C"
        risk_levels = compute_atr_risk_levels(signal_row, "BUY_PUT")
        stop_price = risk_levels["stop_loss"]
        target_price = risk_levels["take_profit"]
    else:
        signal_label, signal_color = "WAIT / NO TRADE", "#FBBF24"
        # Keep reference levels visible even when the confidence gate says
        # WAIT. They are informational only and must not be treated as an entry.
        reference_direction = (
            "BUY_PUT" if class_probabilities.get(0, 0.0) > class_probabilities.get(2, 0.0)
            else "BUY_CALL"
        )
        risk_levels = compute_atr_risk_levels(signal_row, reference_direction)
        stop_price = risk_levels["stop_loss"]
        target_price = risk_levels["take_profit"]

    if confidence is None:
        confidence_text = "No model score"
    elif high_confidence_forecast:
        confidence_text = f"{float(confidence):.1%} directional"
    elif (
        class_probabilities.get(1) is not None
        and float(class_probabilities.get(1) or 0.0) >= max(
            float(class_probabilities.get(0) or 0.0),
            float(class_probabilities.get(2) or 0.0),
        )
    ):
        confidence_text = f"{float(confidence):.1%} HOLD-dominant"
    else:
        confidence_text = f"{float(confidence):.1%} directional (below 55%)"
    entry_label = "Indicative Next Entry" if signal_name != "HOLD" else "Latest Closed Price"
    entry_text = f"${signal_entry_price:,.2f}"
    stop_text = f"${stop_price:,.2f}" if stop_price is not None else "N/A"
    target_text = f"${target_price:,.2f}" if target_price is not None else "N/A"
    # The forecast is made from the last closed candle and applies to the
    # candle that is forming now, not the candle after it.
    next_candle_start = current_candle_start
    exit_text = display_time(next_candle_start + timedelta(minutes=20))
    timing_text = f"Next candle starts in {seconds_to_close // 60:02d}:{seconds_to_close % 60:02d} · forecast uses completed candles only"
    active_positions = load_active_positions()
    memory_stats = trade_memory.recent_stats()
    planned_direction = "CALL" if signal_name == "BUY_CALL" else "PUT" if signal_name == "BUY_PUT" else "NONE"
    memory_allowed, memory_reason = trade_memory.should_allow_entry(planned_direction) if planned_direction != "NONE" else (False, "No directional signal")
    if active_positions:
        decision_title = "HOLD CURRENT TRADE"
        decision_reason = "An active paper trade is already open; wait for its Target, Stop Loss, or time exit."
        decision_color = "#FBBF24"
    elif high_confidence_forecast and memory_allowed:
        decision_title = "TRADE NEXT CANDLE"
        display_threshold = config.PAPER_MIN_ACTIONABLE_CONFIDENCE if config.PAPER_TRADING_MODE else config.MIN_ACTIONABLE_CONFIDENCE
        decision_reason = f"{display_threshold:.0%}+ directional forecast and memory risk gate passed."
        decision_color = "#17C964"
    else:
        decision_title = "DO NOT TRADE NEXT CANDLE"
        decision_reason = memory_reason if not memory_allowed else signal_result.get("reason", "No qualifying setup")
        decision_color = "#F5455C"

    if active_positions:
        live_thought = "I can see an open paper trade, so I am monitoring its price, stop loss, target, and timeout instead of opening another trade."
    elif signal_name in ("BUY_CALL", "BUY_PUT") and high_confidence_forecast and memory_allowed:
        display_threshold = config.PAPER_MIN_ACTIONABLE_CONFIDENCE if config.PAPER_TRADING_MODE else config.MIN_ACTIONABLE_CONFIDENCE
        live_thought = f"I checked the completed candles and found a {planned_direction} setup. Confidence cleared {display_threshold:.0%}, and my trade memory has not blocked this direction, so I am preparing the next-candle plan."
    elif confidence is not None and signal_name == "HOLD":
        live_thought = "I checked trend, candle strength, volatility, model probabilities, and trade history. The evidence is mixed, so I am waiting instead of forcing a trade."
    else:
        live_thought = "I am waiting for enough completed-candle evidence and a clear directional setup before suggesting a trade."
    live_thought += f" Last analysis update: {trade_now.strftime('%H:%M:%S UTC')}."

    signal_col, analysis_col = st.columns([3, 2])
    with signal_col:
        st.markdown(f"""
<div style="background:#10141A;border:1px solid #1E2530;border-left:4px solid {signal_color};border-radius:10px;padding:16px;margin:12px 0 18px 0;">
    <div style="color:#6B7684;font-size:11px;text-transform:uppercase;letter-spacing:1px;">BTC PAPER SIGNAL · {selected_tf.upper()}</div>
    <div style="color:{signal_color};font-size:22px;font-weight:700;margin:5px 0 12px 0;">{signal_label}</div>
    <div style="display:flex;gap:28px;flex-wrap:wrap;color:#E6EAF0;font-size:14px;">
        <span>{entry_label}: <b>{entry_text}</b></span>
        <span>Stop Loss: <b>{stop_text}</b></span>
        <span>Target: <b>{target_text}</b></span>
        <span>Exit By: <b>{exit_text}</b></span>
        <span>Confidence: <b style="color:#ffe400;">{confidence_text}</b></span>
    </div>
    <div style="color:#9AA4B2;font-size:12px;margin-top:10px;">Fixed trade timeframe: {config.TRADE_TIMEFRAME.upper()} · Signal candle: {signal_candle_time} · {timing_text}</div>
    <div style="color:#9AA4B2;font-size:12px;margin-top:4px;">{signal_result.get("reason", "Waiting for a confirmed setup")}</div>
    <div style="color:#9AA4B2;font-size:12px;margin-top:4px;">{probability_text}</div>
    <div style="color:#9AA4B2;font-size:12px;margin-top:4px;">Exit rule: close at Target or Stop Loss; otherwise time-based exit by {exit_text}.</div>
</div>
""", unsafe_allow_html=True)
    with analysis_col:
        st.markdown(f"""
<div style="background:#121821;border:1px solid #2A3542;border-top:3px solid {decision_color};border-radius:10px;padding:16px;margin:12px 0 18px 0;min-height:190px;">
    <div style="color:#8FA0B2;font-size:11px;text-transform:uppercase;letter-spacing:1px;">BOT DECISION / LIVE ANALYSIS</div>
    <div style="color:{decision_color};font-size:20px;font-weight:700;margin:6px 0 10px 0;">{decision_title}</div>
    <div style="color:#E6EAF0;font-size:13px;line-height:1.55;">{decision_reason}</div>
    <div style="color:#AAB6C3;font-size:12px;margin-top:9px;">Signal: <b>{signal_name}</b> · Direction: <b>{planned_direction}</b></div>
    <div style="color:#AAB6C3;font-size:12px;">Confidence: <b>{confidence_text}</b></div>
    <div style="color:#AAB6C3;font-size:12px;">Memory: <b>{memory_stats['trades']} trades</b> · recent win rate <b>{memory_stats['win_rate']:.1f}%</b> · loss streak <b>{memory_stats['recent_loss_streak']}</b></div>
    <div style="color:#AAB6C3;font-size:12px;margin-top:5px;">Analysing: completed candles, ATR, trend filters, model probability, and active-trade state.</div>
    <div style="color:#E6EAF0;font-size:13px;line-height:1.5;margin-top:10px;padding-top:9px;border-top:1px solid #2A3542;"><b>Bot says:</b> {live_thought}</div>
</div>
""", unsafe_allow_html=True)
    if last_confirmed:
        historical_time, historical_result = last_confirmed
        historical_direction = "LONG / BUY BTC" if historical_result["signal"] == "BUY_CALL" else "SHORT / SELL BTC"
        st.caption(f"Last confirmed paper signal: {historical_direction} at {display_time(historical_time)}. Current candle is still WAIT; do not enter late.")

    # ============================================================================
    # BTC CANDLESTICK CHART -- timeframe selector + smooth drag/zoom, no reset
    # on autorefresh (uirevision keeps your current pan/zoom position fixed
    # across periodic data refreshes instead of snapping back each time).
    # ============================================================================
    tf_label_col, _ = st.columns([3, 5])
    with tf_label_col:
        st.markdown("<div class='sec-label' style='margin-bottom:4px;'>BITCOIN (BTC/USDT) CHART</div>", unsafe_allow_html=True)
    if not chart_df_source.empty and len(chart_df_source) >= 5:
        chart_df = chart_df_source.tail(200)
        fig = go.Figure(data=[go.Candlestick(
            x=chart_df.index,
            open=chart_df['open'], high=chart_df['high'],
            low=chart_df['low'], close=chart_df['close'],
            increasing_line_color='#17C964', decreasing_line_color='#F5455C',
        )])
        fig.update_layout(
            height=520, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='#0B0E11', plot_bgcolor='#0B0E11',
            font=dict(color='#C9D1DB'),
            xaxis=dict(gridcolor='#1E2530', rangeslider=dict(visible=False)),
            yaxis=dict(gridcolor='#1E2530'),
            dragmode='pan',
            # uirevision: same value across reruns = plotly preserves the
            # viewer's current zoom/pan instead of resetting it every time
            # new data comes in via autorefresh. Change this value only if
            # you want a fresh/reset view (e.g. on timeframe switch, which
            # already happens naturally since selected_tf changes the key).
            uirevision=f"btc-chart-{selected_tf}",
        )
        st.plotly_chart(
            fig, width="stretch",
            config={
                "scrollZoom": True,        # mouse wheel / trackpad zoom
                "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            },
        )
    else:
        st.warning(f"⏳ Loading {selected_tf} candles...")

st.info(
    f"ℹ️ Paper mode only. Trade timeframe is fixed at **{config.TRADE_TIMEFRAME}**; "
    "signals use closed candles and entries are allowed only during minutes 1-4 of the next candle."
)

# ==============================================================================
# ONE ACTIONABLE SIGNAL TABLE
# ===============================================================================
fixed_starting_capital = float(getattr(config, "BTC_START_CAPITAL_USD", 20.00))
risk_budget = fixed_starting_capital * float(getattr(config, "RISK_PER_TRADE_PCT", 0.005))
display_threshold = config.PAPER_MIN_ACTIONABLE_CONFIDENCE if config.PAPER_TRADING_MODE else config.MIN_ACTIONABLE_CONFIDENCE
signal_direction = {"BUY_CALL": "CALL", "BUY_PUT": "PUT", "HOLD": "HOLD"}.get(signal_name, "HOLD")
if signal_name == "HOLD":
    signal_direction = (
        "WATCH PUT" if class_probabilities.get(0, 0.0) > class_probabilities.get(2, 0.0)
        else "WATCH CALL"
    )
signal_status = "TRADE NEXT CANDLE" if decision_title == "TRADE NEXT CANDLE" else "WAIT"
if signal_name == "HOLD" and confidence is not None:
    signal_status = f"WAIT ({float(confidence):.1%} < {display_threshold:.0%})"
current_row = {
    "Time": f"NEXT: {display_time(next_candle_start)}",
    "Entry Time": display_time(next_candle_start),
    "Direction": signal_direction,
    "Win Probability": f"{float(confidence):.1%}" if confidence is not None else "--",
    "Entry": f"${signal_entry_price:,.2f}",
    "Stop Loss": f"${stop_price:,.2f}" if stop_price is not None else "--",
    "Target": f"${target_price:,.2f}" if target_price is not None else "--",
    "Exit By": exit_text,
    "Amount to Risk": f"${risk_budget:,.2f}",
    "Status": signal_status,
}

signal_rows = [current_row]
if active_positions:
    active_signal_rows = []
    for active_symbol, position in active_positions.items():
        active_signal_rows.append({
            "Time": display_time(position.get("entry_candle_time") or position.get("entry_time")),
            "Entry Time": display_time(position.get("entry_time")),
            "Direction": position.get("type", "--"),
            "Win Probability": (
                f"{float(position['signal_confidence']):.1%}"
                if position.get("signal_confidence") not in (None, "") else "--"
            ),
            "Entry": f"${float(position.get('entry_stock_price', 0)):,.2f}",
            "Stop Loss": f"${float(position.get('sl_stock_price', position.get('stop_loss', 0))):,.2f}",
            "Target": f"${float(position.get('target_stock_price', position.get('target', 0))):,.2f}",
            "Exit By": display_time(
                pd.to_datetime(position.get("entry_time"), errors="coerce", utc=True)
                + pd.Timedelta(minutes=20)
            ),
            "Amount to Risk": f"${risk_budget:,.2f}",
            "Status": "ACTIVE - WAIT FOR EXIT",
        })
    signal_rows = active_signal_rows + signal_rows

st.markdown("<div class='sec-label'>NEXT CANDLE SIGNAL</div>", unsafe_allow_html=True)
st.caption(
    f"One row is the current recommendation. Amount to Risk = ${fixed_starting_capital:,.2f} capital × "
    f"{config.RISK_PER_TRADE_PCT:.2%} risk per trade = ${risk_budget:,.2f} maximum planned loss. "
    f"Only {display_threshold:.0%}+ directional signals can become TRADE NEXT CANDLE. "
    "Confidence is a model score, not a guaranteed win rate. All times are IST."
)

signal_display = pd.DataFrame(signal_rows)
st.dataframe(
    signal_display,
    width="stretch",
    hide_index=True,
    key=f"next_signal_table_{signal_candle_time}_{signal_direction}",
)

with st.expander("Recent trade outcomes", expanded=False):
    history_rows = []
    prediction_path = Path(prediction_tracker.CSV_FILE)
    if prediction_path.exists() and prediction_path.stat().st_size:
        try:
            history_df = pd.read_csv(prediction_path, dtype=str)
            history_df["_sort_time"] = pd.to_datetime(
                history_df.get("Candle_Time"), errors="coerce", utc=True
            )
            history_df = history_df.sort_values(
                "_sort_time", ascending=False, na_position="last"
            )
            for _, row in history_df.iterrows():
                direction = str(row.get("Direction", "")).strip()
                if direction not in {"BUY_CALL", "BUY_PUT"}:
                    continue

                try:
                    entry_value = float(row.get("Entry_Price", ""))
                    stop_value = float(row.get("Stop_Loss", ""))
                    target_value = float(row.get("Take_Profit", ""))
                except (TypeError, ValueError):
                    continue
                if pd.isna(entry_value) or pd.isna(stop_value) or pd.isna(target_value):
                    continue

                outcome = str(row.get("Outcome") or row.get("Status") or "").upper()
                if outcome not in {"WIN", "LOSS", "PENDING"}:
                    continue

                confidence_value = pd.to_numeric(row.get("AI_Confidence"), errors="coerce")
                history_rows.append({
                    "Time": display_time(row.get("Candle_Time")),
                    "Entry Time": "Completed",
                    "Direction": "CALL" if direction == "BUY_CALL" else "PUT",
                    "Win Probability": f"{confidence_value:.1%}" if pd.notna(confidence_value) else "--",
                    "Entry": f"${entry_value:,.2f}",
                    "Stop Loss": f"${stop_value:,.2f}",
                    "Target": f"${target_value:,.2f}",
                    "Exit By": display_time(
                        f"{row.get('Resolved_Date', '')}T{row.get('Resolved_Time', '')}"
                    ) if row.get("Resolved_Date") and row.get("Resolved_Time") else "Pending",
                    "Amount to Risk": f"${risk_budget:,.2f}",
                    "Status": outcome,
                })
        except (OSError, ValueError, TypeError, pd.errors.EmptyDataError, pd.errors.ParserError):
            pass

    if history_rows:
        st.dataframe(
            pd.DataFrame(history_rows),
            width="stretch",
            hide_index=True,
            key=f"history_table_{signal_candle_time}_{signal_direction}",
        )
    else:
        st.info("No valid historical trade outcomes yet.")

# ================================================================================
# SIDEBAR -- asset-agnostic controls kept from the original dashboard
# ================================================================================
st.sidebar.markdown("### ⚙️ System Control")
st.sidebar.markdown("---")
st.sidebar.info("**Market:** BITCOIN (BTC/USDT)")
st.sidebar.info("**Symbol:** BTCUSDT")

starting_capital = fixed_starting_capital
st.sidebar.metric("Fixed BTC Paper Capital", f"${starting_capital:,.2f}")
memory_stats = trade_memory.recent_stats()
st.sidebar.caption(
    f"Trade memory: {memory_stats['trades']} stored · "
    f"{memory_stats['win_rate']:.1f}% recent win rate · "
    f"loss streak {memory_stats['recent_loss_streak']}"
)

st.sidebar.markdown("---")
if "hide_completed_trade_table" not in st.session_state:
    st.session_state["hide_completed_trade_table"] = False
if st.sidebar.button("🧹 Clear Table View", width="stretch"):
    st.session_state["hide_completed_trade_table"] = True
    st.rerun()
st.sidebar.caption("Clears only this screen view; CSV and bot memory remain safe.")

with st.sidebar.expander("🛠️ Active Trade Debug Status"):
    active_trade = None
    try:
        active_trade = trade_logger.get_today_trades()
    except Exception:
        active_trade = None
    if active_trade:
        st.write(f"Open/last trades today: {len(active_trade)}")
        st.json(active_trade[-1])
    else:
        st.info("No active BTC trade currently running.")

st.sidebar.markdown("---")
with st.sidebar.expander("🚦 Live-Trading Readiness Gate", expanded=False):
    readiness = broker_integrator.check_live_trading_readiness()
    if readiness["ready"]:
        st.success("✅ All readiness conditions met — REAL mode can be considered.")
    else:
        st.warning("🚫 REAL mode locked. Unmet conditions:")
        for r in readiness["reasons"]:
            st.markdown(f"- {r}")

# ================================================================================
# PAPER vs REAL EXECUTION MODE (dropdown, gated by the readiness check above)
# ================================================================================
st.sidebar.markdown("---")
with st.sidebar.expander("💱 Execution Mode (Paper / Real)", expanded=True):
    broker_integrator.render_broker_integrator_tab()

current_exec_mode = st.session_state.get('execution_mode', 'PAPER')
if current_exec_mode == "REAL":
    st.sidebar.error("🔴 MODE: REAL MONEY")
else:
    st.sidebar.success("🟡 MODE: PAPER (safe)")

if st.sidebar.button("🔄 Refresh", width="stretch"):
    st.rerun()