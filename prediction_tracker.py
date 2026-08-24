"""Prediction-only outcome tracking; this module never places an order."""

import csv
import json
import os
from datetime import datetime, timezone


CSV_FILE = "prediction_outcomes.csv"
PENDING_FILE = "pending_predictions.json"
MAX_HOLD_MINUTES = 100
FIELDS = [
    "Prediction_ID", "Signal_Date", "Signal_Time", "Candle_Time", "Symbol", "Direction",
    "Entry_Price", "Stop_Loss", "Take_Profit", "AI_Confidence", "HTF_Trend",
    "Signal_Reason", "Status", "Outcome", "Resolved_Date", "Resolved_Time",
    "Duration_Minutes", "Outcome_Reason", "Exit_Price",
]


def _parse_time(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=None)
    except (TypeError, ValueError):
        return None


def _load_pending():
    if not os.path.exists(PENDING_FILE):
        return []
    try:
        with open(PENDING_FILE, encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, list) else []
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return []


def _save_pending(rows):
    with open(PENDING_FILE, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, default=str)


def _load_rows():
    if not os.path.exists(CSV_FILE):
        return []
    try:
        with open(CSV_FILE, newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def _save_rows(rows):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in FIELDS} for row in rows)


def ensure_csv():
    """Create or migrate the prediction CSV before the first scan."""
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        _save_rows([])
        return
    try:
        with open(CSV_FILE, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames == FIELDS:
                return
            rows = list(reader)
        _save_rows(rows)
    except (OSError, csv.Error):
        _save_rows([])


def process_scan_results(results):
    """Resolve forecasts and store every completed candle decision once."""
    ensure_csv()
    pending = _load_pending()
    rows = _load_rows()
    now = datetime.now(timezone.utc)
    by_symbol = {row.get("Symbol"): row for row in results if row.get("Symbol")}
    remaining = []

    for forecast in pending:
        market = by_symbol.get(forecast["symbol"])
        signal_time = _parse_time(forecast["candle_time"])
        market_time = _parse_time(market.get("Candle_Time")) if market else None
        if not market or not signal_time or not market_time or market_time <= signal_time:
            remaining.append(forecast)
            continue

        direction = forecast["direction"]
        entry = float(forecast["entry_price"])
        stop = float(forecast["stop_loss"])
        target = float(forecast["take_profit"])
        high = float(market["Candle_High"])
        low = float(market["Candle_Low"])
        close = float(market["Candle_Close"])
        if direction == "BUY_CALL" and low <= stop:
            outcome, reason, exit_price = "LOSS", "STOP_LOSS_HIT: price reached the downside risk level", stop
        elif direction == "BUY_PUT" and high >= stop:
            outcome, reason, exit_price = "LOSS", "STOP_LOSS_HIT: price reached the upside risk level", stop
        elif direction == "BUY_CALL" and high >= target:
            outcome, reason, exit_price = "WIN", "TAKE_PROFIT_HIT: bullish target reached", target
        elif direction == "BUY_PUT" and low <= target:
            outcome, reason, exit_price = "WIN", "TAKE_PROFIT_HIT: bearish target reached", target
        elif (market_time - signal_time).total_seconds() / 60.0 >= MAX_HOLD_MINUTES:
            favorable = close > entry if direction == "BUY_CALL" else close < entry
            outcome = "WIN" if favorable else "LOSS"
            reason = "TIME_EXPIRY: favorable close" if favorable else "TIME_EXPIRY: unfavorable close"
            exit_price = close
        else:
            remaining.append(forecast)
            continue

        for row in rows:
            if row.get("Prediction_ID") == forecast["prediction_id"]:
                row.update({
                    "Status": "RESOLVED", "Outcome": outcome,
                    "Resolved_Date": now.strftime("%Y-%m-%d"),
                    "Resolved_Time": now.strftime("%H:%M:%S UTC"),
                    "Duration_Minutes": f"{(market_time - signal_time).total_seconds() / 60.0:.2f}",
                    "Outcome_Reason": reason, "Exit_Price": f"{exit_price:.4f}",
                })

    existing_keys = {
        (item.get("symbol"), str(item.get("candle_time"))) for item in pending
    }
    existing_keys.update(
        (row.get("Symbol"), str(row.get("Candle_Time"))) for row in rows
    )
    for result in results:
        signal = result.get("Signal", "HOLD")
        key = (result.get("Symbol"), str(result.get("Candle_Time")))
        if key in existing_keys:
            continue
        prediction_id = now.strftime("%Y%m%d%H%M%S%f")
        signal_time = _parse_time(result.get("Candle_Time")) or now.replace(tzinfo=None)
        if signal == "HOLD" or result.get("Entry_Price") is None:
            rows.append({
                "Prediction_ID": prediction_id,
                "Signal_Date": signal_time.strftime("%Y-%m-%d"),
                "Signal_Time": signal_time.strftime("%H:%M:%S UTC"),
                "Candle_Time": result.get("Candle_Time", ""),
                "Symbol": result.get("Symbol", ""),
                "Direction": "HOLD",
                "Entry_Price": result.get("Candle_Close", result.get("Price", "")),
                "Stop_Loss": "", "Take_Profit": "",
                "AI_Confidence": result.get("Confidence"),
                "HTF_Trend": result.get("HTF_Trend", "UNKNOWN"),
                "Signal_Reason": result.get("Signal_Reason", "No directional setup"),
                "Status": "NO_TRADE", "Outcome": "NO_SIGNAL",
                "Resolved_Date": now.strftime("%Y-%m-%d"),
                "Resolved_Time": now.strftime("%H:%M:%S UTC"),
                "Duration_Minutes": "0.00",
                "Outcome_Reason": "NO_SIGNAL: candle was rejected or model predicted HOLD",
                "Exit_Price": result.get("Candle_Close", result.get("Price", "")),
            })
            existing_keys.add(key)
            continue
        entry = {
            "prediction_id": prediction_id, "symbol": result["Symbol"],
            "candle_time": str(result["Candle_Time"]), "direction": signal,
            "entry_price": result["Entry_Price"], "stop_loss": result["Stop_Loss"],
            "take_profit": result["Take_Profit"],
        }
        remaining.append(entry)
        rows.append({
            "Prediction_ID": prediction_id, "Signal_Date": signal_time.strftime("%Y-%m-%d"),
            "Signal_Time": signal_time.strftime("%H:%M:%S UTC"), "Candle_Time": result["Candle_Time"],
            "Symbol": result["Symbol"],
            "Direction": signal, "Entry_Price": result["Entry_Price"],
            "Stop_Loss": result["Stop_Loss"], "Take_Profit": result["Take_Profit"],
            "AI_Confidence": result.get("Confidence"), "HTF_Trend": result.get("HTF_Trend", "UNKNOWN"),
            "Signal_Reason": result.get("Signal_Reason", ""), "Status": "PENDING", "Outcome": "",
            "Resolved_Date": "", "Resolved_Time": "", "Duration_Minutes": "",
            "Outcome_Reason": "", "Exit_Price": "",
        })
    _save_pending(remaining)
    _save_rows(rows)
    return {"pending": len(remaining), "resolved": sum(row.get("Status") == "RESOLVED" for row in rows), "new": len(rows) - len(_load_rows()) if False else 0}


def get_summary(days=7):
    """Summarize resolved prediction outcomes from the last N days."""
    cutoff = datetime.now(timezone.utc).timestamp() - days * 86400
    rows = _load_rows()
    recent = []
    for row in rows:
        if row.get("Status") != "RESOLVED":
            continue
        stamp = _parse_time(row.get("Signal_Date", "") + "T" + row.get("Signal_Time", "").replace(" UTC", ""))
        if stamp and stamp.replace(tzinfo=timezone.utc).timestamp() >= cutoff:
            recent.append(row)
    wins = sum(row.get("Outcome") == "WIN" for row in recent)
    return {
        "predictions": len(recent), "wins": wins, "losses": len(recent) - wins,
        "win_rate": round(wins / len(recent) * 100.0, 1) if recent else 0.0,
        "pending": sum(row.get("Status") == "PENDING" for row in rows),
    }