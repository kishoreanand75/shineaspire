# ================================================================================
# ANTONY QUANT AI TERMINAL - 7-DAY PERMANENT WEEKLY TRADE LOGGER
# ================================================================================
import json
import os
from datetime import datetime, timezone, timedelta

TRADES_FILE = "trades.json"

def get_ist_now():
    utc_now = datetime.now(timezone.utc)
    return utc_now + timedelta(hours=5, minutes=30)

def load_trades():
    if not os.path.exists(TRADES_FILE):
        return []
    try:
        with open(TRADES_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def save_trades(trades):
    with open(TRADES_FILE, "w") as f:
        json.dump(trades, f, indent=4)

def clear_all_trades():
    """Clears trade logs only when manually requested."""
    save_trades([])

def calculate_brokerage_fees(qty, entry_price, exit_price):
    flat_brokerage = 40.0
    turnover = (entry_price + exit_price) * qty
    stt_and_taxes = turnover * 0.0005
    return round(flat_brokerage + stt_and_taxes, 2)

def record_completed_trade(symbol, strike, entry_price, exit_price, qty, status, win_loss_reason, layer_breakdown):
    trades = load_trades()
    ist_now = get_ist_now()
    
    gross_pnl = round((exit_price - entry_price) * qty if status == "WIN" else (exit_price - entry_price) * qty, 2)
    brokerage = calculate_brokerage_fees(qty, entry_price, exit_price)
    net_pnl = round(gross_pnl - brokerage, 2)
    
    trade_record = {
        "trade_id": len(trades) + 1,
        "date_time": ist_now.strftime("%Y-%m-%d %I:%M:%S %p IST"),
        "date": ist_now.strftime("%Y-%m-%d"),
        "symbol": symbol,
        "strike": strike,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "quantity": qty,
        "gross_pnl": gross_pnl,
        "brokerage_fee": brokerage,
        "net_pnl": net_pnl,
        "result": "WIN" if net_pnl > 0 else "LOSS",
        "post_mortem": win_loss_reason,
        "layers": layer_breakdown
    }
    
    trades.append(trade_record)
    save_trades(trades)
    return trade_record

def get_today_trades():
    trades = load_trades()
    today_str = get_ist_now().strftime("%Y-%m-%d")
    return [t for t in trades if t.get("date") == today_str]

def get_today_summary():
    today_trades = get_today_trades()
    total_trades = len(today_trades)
    wins = len([t for t in today_trades if t["result"] == "WIN"])
    losses = len([t for t in today_trades if t["result"] == "LOSS"])
    net_pnl = sum([t["net_pnl"] for t in today_trades])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    
    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "net_pnl": round(net_pnl, 2),
        "trades_remaining": max(0, 3 - total_trades)
    }

def get_weekly_trades(days=7):
    """Fetches all trades from the past 7 days for 1-Week evaluation."""
    trades = load_trades()
    if not trades:
        return []
    ist_now = get_ist_now()
    cutoff_date = (ist_now - timedelta(days=days)).strftime("%Y-%m-%d")
    return [t for t in trades if t.get("date", "") >= cutoff_date]

def get_weekly_summary(days=7):
    """Calculates aggregate statistics over 1-Week testing period."""
    weekly_trades = get_weekly_trades(days)
    total_trades = len(weekly_trades)
    wins = len([t for t in weekly_trades if t["result"] == "WIN"])
    losses = len([t for t in weekly_trades if t["result"] == "LOSS"])
    net_pnl = sum([t["net_pnl"] for t in weekly_trades])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    
    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "net_pnl": round(net_pnl, 2)
    }


# ================================================================================
# KILL-SWITCH: DAILY LOSS LIMIT
# ================================================================================
def check_daily_kill_switch(starting_capital, max_loss_pct, max_trades):
    """
    Checks today's trades against the daily loss limit and trade-count cap.
    Returns dict: {"halted": bool, "reason": str, "today_pnl": float, "today_trades": int}

    This is a pure read — it does not itself stop execution. The caller
    (dashboard/execution loop) must check "halted" before placing new trades.
    """
    today_summary = get_today_summary()
    today_pnl = today_summary["net_pnl"]
    today_trades = today_summary["total_trades"]

    max_loss_amount = starting_capital * (max_loss_pct / 100.0)

    if today_trades >= max_trades:
        return {
            "halted": True,
            "reason": f"Daily trade cap reached ({today_trades}/{max_trades}) — no more trades today.",
            "today_pnl": today_pnl,
            "today_trades": today_trades,
        }

    if today_pnl <= -max_loss_amount:
        return {
            "halted": True,
            "reason": (
                f"Daily loss limit hit: ₹{abs(today_pnl):,.2f} lost "
                f"(limit ₹{max_loss_amount:,.2f} = {max_loss_pct}% of capital). Trading halted for today."
            ),
            "today_pnl": today_pnl,
            "today_trades": today_trades,
        }

    return {
        "halted": False,
        "reason": "",
        "today_pnl": today_pnl,
        "today_trades": today_trades,
    }


# ================================================================================
# SIGNAL AUDIT LOG — records every signal decision (not just executed trades)
# so a WAIT or a BUY_CALL/BUY_PUT can be traced back to the exact inputs used.
# ================================================================================
SIGNAL_AUDIT_FILE = "signal_audit_log.json"
SIGNAL_HISTORY_FILE = "signal_history.json"


def load_signal_audit_log():
    if not os.path.exists(SIGNAL_AUDIT_FILE):
        return []
    try:
        with open(SIGNAL_AUDIT_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def record_signal_audit(signal_type, reason_code, breakdown, inputs: dict):
    """
    Appends one audit record per dashboard refresh capturing the exact inputs
    (VIX, PCR, heavyweight K/A, OI walls, spot) that produced this signal.
    Keeps the last 2000 records to avoid unbounded file growth.
    """
    log = load_signal_audit_log()
    ist_now = get_ist_now()

    record = {
        "timestamp": ist_now.strftime("%Y-%m-%d %I:%M:%S %p IST"),
        "signal": signal_type,
        "reason_code": reason_code,
        "breakdown": breakdown,
        "inputs": inputs,
    }
    log.append(record)
    log = log[-2000:]

    with open(SIGNAL_AUDIT_FILE, "w") as f:
        json.dump(log, f, indent=2, default=str)

    return record


def record_signal_event(symbol, candle_time, signal, confidence, reason, htf_trend="UNKNOWN"):
    """Persist one deduplicated scanner decision for weekly signal review."""
    try:
        events = []
        if os.path.exists(SIGNAL_HISTORY_FILE):
            with open(SIGNAL_HISTORY_FILE, encoding="utf-8") as handle:
                loaded = json.load(handle)
                events = loaded if isinstance(loaded, list) else []
        key = (symbol, str(candle_time))
        if any((item.get("symbol"), str(item.get("candle_time"))) == key for item in events[-100:]):
            return False
        events.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "candle_time": str(candle_time),
            "signal": signal,
            "confidence": confidence,
            "reason": reason,
            "htf_trend": htf_trend,
        })
        with open(SIGNAL_HISTORY_FILE, "w", encoding="utf-8") as handle:
            json.dump(events[-5000:], handle, indent=2, default=str)
        return True
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[trade_logger] signal history warning: {exc}")
        return False


def get_weekly_signal_summary(days=7):
    """Count deduplicated directional signals recorded during the last N days."""
    if not os.path.exists(SIGNAL_HISTORY_FILE):
        return {"signals": 0, "buy_signals": 0, "sell_signals": 0, "hold_decisions": 0}
    cutoff = datetime.now(timezone.utc).timestamp() - (days * 86400)
    try:
        with open(SIGNAL_HISTORY_FILE, encoding="utf-8") as handle:
            events = json.load(handle)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {"signals": 0, "buy_signals": 0, "sell_signals": 0, "hold_decisions": 0}
    recent = []
    for event in events if isinstance(events, list) else []:
        try:
            stamp = datetime.fromisoformat(str(event.get("timestamp", "")).replace("Z", "+00:00"))
            if stamp.timestamp() >= cutoff:
                recent.append(event.get("signal", "HOLD"))
        except (TypeError, ValueError, OverflowError):
            continue
    buys = sum(signal == "BUY_CALL" for signal in recent)
    sells = sum(signal == "BUY_PUT" for signal in recent)
    return {"signals": buys + sells, "buy_signals": buys, "sell_signals": sells,
            "hold_decisions": sum(signal == "HOLD" for signal in recent)}