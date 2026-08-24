"""Persistent trade memory used for review and conservative signal gating."""

import csv
import json
import os
import sqlite3
from datetime import datetime, timezone


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_memory.db")


def _connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recorded_at TEXT NOT NULL,
            symbol TEXT NOT NULL,
            direction TEXT,
            entry_price REAL NOT NULL,
            exit_price REAL NOT NULL,
            stop_loss REAL,
            target REAL,
            quantity REAL,
            gross_pnl REAL,
            fees REAL,
            net_pnl REAL NOT NULL,
            exit_reason TEXT,
            signal_confidence REAL,
            signal_reason TEXT,
            post_mortem TEXT,
            market_context TEXT,
            source TEXT NOT NULL,
            UNIQUE(symbol, entry_price, exit_price, recorded_at, source)
        )
        """
    )
    try:
        conn.execute("ALTER TABLE trades ADD COLUMN post_mortem TEXT")
    except sqlite3.OperationalError:
        pass
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_memory_time ON trades(recorded_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_memory_direction ON trades(direction)")
    return conn


def record_trade(trade, source="paper_broker"):
    """Persist a completed trade without allowing a storage error to stop trading."""
    try:
        with _connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO trades (
                    recorded_at, symbol, direction, entry_price, exit_price,
                    stop_loss, target, quantity, gross_pnl, fees, net_pnl,
                    exit_reason, signal_confidence, signal_reason,
                    post_mortem, market_context, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.get("exit_time", datetime.now(timezone.utc).isoformat()),
                    str(trade.get("symbol", "")),
                    trade.get("direction", trade.get("type", "")),
                    float(trade.get("entry_price", 0.0)),
                    float(trade.get("exit_price", 0.0)),
                    _number(trade.get("stop_loss")),
                    _number(trade.get("target")),
                    _number(trade.get("quantity", trade.get("qty"))),
                    _number(trade.get("gross_pnl")),
                    _number(trade.get("fees")),
                    float(trade.get("net_pnl", 0.0)),
                    str(trade.get("exit_reason", "")),
                    _number(trade.get("signal_confidence")),
                    str(trade.get("signal_reason", "")),
                    str(trade.get("post_mortem", "")),
                    json.dumps(trade.get("market_context", {}), default=str),
                    source,
                ),
            )
    except Exception as exc:
        print(f"[trade_memory] WARNING: failed to save trade: {exc}")


def _number(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def migrate_csv_once(csv_path="trades.csv"):
    """Import legacy CSV rows once so earlier trades are available to memory."""
    if not os.path.exists(csv_path):
        return 0
    imported = 0
    try:
        with open(csv_path, newline="", encoding="utf-8") as handle:
            rows = csv.DictReader(handle)
            for row in rows:
                net_value = str(row.get("Net_PnL", row.get("PnL", "0"))).replace("$", "").replace("₹", "").replace(",", "")
                trade = {
                    "exit_time": row.get("Exit_Time", row.get("Entry_Time", "")),
                    "symbol": row.get("Symbol", ""),
                    "direction": row.get("Option_Type", ""),
                    "entry_price": row.get("Entry_Price") or row.get("Premium_Entry_Price", 0),
                    "exit_price": row.get("Exit_Price") or row.get("Premium_Exit_Price", 0),
                    "stop_loss": row.get("Stop_Loss") or row.get("Premium_Stop_Loss"),
                    "target": row.get("Target") or row.get("Take_Profit") or row.get("Premium_Take_Profit"),
                    "quantity": row.get("Quantity"),
                    "net_pnl": float(net_value or 0),
                    "exit_reason": row.get("Exit_Reason", "LEGACY_CSV"),
                }
                before = count_trades()
                record_trade(trade, source="legacy_csv")
                imported += int(count_trades() > before)
    except Exception as exc:
        print(f"[trade_memory] WARNING: legacy import failed: {exc}")
    return imported


def migrate_json_once(json_path="trades.json"):
    """Import legacy JSON trade records into the same persistent memory."""
    if not os.path.exists(json_path):
        return 0
    imported = 0
    try:
        with open(json_path, encoding="utf-8") as handle:
            rows = json.load(handle)
        for row in rows if isinstance(rows, list) else []:
            trade = {
                "exit_time": row.get("date_time", ""),
                "symbol": row.get("symbol", ""),
                "direction": row.get("direction", row.get("option_type", "")),
                "entry_price": row.get("entry_price", 0),
                "exit_price": row.get("exit_price", 0),
                "quantity": row.get("quantity", 0),
                "net_pnl": row.get("net_pnl", 0),
                "exit_reason": row.get("post_mortem", "LEGACY_JSON"),
                "market_context": row.get("layers", {}),
            }
            before = count_trades()
            record_trade(trade, source="legacy_json")
            imported += int(count_trades() > before)
    except Exception as exc:
        print(f"[trade_memory] WARNING: legacy JSON import failed: {exc}")
    return imported


def recent_stats(limit=50):
    with _connection() as conn:
        rows = conn.execute(
            "SELECT direction, net_pnl FROM trades ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    pnls = [float(row["net_pnl"]) for row in rows]
    losses = sum(pnl <= 0 for pnl in pnls)
    wins = sum(pnl > 0 for pnl in pnls)
    streak = 0
    for pnl in pnls:
        if pnl <= 0:
            streak += 1
        else:
            break
    return {
        "trades": len(pnls),
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / len(pnls) * 100.0) if pnls else 0.0,
        "net_pnl": sum(pnls),
        "recent_loss_streak": streak,
    }


def should_allow_entry(direction, min_history=5):
    """Return a conservative memory-based gate for new paper entries."""
    stats = recent_stats()
    if stats["trades"] < min_history:
        return True, "Insufficient trade history for adaptive gating"
    if stats["recent_loss_streak"] >= 3:
        return False, "Three consecutive losses in memory; entry paused for review"
    with _connection() as conn:
        rows = conn.execute(
            "SELECT net_pnl FROM trades WHERE UPPER(direction) = UPPER(?) ORDER BY id DESC LIMIT 20",
            (direction,),
        ).fetchall()
    direction_pnls = [float(row["net_pnl"]) for row in rows]
    if len(direction_pnls) >= 5 and sum(direction_pnls) < 0:
        return False, f"{direction} has negative recent memory performance; entry paused"
    return True, "Memory gate passed"


def count_trades():
    with _connection() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0])


def get_weekly_summary(days=7):
    """Return completed-trade totals from the canonical SQLite memory store."""
    cutoff = datetime.now(timezone.utc).timestamp() - (days * 86400)
    with _connection() as conn:
        rows = conn.execute("SELECT recorded_at, net_pnl FROM trades ORDER BY id").fetchall()
    recent = []
    for row in rows:
        try:
            recorded = datetime.fromisoformat(str(row["recorded_at"]).replace("Z", "+00:00"))
            if recorded.tzinfo is None:
                recorded = recorded.replace(tzinfo=timezone.utc)
            if recorded.timestamp() >= cutoff:
                recent.append(float(row["net_pnl"]))
        except (TypeError, ValueError, OverflowError):
            continue
    wins = sum(value > 0 for value in recent)
    return {
        "total_trades": len(recent),
        "wins": wins,
        "losses": len(recent) - wins,
        "win_rate": round((wins / len(recent) * 100.0) if recent else 0.0, 1),
        "net_pnl": round(sum(recent), 2),
    }