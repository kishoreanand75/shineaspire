# market_recorder.py
# Persists every market scan snapshot (price, indicators, signal) to a local
# SQLite database, so historical data accumulates over time instead of being
# lost after each 5-second scan cycle. Read it back later for your own
# analysis, retraining, or review -- this file only writes and reads data,
# it doesn't interpret or trade on it.

import sqlite3
import datetime
import json
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "market_history.db")


def _get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS market_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            captured_at TEXT NOT NULL,
            asset_name TEXT NOT NULL,
            symbol TEXT,
            price REAL,
            rsi REAL,
            signal TEXT,
            extra_json TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_asset_time ON market_snapshots(asset_name, captured_at)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS candle_history (
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            candle_time TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            PRIMARY KEY (symbol, timeframe, candle_time)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_candle_history ON candle_history(symbol, timeframe, candle_time)")
    return conn


def record_candles(df, symbol: str, timeframe: str) -> None:
    """Cache OHLCV candles so a short Binance outage does not erase context."""
    if df is None or df.empty:
        return
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        return
    try:
        conn = _get_connection()
        rows = [
            (symbol, timeframe, str(index), float(row.Open), float(row.High),
             float(row.Low), float(row.Close), float(row.Volume))
            for index, row in df.iterrows()
        ]
        with conn:
            conn.executemany(
                "INSERT OR REPLACE INTO candle_history "
                "(symbol, timeframe, candle_time, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
            )
        conn.close()
    except Exception as exc:
        print(f"[market_recorder] WARNING: failed to cache candles: {exc}")


def load_candles(symbol: str, timeframe: str, limit: int = 500):
    """Load the newest cached OHLCV candles in chronological order."""
    conn = _get_connection()
    rows = conn.execute(
        "SELECT candle_time, open, high, low, close, volume FROM candle_history "
        "WHERE symbol = ? AND timeframe = ? ORDER BY candle_time DESC LIMIT ?",
        (symbol, timeframe, limit),
    ).fetchall()
    conn.close()
    if not rows:
        return None
    import pandas as pd
    data = pd.DataFrame(rows, columns=["time", "Open", "High", "Low", "Close", "Volume"])
    data["time"] = pd.to_datetime(data["time"])
    return data.set_index("time").sort_index()


def record_snapshot(all_results: list):
    """
    Saves one scan cycle's results (the same list scan_all_assets() returns)
    to the database. Call this once per loop iteration in main.py.
    Never raises -- a storage hiccup should not crash the trading loop.
    """
    if not all_results:
        return
    now_iso = datetime.datetime.now().isoformat()
    try:
        conn = _get_connection()
        with conn:
            for item in all_results:
                conn.execute(
                    "INSERT INTO market_snapshots (captured_at, asset_name, symbol, price, rsi, signal, extra_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        now_iso,
                        item.get("Name", ""),
                        item.get("Symbol", ""),
                        item.get("Price"),
                        item.get("RSI"),
                        item.get("Signal", ""),
                        json.dumps({k: v for k, v in item.items() if k not in ("Name", "Symbol", "Price", "RSI", "Signal")}, default=str),
                    ),
                )
        conn.close()
    except Exception as e:
        print(f"[market_recorder] WARNING: failed to save snapshot: {e}")


def load_history(asset_name: str = None, since: str = None, limit: int = 5000):
    """
    Reads back saved snapshots for your own analysis.
    asset_name: filter to one asset (e.g. "NIFTY50"), or None for all.
    since: ISO datetime string, e.g. "2026-08-01" -- only rows after this.
    limit: max rows returned, most recent first.
    Returns a list of dicts.
    """
    conn = _get_connection()
    query = "SELECT captured_at, asset_name, symbol, price, rsi, signal, extra_json FROM market_snapshots WHERE 1=1"
    params = []
    if asset_name:
        query += " AND asset_name = ?"
        params.append(asset_name)
    if since:
        query += " AND captured_at >= ?"
        params.append(since)
    query += " ORDER BY captured_at DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    results = []
    for row in rows:
        captured_at, name, symbol, price, rsi, signal, extra_json = row
        rec = {
            "captured_at": captured_at,
            "asset_name": name,
            "symbol": symbol,
            "price": price,
            "rsi": rsi,
            "signal": signal,
        }
        try:
            rec.update(json.loads(extra_json) if extra_json else {})
        except Exception:
            pass
        results.append(rec)
    return results


def get_snapshot_count():
    """Quick sanity check -- how many rows are stored so far."""
    conn = _get_connection()
    count = conn.execute("SELECT COUNT(*) FROM market_snapshots").fetchone()[0]
    conn.close()
    return count