# ================================================================================
# ANTONY QUANT AI TERMINAL - LIVE HEAVYWEIGHT ALIGNMENT ENGINE
# ================================================================================
# Layer 1 of the 5-layer decision engine checks whether NIFTY's biggest index
# constituents are actually moving in the same direction as the index. Until now
# this was hardcoded (heavy_k = 4, heavy_a = 0.82) regardless of what these
# stocks were actually doing. This module fetches their REAL 15m price action.
#
# Weights approximate each stock's NIFTY 50 index weightage (source: NSE index
# factsheets, rounded). These are used to compute the alignment score (A),
# while K is simply the count of stocks agreeing with the index direction.
# ================================================================================

import yfinance as yf
import pandas as pd

# Top 5 NIFTY heavyweights by index weight, with their approximate weights.
# Ticker suffix .NS = NSE listing on Yahoo Finance.
HEAVYWEIGHTS = {
    "RELIANCE.NS": 0.28,
    "HDFCBANK.NS": 0.24,
    "ICICIBANK.NS": 0.20,
    "INFY.NS": 0.16,
    "TCS.NS": 0.12,
}


def fetch_heavyweight_alignment(nifty_direction, timeframe="15m", period="2d"):
    """
    Fetches real 15m price action for the 5 NIFTY heavyweights and checks how
    many are moving in the same direction as the index.

    Returns a dict:
        {
            "success": bool,
            "reason": str,                 # populated only if success is False
            "heavyweight_k": int,           # count aligned with nifty_direction (0-5)
            "heavyweight_a": float,         # weighted alignment score (0.0-1.0)
            "details": {ticker: "UP"/"DOWN"/"FLAT"/"ERROR", ...}
        }

    On failure (data fetch error), success=False — caller must NOT substitute
    a guessed K/A value; it should treat Layer 1 as unavailable.
    """
    details = {}
    aligned_weight = 0.0
    total_weight = 0.0
    aligned_count = 0
    fetch_errors = 0

    for ticker, weight in HEAVYWEIGHTS.items():
        try:
            df = yf.download(tickers=ticker, period=period, interval=timeframe, progress=False)
            if df.empty or len(df) < 2:
                details[ticker] = "ERROR"
                fetch_errors += 1
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]

            last_close = float(df['close'].iloc[-1])
            prev_close = float(df['close'].iloc[-2])

            if last_close > prev_close:
                stock_dir = "UP"
            elif last_close < prev_close:
                stock_dir = "DOWN"
            else:
                stock_dir = "FLAT"

            details[ticker] = stock_dir
            total_weight += weight

            if stock_dir == nifty_direction:
                aligned_weight += weight
                aligned_count += 1

        except Exception:
            details[ticker] = "ERROR"
            fetch_errors += 1
            continue

    # If more than 2 of 5 stocks failed to fetch, don't trust the reading
    if fetch_errors > 2:
        return {
            "success": False,
            "reason": f"{fetch_errors}/5 heavyweight stocks failed to fetch — data too incomplete to trust",
            "heavyweight_k": None,
            "heavyweight_a": None,
            "details": details,
        }

    heavyweight_a = round(aligned_weight / total_weight, 3) if total_weight > 0 else 0.0

    return {
        "success": True,
        "reason": "",
        "heavyweight_k": aligned_count,
        "heavyweight_a": heavyweight_a,
        "details": details,
    }