import logging
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
import config

logger = logging.getLogger(__name__)


def get_ist_now():
    """Return the current UTC instant; Binance candle boundaries are UTC based."""
    return datetime.now(timezone.utc)


def fetch_btc_live_data(symbol=config.DEFAULT_SYMBOL, timeframe=config.TIMEFRAME, period=None):
    """Fetch only BTC/USDT candles from Binance's public market API."""
    if symbol.upper() != config.DEFAULT_SYMBOL:
        raise ValueError(f"Bitcoin-only feed accepts {config.DEFAULT_SYMBOL}, got {symbol}")
    if timeframe not in config.SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported Binance timeframe: {timeframe}")

    # 500 candles (vs the old 200) gives ~41+ hours on the 5m timeframe, which
    # is required for PDH/PDL and VWAP to be computed over real, complete
    # daily windows instead of a truncated ~16-hour slice.
    fetch_limit = 500
    endpoints = [
        f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval={timeframe}&limit={fetch_limit}",
        f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={timeframe}&limit={fetch_limit}",
        f"https://api1.binance.com/api/v3/klines?symbol={symbol}&interval={timeframe}&limit={fetch_limit}",
        f"https://api2.binance.com/api/v3/klines?symbol={symbol}&interval={timeframe}&limit={fetch_limit}",
    ]
    endpoint_errors = []
    for url in endpoints:
        try:
            k_res = requests.get(url, timeout=2)
            if k_res.status_code == 200:
                raw_data = k_res.json()
                cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'q_vol', 'trades', 'tb_base', 'tb_quote', 'ignore']
                df = pd.DataFrame(raw_data, columns=cols)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                df['time'] = pd.to_datetime(df['time'], unit='ms')
                df.set_index('time', inplace=True)
                try:
                    import market_recorder
                    market_recorder.record_candles(
                        df.rename(columns={col: col.title() for col in df.columns}), symbol, timeframe
                    )
                except Exception:
                    logger.warning("market_recorder.record_candles failed for %s/%s", symbol, timeframe, exc_info=True)
                return df
            endpoint_errors.append(f"{url} -> HTTP {k_res.status_code}")
        except Exception as e:
            endpoint_errors.append(f"{url} -> {type(e).__name__}: {e}")
            continue

    # Every live endpoint failed. This used to fall through to an empty
    # DataFrame with no logging at all, which was indistinguishable from a
    # genuinely quiet market to anything downstream. Log it loudly.
    logger.error(
        "All Binance endpoints failed for %s/%s. Falling back to cached candles. Errors: %s",
        symbol, timeframe, "; ".join(endpoint_errors)
    )

    try:
        import market_recorder
        cached = market_recorder.load_candles(symbol, timeframe, limit=fetch_limit)
        if cached is not None and not cached.empty:
            logger.warning(
                "Serving CACHED (stale) candles for %s/%s because live fetch failed.",
                symbol, timeframe
            )
            return cached.rename(columns={col: col.lower() for col in cached.columns})
    except Exception:
        logger.error("market_recorder.load_candles fallback also failed for %s/%s", symbol, timeframe, exc_info=True)

    logger.critical(
        "DATA FEED DOWN: no live data and no cache available for %s/%s. "
        "Returning empty DataFrame -- callers must not treat this the same as 'no signal'.",
        symbol, timeframe
    )
    return pd.DataFrame()


def fetch_btc_historical_data(symbol=config.DEFAULT_SYMBOL, timeframe=config.TRADE_TIMEFRAME, days=60):
    """Fetch paginated BTC candles for backtesting, newest data first is avoided."""
    if symbol.upper() != config.DEFAULT_SYMBOL:
        raise ValueError(f"Bitcoin-only history accepts {config.DEFAULT_SYMBOL}, got {symbol}")
    if timeframe not in config.SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported Binance timeframe: {timeframe}")

    interval_ms = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}[timeframe]
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    rows = []
    url = "https://data-api.binance.vision/api/v3/klines"

    while start_ms < end_ms:
        try:
            response = requests.get(
                url,
                params={"symbol": symbol, "interval": timeframe, "startTime": start_ms, "endTime": end_ms, "limit": 1000},
                timeout=5,
            )
            response.raise_for_status()
            batch = response.json()
        except Exception:
            break
        if not batch:
            break
        rows.extend(batch)
        next_start = int(batch[-1][0]) + interval_ms
        if next_start <= start_ms:
            break
        start_ms = next_start

    if not rows:
        return pd.DataFrame()
    cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'q_vol', 'trades', 'tb_base', 'tb_quote', 'ignore']
    df = pd.DataFrame(rows, columns=cols).drop_duplicates(subset=["time"])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    return df.set_index('time').sort_index()


def fetch_htf_trend(symbol=config.DEFAULT_SYMBOL, htf: str = "1h") -> dict:
    """
    Higher-timeframe trend check used to gate next-candle predictions on the
    lower (trade) timeframe. Uses EMA 50 on the HTF candles:
      - Close > EMA50 -> BULLISH
      - Close < EMA50 -> BEARISH
      - otherwise      -> NEUTRAL

    Returns {"trend": "BULLISH"|"BEARISH"|"NEUTRAL"|"UNKNOWN", "ema50": float|None,
             "close": float|None, "htf": htf}. Never raises -- callers treat
    "UNKNOWN" as "do not filter" rather than crashing the scan loop.
    """
    result = {"trend": "UNKNOWN", "ema50": None, "close": None, "htf": htf}
    if htf not in config.SUPPORTED_TIMEFRAMES:
        return result
    try:
        import ta as _ta  # local import: keep data_feed's top-level deps minimal
        df = fetch_btc_live_data(symbol, timeframe=htf)
        if df is None or df.empty or len(df) < 55:
            return result
        df = df.rename(columns={col: col.title() for col in df.columns})
        ema50 = _ta.trend.ema_indicator(df['Close'], window=50)
        last_close = float(df['Close'].iloc[-2])   # last CLOSED candle, not the forming one
        last_ema50 = float(ema50.iloc[-2])
        if pd.isna(last_ema50):
            return result
        trend = "BULLISH" if last_close > last_ema50 else ("BEARISH" if last_close < last_ema50 else "NEUTRAL")
        return {"trend": trend, "ema50": last_ema50, "close": last_close, "htf": htf}
    except Exception:
        return result


def fetch_free_market_context(symbol=config.DEFAULT_SYMBOL):
    """Fetch key BTC context from public, keyless endpoints only."""
    context = {
        "funding_rate": None,
        "open_interest": None,
        "order_book_imbalance": None,
        "fear_greed_value": None,
        "fear_greed_label": None,
        "news_headlines": [],
        "sources": [],
    }
    try:
        response = requests.get(
            "https://fapi.binance.com/fapi/v1/premiumIndex",
            params={"symbol": symbol}, timeout=3,
        )
        if response.ok:
            context["funding_rate"] = float(response.json().get("lastFundingRate", 0.0))
            context["sources"].append("Binance Futures funding")
    except (requests.RequestException, TypeError, ValueError):
        pass
    try:
        response = requests.get(
            "https://fapi.binance.com/fapi/v1/openInterest",
            params={"symbol": symbol}, timeout=3,
        )
        if response.ok:
            context["open_interest"] = float(response.json().get("openInterest", 0.0))
            context["sources"].append("Binance Futures open interest")
    except (requests.RequestException, TypeError, ValueError):
        pass
    try:
        response = requests.get(
            "https://api.binance.com/api/v3/depth",
            params={"symbol": symbol, "limit": 20}, timeout=3,
        )
        if response.ok:
            book = response.json()
            bid_volume = sum(float(row[1]) for row in book.get("bids", []))
            ask_volume = sum(float(row[1]) for row in book.get("asks", []))
            total_volume = bid_volume + ask_volume
            context["order_book_imbalance"] = ((bid_volume - ask_volume) / total_volume) if total_volume else 0.0
            context["sources"].append("Binance public order book")
    except (requests.RequestException, TypeError, ValueError, KeyError):
        pass
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=3)
        if response.ok:
            item = response.json().get("data", [{}])[0]
            context["fear_greed_value"] = int(item.get("value"))
            context["fear_greed_label"] = item.get("value_classification", "")
            context["sources"].append("Alternative.me Fear & Greed")
    except (requests.RequestException, TypeError, ValueError, IndexError):
        pass
    context["news_headlines"] = _fetch_public_rss_headlines()
    if context["news_headlines"]:
        context["sources"].append("Public RSS headlines")
    return context


def _fetch_public_rss_headlines(limit=5):
    """Read public RSS titles without requiring a news API key."""
    feeds = (
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
    )
    headlines = []
    for feed_url in feeds:
        try:
            response = requests.get(feed_url, timeout=3, headers={"User-Agent": "AntonyQuant/1.0"})
            if not response.ok:
                continue
            root = ET.fromstring(response.content)
            for item in root.findall(".//item"):
                title = item.findtext("title")
                if title and title.strip() not in headlines:
                    headlines.append(title.strip())
                if len(headlines) >= limit:
                    return headlines
        except (requests.RequestException, ET.ParseError):
            continue
    return headlines