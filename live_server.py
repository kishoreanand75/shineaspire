"""No-reload BTC dashboard server: Binance polling + WebSocket broadcast."""

import asyncio
import json
import os
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import websockets
from xgboost import XGBClassifier

import config
import data_feed
import trade_memory
from signal_engine import FEATURE_COLUMNS, build_features, generate_signal

ROOT = Path(__file__).resolve().parent
HTTP_HOST = "127.0.0.1"
HTTP_PORT = 8502
WS_HOST = "127.0.0.1"
WS_PORT = 8765
MODEL_FILE = ROOT / "xgboost_model.json"
clients = set()
context_cache = {"value": {}, "expires_at": 0.0}

model = None
if MODEL_FILE.exists():
    model = XGBClassifier()
    model.load_model(str(MODEL_FILE))


class QuietHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format, *args):
        pass


def start_http_server():
    server = ThreadingHTTPServer((HTTP_HOST, HTTP_PORT), QuietHandler)
    print(f"Live dashboard: http://localhost:{HTTP_PORT}/live_dashboard.html")
    server.serve_forever()


def _probabilities(signal_input):
    if model is None:
        return {"PUT": None, "HOLD": None, "CALL": None}
    try:
        features = build_features(signal_input).iloc[-1]
        names = getattr(model, "feature_names_in_", FEATURE_COLUMNS)
        values = pd.DataFrame([{column: features[column] for column in names}]).fillna(0)
        probs = model.predict_proba(values)[0]
        result = {"PUT": 0.0, "HOLD": 0.0, "CALL": 0.0}
        for label, probability in zip(model.classes_, probs):
            result[{0: "PUT", 1: "HOLD", 2: "CALL"}.get(int(label), "HOLD")] = float(probability)
        return result
    except Exception:
        return {"PUT": None, "HOLD": None, "CALL": None}


def _active_positions():
    try:
        with open(ROOT / "active_trade.json", encoding="utf-8") as handle:
            state = json.load(handle)
        return state.get("positions", {}) if state.get("status") == "ACTIVE" else {}
    except (OSError, ValueError, TypeError):
        return {}


def get_snapshot():
    candles = data_feed.fetch_btc_live_data(config.DEFAULT_SYMBOL, config.TRADE_TIMEFRAME)
    if candles.empty or len(candles) < 30:
        return {"status": "connecting", "message": "Waiting for Binance candle data..."}

    candles = candles.rename(columns={column: column.title() for column in candles.columns})
    candles = candles.sort_index()
    closed = candles.iloc[:-1]
    current = candles.iloc[-1]
    result = generate_signal(closed, asset_symbol=config.DEFAULT_SYMBOL, model=model)
    probabilities = _probabilities(closed)
    now = time.monotonic()
    if now >= context_cache["expires_at"]:
        context_cache["value"] = data_feed.fetch_free_market_context(config.DEFAULT_SYMBOL)
        context_cache["expires_at"] = now + 60.0
    market_context = context_cache["value"]
    direction = {"BUY_CALL": "CALL", "BUY_PUT": "PUT"}.get(result.get("signal"), "NONE")
    confidence = result.get("confidence")
    high_confidence = (
        direction != "NONE" and confidence is not None
        and float(confidence) >= config.MIN_ACTIONABLE_CONFIDENCE
    )
    active = _active_positions()
    stats = trade_memory.recent_stats()
    memory_allowed, memory_reason = trade_memory.should_allow_entry(direction) if direction != "NONE" else (False, "No directional signal")

    if active:
        decision = "HOLD CURRENT TRADE"
        decision_reason = "An active paper trade is open; monitoring its exit conditions."
    elif high_confidence and memory_allowed:
        decision = "TRADE NEXT CANDLE"
        decision_reason = (
            f"Directional confidence is at least {config.MIN_ACTIONABLE_CONFIDENCE:.0%} "
            "and the memory risk gate passed."
        )
    else:
        decision = "DO NOT TRADE NEXT CANDLE"
        decision_reason = memory_reason if not memory_allowed else result.get("reason", "No qualifying setup")

    return {
        "status": "live",
        "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "symbol": config.DEFAULT_SYMBOL,
        "price": float(current["Close"]),
        "closed_price": float(closed.iloc[-1]["Close"]),
        "candle": {"open": float(current["Open"]), "high": float(current["High"]), "low": float(current["Low"]), "close": float(current["Close"])},
        "signal": result.get("signal", "HOLD") if high_confidence else "HOLD",
        "raw_signal": result.get("signal", "HOLD") if high_confidence else "MODEL NOT VALIDATED",
        "raw_model_signal": result.get("signal", "HOLD"),
        "direction": direction if high_confidence else "NONE",
        "confidence": float(confidence) if confidence is not None else None,
        "probabilities": probabilities,
        "reason": result.get("reason", "No qualifying setup"),
        "decision": decision,
        "decision_reason": decision_reason,
        "memory": {**stats, "gate": memory_allowed, "gate_reason": memory_reason},
        "active_positions": active,
        "analysing": "Completed candles, ATR, trend filters, model probabilities, active trade, and trade memory.",
        "market_context": market_context,
    }


async def broadcast_loop():
    while True:
        try:
            payload = json.dumps(get_snapshot(), default=str)
            if clients:
                await asyncio.gather(*(client.send(payload) for client in clients.copy()), return_exceptions=True)
        except Exception as exc:
            print(f"[live_server] data warning: {exc}")
        await asyncio.sleep(5)


async def websocket_handler(websocket):
    clients.add(websocket)
    try:
        await websocket.send(json.dumps(get_snapshot(), default=str))
        await websocket.wait_closed()
    finally:
        clients.discard(websocket)


async def main():
    threading.Thread(target=start_http_server, daemon=True).start()
    async with websockets.serve(websocket_handler, WS_HOST, WS_PORT):
        print(f"WebSocket stream: ws://localhost:{WS_PORT}")
        await broadcast_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nLive server stopped.")
