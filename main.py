# main.py - Multi-Position Risk-Managed Scanner Bot
import time
import json
import os
import atexit
import subprocess
import sys
from multi_strategy import scan_all_assets
from paper_broker import PaperBroker
import prediction_tracker
import market_recorder
import config
import auto_trainer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

MAX_CONCURRENT_POSITIONS = 3
RISK_PER_TRADE_PCT = 0.02  # risk 2% of capital per trade -> drives position sizing
BOT_LOCK_FILE = "paper_bot.lock"


def acquire_bot_lock():
    """Allow only one paper bot process to write shared state files."""
    try:
        if os.path.exists(BOT_LOCK_FILE):
            with open(BOT_LOCK_FILE, encoding="utf-8") as handle:
                pid = int(handle.read().strip())
            if _process_is_running(pid):
                return False
            os.remove(BOT_LOCK_FILE)
        fd = os.open(BOT_LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        atexit.register(release_bot_lock)
        return True
    except (OSError, ValueError):
        return False


def _process_is_running(pid):
    """Check a Windows PID without os.kill(pid, 0), which can raise WinError 87."""
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except (OSError, SystemError):
            return False
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True, check=False,
        )
        return str(pid) in result.stdout
    except (OSError, subprocess.SubprocessError):
        return False


def release_bot_lock():
    try:
        if os.path.exists(BOT_LOCK_FILE):
            with open(BOT_LOCK_FILE, encoding="utf-8") as handle:
                owner_pid = int(handle.read().strip())
            if owner_pid == os.getpid():
                os.remove(BOT_LOCK_FILE)
    except (OSError, ValueError):
        pass


def backtest_allows_paper_entries():
    """Prevent the execution loop from trading while validation is failing."""
    if config.ALLOW_UNVALIDATED_PAPER_TRADING:
        return True, "Paper evidence collection mode enabled; real mode remains locked"
    report_path = "backtest_report.json"
    if not os.path.exists(report_path):
        return False, "No backtest report available"
    try:
        with open(report_path, encoding="utf-8") as handle:
            metrics = json.load(handle).get("metrics", {})
        profit_factor = metrics.get("profit_factor", 0.0)
        drawdown = abs(float(metrics.get("max_drawdown_pct", 100.0)))
        if isinstance(profit_factor, str):
            profit_factor = 0.0
        if float(profit_factor) < config.MIN_PAPER_PROFIT_FACTOR:
            return False, f"Validated profit factor {profit_factor} is below {config.MIN_PAPER_PROFIT_FACTOR}"
        if drawdown > config.MAX_PAPER_DRAWDOWN_PCT:
            return False, f"Validated drawdown {drawdown:.2f}% exceeds {config.MAX_PAPER_DRAWDOWN_PCT:.2f}%"
        return True, "Backtest validation passed"
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, f"Backtest report unreadable: {exc}"

def run_bitcoin_bot():
    if not acquire_bot_lock():
        print("[BOT LOCK] Another main.py instance is already running; exiting this duplicate.")
        return
    print("==========================================================")
    print("🚀 BITCOIN-ONLY ALGO BOT STARTED (Risk-Managed) 🚀")
    print("==========================================================")

    broker = PaperBroker(
        initial_capital=config.BTC_START_CAPITAL_USD,
        max_concurrent_positions=1,
        risk_per_trade_pct=RISK_PER_TRADE_PCT,
    )
    print("📊 PAPER-ONLY MODE: simulated orders enabled; real-money orders disabled")
    prediction_tracker.ensure_csv()
    last_entry_candle = None
    while True:
        try:
            auto_trainer.training_finished()
            training_status = auto_trainer.maybe_retrain()
            best_trade, all_results = scan_all_assets()

            # Persist this cycle's market data (price, RSI, signal, etc.) so
            # it accumulates over time instead of being lost between loops.
            market_recorder.record_snapshot(all_results)

            summary_str = " | ".join([
                f"{item['Name']}: {item['Signal']}"
                + (f" ({item.get('Signal_Reason', '')})" if item.get('Signal') in ("ERROR", "HOLD") and item.get('Signal_Reason') else "")
                for item in all_results
            ])
            tracker_state = prediction_tracker.process_scan_results(all_results)
            signal_candle = best_trade.get("Candle_Time") if best_trade else None
            confidence = best_trade.get("Confidence") if best_trade else None
            if best_trade and best_trade.get("Confidence_Source") == "PAPER_EXPERIMENTAL":
                execution_threshold = config.PAPER_MIN_DIRECTIONAL_PROBABILITY
            else:
                execution_threshold = (
                    config.PAPER_MIN_ACTIONABLE_CONFIDENCE
                    if config.PAPER_TRADING_MODE else config.MIN_ACTIONABLE_CONFIDENCE
                )
            actionable = confidence is not None and float(confidence) >= execution_threshold
            inside_entry_window = 60 <= (time.time() % 300) < 240
            if (
                best_trade is not None and actionable and inside_entry_window
                and signal_candle != last_entry_candle and not broker.positions
            ):
                signal = best_trade["Signal"]
                option_type = "CALL" if signal == "BUY_CALL" else "PUT"
                premium = round(float(best_trade["Price"]) * 0.02, 2)
                broker.buy_option(
                    symbol=f"BTCUSDT_OPT_{option_type}", option_type=option_type,
                    entry_price=premium, stock_price=float(best_trade["Price"]), qty=None,
                    stop_loss_pct=0.15, target_pct=0.30,
                    signal_confidence=best_trade.get("Confidence"),
                    signal_reason=best_trade.get("Signal_Reason", ""),
                    market_context=best_trade.get("Market_Context", {}),
                )
                last_entry_candle = signal_candle

            for open_symbol, position in list(broker.positions.items()):
                market = next((item for item in all_results if item.get("Symbol") == "BTCUSDT"), None)
                if market:
                    position["last_stock_price"] = float(market["Price"])
                    stock_change = float(market["Price"]) - float(position.get("entry_stock_price", market["Price"]))
                    premium_change = stock_change * 0.5 if position["type"] == "CALL" else -stock_change * 0.5
                    broker.update_market_price(open_symbol, max(1.0, round(position["entry_price"] + premium_change, 2)))

            print(f"\r[SCANNING] {summary_str} | paper_open:{len(broker.positions)} predictions pending:{tracker_state['pending']} resolved:{tracker_state['resolved']} training:{training_status}", end="")

            time.sleep(5)

        except Exception as e:
            print(f"\n[ERROR] {e}")
            time.sleep(5)

if __name__ == "__main__":
    run_bitcoin_bot()