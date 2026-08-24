# main.py - Multi-Position Risk-Managed Scanner Bot
import time
import json
import os
import atexit
import subprocess
import sys
import socket
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
RISK_PER_TRADE_PCT = config.RISK_PER_TRADE_PCT
BOT_LOCK_FILE = "paper_bot.lock"
DASHBOARD_PROCESS = None


def start_dashboard():
    """Start Streamlit once so the bot and dashboard use one command."""
    global DASHBOARD_PROCESS
    if os.getenv("DISABLE_AUTO_DASHBOARD") == "1":
        print("[DASHBOARD] Automatic dashboard launch disabled by environment.")
        return

    dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.py")
    if not os.path.exists(dashboard_path):
        print(f"[DASHBOARD] File not found: {dashboard_path}")
        return

    try:
        with socket.create_connection(("127.0.0.1", 8501), timeout=0.5):
            print("[DASHBOARD] Streamlit is already running at http://localhost:8501")
            return
    except OSError:
        pass

    try:
        DASHBOARD_PROCESS = subprocess.Popen(
            [
                sys.executable, "-m", "streamlit", "run", dashboard_path,
                "--server.address", "127.0.0.1", "--server.port", "8501",
                "--server.headless", "true",
            ],
            cwd=os.path.dirname(dashboard_path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        atexit.register(stop_dashboard)
        print("[DASHBOARD] Started automatically at http://localhost:8501")
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[DASHBOARD] Automatic launch failed: {exc}")


def stop_dashboard():
    """Stop only the Streamlit process launched by this bot instance."""
    global DASHBOARD_PROCESS
    if DASHBOARD_PROCESS is not None and DASHBOARD_PROCESS.poll() is None:
        DASHBOARD_PROCESS.terminate()
        DASHBOARD_PROCESS = None


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
    start_dashboard()
    print("==========================================================")
    print("🚀 BITCOIN-ONLY ALGO BOT STARTED (Risk-Managed) 🚀")
    print("==========================================================")

    broker = PaperBroker(
        initial_capital=config.BTC_START_CAPITAL_USD,
        max_concurrent_positions=(
            config.PAPER_SCHEDULED_MAX_CONCURRENT_POSITIONS
            if config.PAPER_TRADING_MODE and config.PAPER_SCHEDULED_TRADES_ENABLED
            else 1
        ),
        risk_per_trade_pct=RISK_PER_TRADE_PCT,
    )
    print("📊 PAPER-ONLY MODE: simulated orders enabled; real-money orders disabled")
    prediction_tracker.ensure_csv()
    last_entry_candle = None
    last_scheduled_slot = None
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
            execution_threshold = (
                config.PAPER_MIN_ACTIONABLE_CONFIDENCE
                if config.PAPER_TRADING_MODE else config.MIN_ACTIONABLE_CONFIDENCE
            )
            actionable = confidence is not None and float(confidence) >= execution_threshold
            inside_entry_window = 60 <= (time.time() % 300) < 240
            scheduled_mode = config.PAPER_TRADING_MODE and config.PAPER_SCHEDULED_TRADES_ENABLED
            interval_seconds = config.PAPER_SCHEDULED_INTERVAL_MINUTES * 60
            scheduled_slot = int(time.time() // interval_seconds) if scheduled_mode else None
            scheduled_entry = scheduled_mode and scheduled_slot != last_scheduled_slot
            trade_candidate = best_trade or next(
                (item for item in all_results if item.get("Symbol") == config.DEFAULT_SYMBOL),
                None,
            )
            strategy_entry = (
                trade_candidate is not None and best_trade is not None and actionable and inside_entry_window
                and signal_candle != last_entry_candle and not broker.positions
            )
            # Scheduled entries are intentionally disabled by default because
            # they can turn HOLD/low-confidence predictions into trades.
            scheduled_entry = scheduled_entry and actionable and signal_candle != last_entry_candle
            if (strategy_entry or scheduled_entry) and trade_candidate is not None:
                signal = trade_candidate.get("Signal", "HOLD")
                if signal == "BUY_CALL":
                    option_type = "CALL"
                elif signal == "BUY_PUT":
                    option_type = "PUT"
                else:
                    # Never manufacture a direction from candle colour.
                    continue
                slot_suffix = f"_{scheduled_slot}" if scheduled_entry else ""
                premium = round(float(trade_candidate["Price"]) * 0.02, 2)
                broker.buy_option(
                    symbol=f"BTCUSDT_OPT_{option_type}{slot_suffix}", option_type=option_type,
                    entry_price=premium, stock_price=float(trade_candidate["Price"]), qty=None,
                    stop_loss_price=trade_candidate.get("Stop_Loss"),
                    target_price=trade_candidate.get("Take_Profit"),
                    signal_confidence=trade_candidate.get("Confidence"),
                    signal_reason=trade_candidate.get("Signal_Reason", ""),
                    market_context=trade_candidate.get("Market_Context", {}),
                    entry_candle_time=trade_candidate.get("Candle_Time"),
                )
                last_entry_candle = signal_candle
                if scheduled_entry:
                    last_scheduled_slot = scheduled_slot

            for open_symbol, position in list(broker.positions.items()):
                market = next((item for item in all_results if item.get("Symbol") == "BTCUSDT"), None)
                if market:
                    position["last_stock_price"] = float(market["Price"])
                    position["exit_candle_time"] = market.get("Candle_Time")
                    stock_change = float(market["Price"]) - float(position.get("entry_stock_price", market["Price"]))
                    premium_change = stock_change * 0.5 if position["type"] == "CALL" else -stock_change * 0.5
                    broker.update_market_price(
                        open_symbol, max(1.0, round(position["entry_price"] + premium_change, 2)),
                        stock_price=float(market["Price"]),
                    )
                    if open_symbol in broker.positions:
                        position = broker.positions[open_symbol]
                        position["current_candle_time"] = market.get("Candle_Time", "")
                        position["current_price"] = float(market["Price"])
                        position["unrealized_pnl"] = round(
                            stock_change * float(position.get("qty", 0.0))
                            * (1 if position["type"] == "CALL" else -1),
                            2,
                        )
                        broker._update_active_json()

            print(f"\r[SCANNING] {summary_str} | paper_open:{len(broker.positions)} predictions pending:{tracker_state['pending']} resolved:{tracker_state['resolved']} training:{training_status}", end="")

            time.sleep(5)

        except Exception as e:
            print(f"\n[ERROR] {e}")
            time.sleep(5)

if __name__ == "__main__":
    run_bitcoin_bot()