# paper_broker.py - Institutional Paper Broker with SEBI Order Slicing & VIX Protection
import os
import csv
import json
import datetime
import tempfile
from notifier import send_telegram_alert
import trade_memory
import config

CSV_FILE = "trades.csv"
ACTIVE_JSON = "active_trade.json"
STATE_JSON = "live_state.json"
BROKERAGE_PER_TRADE = 45.0

SINGLE_CANDLE_TIMEOUT_EXIT = False  # FIXED: Disabled forced 5-min exit!
MAX_HOLDING_MINUTES = 20.0          # Trades given full 20 mins to hit Target 1 or Target 2
MAX_DAILY_TRADES = 3                # Enforced 3-Trade Daily Cap


def slice_order_quantity(symbol, quantity):
    """Return one fractional BTC paper-order slice; NSE lot slicing is removed."""
    return [round(float(quantity), 6)]

def evaluate_paper_trade_exit(trade: dict, live_price: float, elapsed_minutes: float) -> tuple:
    """
    Evaluates whether an active trade has hit Target 1 (+6%), Target 2 (+12%), or Stop Loss (-3%).
    NO MORE FORCED 5-MINUTE SINGLE CANDLE TIMEOUT EXITS!
    """
    if not trade or trade.get('status') != 'ACTIVE':
        return False, "NO_ACTIVE_TRADE", 0.0, 0.0

    entry_price = float(trade.get('entry_price', 0.0))
    target_price = float(trade.get('target_price', 0.0))
    stop_loss = float(trade.get('stop_loss', 0.0))
    option_type = trade.get('option_type', trade.get('type', 'CALL'))
    qty = float(trade.get('quantity', trade.get('qty', 0.001)))

    # Calculate Direct Spot PnL (No Synthetic Option Delta Decay Lag)
    if option_type in ['CALL', 'BUY_CALL', 'LONG', 'BUY']:
        pnl = (live_price - entry_price) * qty
        is_target_hit = live_price >= target_price if target_price > 0 else False
        is_sl_hit = live_price <= stop_loss if stop_loss > 0 else False
    else: # PUT / SHORT / BUY_PUT
        pnl = (entry_price - live_price) * qty
        is_target_hit = live_price <= target_price if target_price > 0 else False
        is_sl_hit = live_price >= stop_loss if stop_loss > 0 else False

    # 1. Target Hit Exit (+6% / +12%)
    if is_target_hit:
        return True, "TARGET_1_HIT (+6.0% Gain)", live_price, pnl

    # 2. Hard Stop Loss Exit (-3.0% Cap)
    if is_sl_hit:
        return True, "HARD_STOP_LOSS_HIT (-3.0% Risk Cap)", live_price, pnl

    # 3. Max Holding Timeout Exit (20 Minutes Limit - NOT 5 Minutes!)
    if elapsed_minutes >= MAX_HOLDING_MINUTES:
        return True, f"MAX_TIME_EXPIRATION_EXIT ({int(MAX_HOLDING_MINUTES)} Mins Limit)", live_price, pnl

    return False, "HOLDING", live_price, pnl


def execute_paper_trade(symbol, option_type, entry_price, target_price, stop_loss, ai_confidence, qty=0.001):
    """Executes Paper Trade and sends Direct Fail-Safe Telegram Push Alert"""
    
    trade_dict = {
        'symbol': symbol,
        'option_type': option_type,
        'entry_price': entry_price,
        'target_price': target_price,
        'stop_loss': stop_loss,
        'quantity': qty,
        'status': 'ACTIVE',
        'start_time': datetime.datetime.now(),
        'entry_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Clean HTML Format for Telegram API
    clean_symbol = str(symbol).replace('_OPT_', ' ').replace('_', ' ')
    conf_pct = float(ai_confidence) if float(ai_confidence) <= 1.0 else float(ai_confidence) / 100.0

    entry_html_msg = f"🚀 <b>NEW ACTIVE TRADE ENTERED!</b>\n" \
                     f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" \
                     f"• <b>Symbol</b>      : {clean_symbol}\n" \
                     f"• <b>Direction</b>   : {option_type}\n" \
                     f"• <b>Entry Price</b> : ${entry_price:,.2f}\n" \
                     f"• <b>Target 1</b>    : ${target_price:,.2f}\n" \
                     f"• <b>Stop Loss</b>   : ${stop_loss:,.2f}\n" \
                     f"• <b>AI Score</b>    : {conf_pct:.1%} Confidence\n" \
                     f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" \
                     f"<i>ANTONY Quant AI Algo Trading Terminal</i>"
    try:
        sent = send_telegram_alert(entry_html_msg)
        if sent:
            print(f"✅ TELEGRAM ENTRY ALERT SENT FOR {symbol}!")
    except Exception as e:
        print(f"Telegram Dispatch Error: {e}")

    return trade_dict

def execute_paper_trade_entry(symbol: str, option_type: str, entry_price: float, qty: int, target_price: float, sl_price: float, ai_confidence: float = 72.0):
    """Executes Paper Entry, Logs IST Time, and Triggers Immediate Telegram Entry Alert"""
    return execute_paper_trade(symbol, option_type, entry_price, target_price, sl_price, ai_confidence, qty=qty)

def get_current_ist_timestamp_str():
    """Returns current time formatted as string. (IST offset assumed via system timezone / IST-configured host.)"""
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def execute_paper_trade_exit(trade_record: dict, exit_price: float, exit_reason: str):
    """Executes Paper Exit, Logs IST Exit Time, and Triggers Telegram Exit Alert"""
    ist_exit_time_str = get_current_ist_timestamp_str()
    
    trade_record['Exit_Time'] = ist_exit_time_str
    trade_record['Exit_Price'] = round(exit_price, 2)
    trade_record['Exit_Reason'] = exit_reason
    trade_record['status'] = "CLOSED"
    
    # Calculate PnL & Friction
    symbol = trade_record.get('Symbol', trade_record.get('symbol', ''))
    is_crypto = any(k in symbol.upper() for k in ["BITCOIN", "ETHEREUM", "BTC", "ETH", "SOL", "BNB", "XRP"])
    curr = "$" if is_crypto else "₹"
    
    entry_p = float(trade_record.get('Entry_Price', trade_record.get('entry_price', 0.0)))
    q = float(trade_record.get('Quantity', trade_record.get('quantity', trade_record.get('qty', 1))))
    
    opt_t = trade_record.get('Option_Type', trade_record.get('option_type', 'CALL'))
    if opt_t in ['CALL', 'BUY_CALL', 'LONG', 'BUY']:
        gross_pnl = (exit_price - entry_p) * q
    else:
        gross_pnl = (entry_p - exit_price) * q
        
    if is_crypto:
        fees = round((abs(entry_p * q) + abs(exit_price * q)) * config.BINANCE_TAKER_FEE, 2)
    else:
        fees = 48.00
    net_pnl = round(gross_pnl - fees, 2)
    
    trade_record['Gross_PnL'] = f"{curr}{gross_pnl:+,.2f}"
    trade_record['Brokerage_&_Taxes'] = f"-{curr}{fees:,.2f}"
    trade_record['Net_PnL'] = f"{curr}{net_pnl:+,.2f}"
    
    # 🔔 TRIGGER TELEGRAM EXIT ALERT
    exit_msg = f"""🏁 <b>TRADE COMPLETED & LOGGED!</b>

📌 <b>Asset Symbol:</b> {symbol}
🔚 <b>Exit Reason:</b> {exit_reason}
💵 <b>Entry Price:</b> {curr}{entry_p:,.2f} ➔ <b>Exit Price:</b> {curr}{exit_price:,.2f}
📊 <b>Net Realized P&L:</b> {curr}{net_pnl:+,.2f}
⏰ <b>Exit Time:</b> {ist_exit_time_str}
"""
    try:
        send_telegram_alert(exit_msg)
    except Exception as e:
        print(f"Telegram Exit Alert Error: {e}")
    
    return trade_record
    
    return trade_record

def execute_paper_exit(trade_record, exit_price, exit_reason):
    return execute_paper_trade_exit(trade_record, exit_price, exit_reason)

def enforce_strict_risk_reward_exit(trade_record: dict, current_price: float) -> dict:
    """Enforces Max Loss Cap at -$25.00 and Allows Target Gains up to +$50 / +$100"""
    entry_price = float(trade_record['Entry_Price'])
    qty = int(trade_record['Quantity'])
    symbol = trade_record['Symbol']
    
    is_crypto = any(k in symbol.upper() for k in ["BITCOIN", "ETHEREUM", "BTC", "ETH"])
    
    current_pnl = (current_price - entry_price) * qty
    
    if is_crypto:
        # STRICT CRYPTO CAP: Max Loss = -$25.00
        if current_pnl <= -25.00:
            return {"should_exit": True, "reason": "🛑 STRICT MAX LOSS CAP (-$25.00 Hit)", "exit_price": current_price}
            
        # TARGET 1: +$50.00 (+6% Gain)
        if current_pnl >= 50.00 and not trade_record.get('t1_hit', False):
            trade_record['t1_hit'] = True
            return {"should_exit": False, "action": "PARTIAL_PROFIT_BOOKING", "reason": "🎯 TARGET 1 HIT (+$50.00 Gain)"}

        # TARGET 2: +$100.00 (+12% Gain)
        if current_pnl >= 100.00:
            return {"should_exit": True, "reason": "🎯 TARGET 2 FULL EXIT (+$100.00 Gain)", "exit_price": current_price}

    return {"should_exit": False, "reason": "HOLDING"}

def apply_multi_asset_trailing_lock(trade_record: dict, current_price: float) -> dict:
    entry_price = float(trade_record['Entry_Price'])
    qty = int(trade_record['Quantity'])
    symbol = trade_record['Symbol']
    
    current_pnl = (current_price - entry_price) * qty
    is_crypto = any(k in symbol.upper() for k in ["BITCOIN", "ETHEREUM", "BTC", "ETH"])

    if is_crypto:
        # CRYPTO DOLLAR RULES ($)
        trigger_pnl = 35.00   # +$35.00 Gain Trigger
        lock_pnl = 15.00      # Lock +$15.00
        max_sl = -25.00       # Max Loss Cap -$25.00
    else:
        # NSE RUPEES RULES (₹)
        trigger_pnl = 250.00  # +₹250.00 Gain Trigger
        lock_pnl = 100.00     # Lock +₹100.00 (Covers ₹52 Brokerage)
        max_sl = -180.00      # Max Loss Cap -₹180.00

    # Max Loss Cut
    if current_pnl <= max_sl:
        return {"should_exit": True, "reason": "🛑 STRICT MAX LOSS CAP HIT", "exit_price": current_price}

    # Dynamic Profit Lock Trigger
    max_pnl = max(trade_record.get('max_pnl_seen', 0.0), current_pnl)
    trade_record['max_pnl_seen'] = max_pnl

    if max_pnl >= trigger_pnl:
        if current_pnl <= (max_pnl - lock_pnl):
            return {"should_exit": True, "reason": "🔒 DYNAMIC TRAILING PROFIT LOCK TRIGGERED", "exit_price": current_price}

    return {"should_exit": False, "reason": "HOLDING"}



class PaperBroker:
    def __init__(self, initial_capital=100000.0, max_concurrent_positions=3, risk_per_trade_pct=0.02):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        # positions is now a dict keyed by symbol -> supports multiple concurrent trades
        self.positions = {}
        self.daily_trades_count = 0
        self.daily_pnl = 0.0
        self.max_trades_per_day = (
            config.PAPER_MAX_DAILY_TRADES if config.PAPER_TRADING_MODE else config.MAX_DAILY_TRADES
        )
        self.max_daily_loss_limit = initial_capital * (config.MAX_DAILY_LOSS_PCT / 100.0)
        self.max_concurrent_positions = max_concurrent_positions
        # % of current capital risked per trade -> drives position sizing
        self.risk_per_trade_pct = risk_per_trade_pct
        trade_memory.migrate_csv_once()
        trade_memory.migrate_json_once()
        self._init_csv()
        self._load_state()

    # ---- Backward-compatible single-position accessor ----
    # Old code (main.py etc.) checks `broker.position is None`.
    # This property keeps that working by returning the first open position, if any.
    @property
    def position(self):
        if not self.positions:
            return None
        return next(iter(self.positions.values()))

    def has_room_for_new_position(self, symbol):
        if symbol in self.positions:
            return False  # already have a position on this symbol
        return len(self.positions) < self.max_concurrent_positions

    def calculate_position_size(self, entry_price, stop_loss_pct):
        """
        Risk-based position sizing: risk a fixed % of current capital per trade,
        sized so that hitting the stop-loss loses ~risk_per_trade_pct of capital.
        """
        if entry_price <= 0 or stop_loss_pct <= 0:
            return 0
        risk_amount = self.capital * self.risk_per_trade_pct
        loss_per_unit = entry_price * stop_loss_pct
        if loss_per_unit <= 0:
            return 0
        qty = risk_amount / loss_per_unit
        return round(max(qty, 0.0001), 6)

    def _init_csv(self):
        columns = [
            "Trade_ID", "Entry_Time", "Entry_Candle_Time", "Exit_Time", "Exit_Candle_Time", "Duration_Minutes",
            "Symbol", "Direction", "Entry_Price", "Exit_Price", "Stop_Loss", "Take_Profit", "Premium_Entry_Price", "Premium_Exit_Price", "Premium_Stop_Loss", "Premium_Take_Profit",
            "Quantity", "Exit_Reason", "Outcome", "Gross_PnL", "Brokerage_Taxes", "Net_PnL",
            "Capital_Balance", "AI_Confidence", "Signal_Reason", "Post_Mortem", "Market_Context",
        ]
        if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
            with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(columns)
            return

        with open(CSV_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_columns = reader.fieldnames or []
            rows = list(reader)
        if existing_columns == columns:
            return
        with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", delete=False) as temp:
            writer = csv.DictWriter(temp, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                legacy_schema = "Premium_Entry_Price" not in existing_columns
                if legacy_schema:
                    row = {
                        **row,
                        "Premium_Entry_Price": row.get("Entry_Price", ""),
                        "Premium_Exit_Price": row.get("Exit_Price", ""),
                        "Premium_Stop_Loss": row.get("Stop_Loss", ""),
                        "Premium_Take_Profit": row.get("Take_Profit", ""),
                        "Entry_Price": "",
                        "Exit_Price": "",
                        "Stop_Loss": "",
                        "Take_Profit": "",
                    }
                writer.writerow({column: row.get(column, "") for column in columns})
            temp_path = temp.name
        os.replace(temp_path, CSV_FILE)

    def _update_active_json(self):
        if self.positions:
            data = {"status": "ACTIVE", "positions": self.positions}
        else:
            data = {"status": "NO_POSITION"}

        with open(ACTIVE_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        state_data = {
            "initial_capital": self.initial_capital,
            "capital": self.capital,
            "daily_trades_count": self.daily_trades_count,
            "daily_pnl": self.daily_pnl,
            "positions": self.positions
        }
        with open(STATE_JSON, "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=4)

    def _load_state(self):
        if os.path.exists(STATE_JSON):
            try:
                with open(STATE_JSON, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    stored_initial_capital = state.get("initial_capital")
                    same_account = stored_initial_capital is not None and float(stored_initial_capital) == float(self.initial_capital)
                    if same_account:
                        self.capital = state.get("capital", self.initial_capital)
                        self.daily_trades_count = state.get("daily_trades_count", 0)
                        self.daily_pnl = state.get("daily_pnl", 0.0)
                    else:
                        self.capital = self.initial_capital
                        self.daily_trades_count = 0
                        self.daily_pnl = 0.0
                        self.positions = {}
                        self._update_active_json()
                        return
                    # Backward compatible: old files stored a single "position" dict
                    if "positions" in state and state["positions"]:
                        self.positions = state["positions"]
                    elif state.get("position"):
                        old_pos = state["position"]
                        self.positions = {old_pos["symbol"]: old_pos}
                    else:
                        self.positions = {}
            except:
                pass

    def _clear_active_json(self, symbol=None):
        if symbol and symbol in self.positions:
            del self.positions[symbol]
        with open(ACTIVE_JSON, "w", encoding="utf-8") as f:
            json.dump({"status": "NO_POSITION" if not self.positions else "ACTIVE",
                       "open_positions": list(self.positions.keys())}, f, indent=4)
        self._update_active_json()

    def buy_option(self, symbol, option_type, entry_price, stock_price=0.0, qty=None, stop_loss_pct=0.15, target_pct=0.30,
                   stop_loss_price=None, target_price=None,
                   signal_confidence=None, signal_reason="", market_context=None, entry_candle_time=None):
        if self.daily_trades_count >= self.max_trades_per_day:
            print(f"\n[RISK GUARD] 🚫 Max Trades Limit ({self.max_trades_per_day}) reached for today.")
            return

        if self.daily_pnl <= -self.max_daily_loss_limit:
            print(f"\n[RISK GUARD] 🚨 Hard Daily Loss Limit (-{config.MAX_DAILY_LOSS_PCT:.1f}%) hit! Bot Kill-Switch Activated.")
            return

        if not self.has_room_for_new_position(symbol):
            print(f"\n[RISK GUARD] 🚫 Max Concurrent Positions ({self.max_concurrent_positions}) reached, or {symbol} already open.")
            return

        has_spot_levels = stop_loss_price is not None and target_price is not None and stock_price > 0
        if has_spot_levels:
            stop_loss_price = float(stop_loss_price)
            target_price = float(target_price)
            delta = 0.5
            stop_loss = round(entry_price + (stop_loss_price - stock_price) * delta, 2)
            target = round(entry_price + (target_price - stock_price) * delta, 2)
            effective_stop_pct = abs(entry_price - stop_loss) / entry_price
        elif option_type == "CALL":
            stop_loss = round(entry_price * (1 - stop_loss_pct), 2)
            target = round(entry_price * (1 + target_pct), 2)
            effective_stop_pct = stop_loss_pct
        else:
            stop_loss = round(entry_price * (1 + stop_loss_pct), 2)
            target = round(entry_price * (1 - target_pct), 2)
            effective_stop_pct = stop_loss_pct

        # Size from the same effective stop that this position will use.
        if qty is None:
            qty = self.calculate_position_size(entry_price, effective_stop_pct)
            if qty <= 0:
                print(f"\n[RISK GUARD] 🚫 Position size calculated as 0 for {symbol}, skipping trade.")
                return

        # SEBI Slicing Validation
        child_slices = slice_order_quantity(symbol, qty)
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        trade_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S%f")
        p_curr = "$" if "USD" in symbol or "BTC" in symbol or "ETH" in symbol else "₹"

        if has_spot_levels:
            target_stock_price = round(target_price, 2)
            sl_stock_price = round(stop_loss_price, 2)
        elif option_type == "CALL":
            target_stock_price = round(stock_price + (entry_price * target_pct / 0.5), 2)
            sl_stock_price = round(stock_price - (entry_price * stop_loss_pct / 0.5), 2)
        else:
            target_stock_price = round(stock_price - (entry_price * target_pct / 0.5), 2)
            sl_stock_price = round(stock_price + (entry_price * stop_loss_pct / 0.5), 2)

        self.positions[symbol] = {
            "trade_id": trade_id,
            "entry_time": now_str,
            "entry_candle_time": entry_candle_time or now_str,
            "symbol": symbol,
            "type": option_type,
            "entry_price": round(entry_price, 2),
            "entry_stock_price": round(stock_price, 2),
            "target_stock_price": target_stock_price,
            "sl_stock_price": sl_stock_price,
            "risk_model": "ATR_SPOT_MAPPED_TO_SYNTHETIC_PREMIUM" if has_spot_levels else "LEGACY_PREMIUM_PERCENT",
            "qty": qty,
            "slices": child_slices,
            "stop_loss": stop_loss,
            "initial_stop_loss": stop_loss,
            "target": target,
            "signal_confidence": signal_confidence,
            "signal_reason": signal_reason,
            "market_context": market_context or {},
            "max_premium_seen": entry_price,
            "trailed_to_breakeven": False
        }
        self.daily_trades_count += 1
        self._update_active_json()
        
        telegram_msg = (
            f"🚨 <b>ALGO TRADE ENTERED!</b>\n\n"
            f"<b>Symbol:</b> {symbol} ({option_type})\n"
            f"<b>Stock Price:</b> {p_curr}{stock_price:,.2f}\n"
            f"<b>Option Premium:</b> {p_curr}{entry_price:.2f}\n"
            f"<b>Quantity:</b> {qty} (SEBI Slices: {len(child_slices)})\n"
            f"<b>Stop Loss:</b> {p_curr}{stop_loss:.2f}\n"
            f"<b>Target:</b> {p_curr}{target:.2f}\n"
            f"<b>Time:</b> {now_str}"
        )
        send_telegram_alert(telegram_msg)
        print(f"\n[{now_str}] 📥 [TRADE ENTERED] {symbol} ({option_type}) | Stock Entry: {p_curr}{stock_price:.2f}")

    def update_market_price(self, symbol, current_price, stock_price=None):
        """Update price for a SPECIFIC symbol's open position (multi-position aware)."""
        pos = self.positions.get(symbol)
        if pos is None:
            return

        entry = pos["entry_price"]
        qty = pos["qty"]
        now_time = datetime.datetime.now()
        now_str = now_time.strftime("%Y-%m-%d %H:%M:%S")

        # Track Max Premium Seen
        max_seen = pos.get("max_premium_seen", entry)
        if current_price > max_seen:
            pos["max_premium_seen"] = current_price

        # Determine trigger exit
        trigger_exit = False
        reason = ""
        if stock_price is not None and pos.get("risk_model") == "ATR_SPOT_MAPPED_TO_SYNTHETIC_PREMIUM":
            pos["last_stock_price"] = float(stock_price)
            spot_change = float(stock_price) - float(pos.get("entry_stock_price", stock_price))
            current_price = max(1.0, round(pos["entry_price"] + spot_change * 0.5, 2))
        if pos["type"] == "CALL":
            target_hit = current_price >= pos["target"]
            stop_hit = current_price <= pos["stop_loss"]
        else:
            target_hit = current_price <= pos["target"]
            stop_hit = current_price >= pos["stop_loss"]
        if target_hit:
            trigger_exit = True
            reason = "TARGET_HIT"
        elif stop_hit:
            trigger_exit = True
            reason = "STOP_LOSS_HIT"
        else:
            try:
                entry_time = datetime.datetime.strptime(pos["entry_time"], "%Y-%m-%d %H:%M:%S")
                elapsed_minutes = (now_time - entry_time).total_seconds() / 60.0
            except (KeyError, TypeError, ValueError):
                elapsed_minutes = 0.0
            if elapsed_minutes >= MAX_HOLDING_MINUTES:
                trigger_exit = True
                reason = f"MAX_TIME_EXPIRATION_EXIT ({int(MAX_HOLDING_MINUTES)} Mins Limit)"
        if not trigger_exit and now_time.time() >= datetime.time(15, 15) and "USD" not in pos["symbol"]:
            trigger_exit = True
            reason = "3:15_PM_MARKET_CLOSE"
        # Portfolio-level circuit breaker: force-exit if daily loss limit hit mid-trade
        elif not trigger_exit and self.daily_pnl <= -self.max_daily_loss_limit:
            trigger_exit = True
            reason = "DAILY_LOSS_LIMIT_FORCE_EXIT"

        if trigger_exit:
            temp_record = {
                'Symbol': pos['symbol'],
                'Option_Type': pos['type'],
                'Entry_Price': pos['entry_price'],
                'Quantity': pos['qty']
            }
            res_record = execute_paper_exit(temp_record, current_price, reason)

            # Extract float values for calculations
            direction = 1 if pos["type"] == "CALL" else -1
            gross_pnl = (current_price - entry) * qty * direction
            if any(crypto in symbol.upper() for crypto in ["BITCOIN", "ETHEREUM", "BTC", "ETH", "SOL", "BNB", "XRP"]):
                entry_notional = abs(entry * qty)
                exit_notional = abs(current_price * qty)
                deducted_charges = round((entry_notional + exit_notional) * config.BINANCE_TAKER_FEE, 2)
            else:
                stt_gst = (current_price * qty * 0.0015) + 7.50
                deducted_charges = round(40.00 + stt_gst, 2)
            net_pnl = round(gross_pnl - deducted_charges, 2)

            self.capital += net_pnl
            self.daily_pnl += net_pnl

            # Log to CSV using res_record strings
            self._log_trade(
                pos,
                current_price,
                reason,
                res_record['Gross_PnL'],
                res_record['Brokerage_&_Taxes'],
                res_record['Net_PnL'],
                now_str
            )
            self._clear_active_json(symbol=symbol)

    def update_all_positions(self, price_lookup: dict):
        """
        Convenience helper: pass a dict of {symbol: current_price} for ALL open
        positions in one call, e.g. from a scan loop across multiple assets.
        """
        for symbol in list(self.positions.keys()):
            if symbol in price_lookup:
                self.update_market_price(symbol, price_lookup[symbol])

    def _log_trade(self, pos, exit_price, reason, gross_pnl_str, brokerage_str, net_pnl_str, exit_time_str):
        from ai_analyst import explain_trade_outcome
        net_pnl = _money_value(net_pnl_str)
        post_mortem = explain_trade_outcome(
            reason, net_pnl, pos.get("signal_reason", ""), pos.get("market_context", {})
        )
        entry_time = _parse_trade_datetime(pos.get("entry_time"))
        exit_time = _parse_trade_datetime(exit_time_str)
        duration_minutes = max(0.0, (exit_time - entry_time).total_seconds() / 60.0) if entry_time and exit_time else None
        outcome = "WIN" if net_pnl > 0 else "LOSS"
        trade_id = pos.get("trade_id", datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S%f"))
        with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "Trade_ID", "Entry_Time", "Entry_Candle_Time", "Exit_Time", "Exit_Candle_Time", "Duration_Minutes",
                "Symbol", "Direction", "Entry_Price", "Exit_Price", "Stop_Loss", "Take_Profit", "Premium_Entry_Price", "Premium_Exit_Price", "Premium_Stop_Loss", "Premium_Take_Profit",
                "Quantity", "Exit_Reason", "Outcome", "Gross_PnL", "Brokerage_Taxes", "Net_PnL",
                "Capital_Balance", "AI_Confidence", "Signal_Reason", "Post_Mortem", "Market_Context",
            ])
            writer.writerow({
                "Trade_ID": trade_id,
                "Entry_Time": pos["entry_time"],
                "Entry_Candle_Time": pos.get("entry_candle_time", ""),
                "Exit_Time": exit_time_str,
                "Exit_Candle_Time": pos.get("exit_candle_time", ""),
                "Duration_Minutes": f"{duration_minutes:.2f}" if duration_minutes is not None else "",
                "Symbol": pos["symbol"], "Direction": pos["type"],
                "Entry_Price": f"{pos.get('entry_stock_price', pos['entry_price']):.2f}",
                "Exit_Price": f"{pos.get('last_stock_price', ''):.2f}" if pos.get("last_stock_price") is not None else "",
                "Stop_Loss": f"{pos.get('sl_stock_price', ''):.2f}" if pos.get("sl_stock_price") is not None else "",
                "Take_Profit": f"{pos.get('target_stock_price', ''):.2f}" if pos.get("target_stock_price") is not None else "",
                "Premium_Entry_Price": f"{pos['entry_price']:.2f}",
                "Premium_Exit_Price": f"{exit_price:.2f}",
                "Premium_Stop_Loss": f"{pos['stop_loss']:.2f}", "Premium_Take_Profit": f"{pos['target']:.2f}",
                "Quantity": pos["qty"], "Exit_Reason": reason, "Outcome": outcome,
                "Gross_PnL": gross_pnl_str, "Brokerage_Taxes": brokerage_str, "Net_PnL": net_pnl_str,
                "Capital_Balance": f"{self.capital:.2f}",
                "AI_Confidence": pos.get("signal_confidence", ""),
                "Signal_Reason": pos.get("signal_reason", ""), "Post_Mortem": post_mortem,
                "Market_Context": json.dumps(pos.get("market_context", {}), default=str),
            })

        p_curr = "$" if "USD" in pos["symbol"] or "BTC" in pos["symbol"] or "ETH" in pos["symbol"] else "₹"
        exit_msg = (
            f"🏁 <b>TRADE COMPLETED & LOGGED!</b>\n\n"
            f"<b>Symbol:</b> {pos['symbol']} ({pos['type']})\n"
            f"<b>Exit Reason:</b> {reason}\n"
            f"<b>Entry Price:</b> {p_curr}{pos['entry_price']:.2f}\n"
            f"<b>Exit Price:</b> {p_curr}{exit_price:.2f}\n"
            f"<b>Gross P&L:</b> {gross_pnl_str}\n"
            f"<b>Brokerage & Taxes:</b> {brokerage_str}\n"
            f"<b>Net Realized P&L:</b> {net_pnl_str}\n"
            f"<b>Bot Post-Mortem:</b> {post_mortem}\n"
            f"<b>Account Capital:</b> {p_curr}{self.capital:,.2f}\n"
            f"<b>Open Positions:</b> {len(self.positions) - 1}"
        )
        send_telegram_alert(exit_msg)
        trade_memory.record_trade({
            "exit_time": exit_time_str,
            "symbol": pos["symbol"],
            "direction": pos["type"],
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "stop_loss": pos["stop_loss"],
            "target": pos["target"],
            "quantity": pos["qty"],
            "gross_pnl": _money_value(gross_pnl_str),
            "fees": _money_value(brokerage_str),
            "net_pnl": _money_value(net_pnl_str),
            "exit_reason": reason,
            "post_mortem": post_mortem,
            "signal_confidence": pos.get("signal_confidence"),
            "signal_reason": pos.get("signal_reason", ""),
            "market_context": pos.get("market_context", {}),
        })
        print(f"✅ [SAVED & NOTIFIED] {pos['symbol']} trade logged & Telegram alert sent.\n")


def _money_value(value):
    try:
        return float(str(value).replace("$", "").replace("₹", "").replace(",", "").replace("+", "").replace("-", "")) * (-1 if "-" in str(value) else 1)
    except (TypeError, ValueError):
        return 0.0


def _parse_trade_datetime(value):
    try:
        return datetime.datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None