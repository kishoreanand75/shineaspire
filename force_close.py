# force_close.py - Dynamic Live Force Close & CSV Logger
import os
import csv
import json
from datetime import datetime

CSV_FILE = "trades.csv"
ACTIVE_JSON = "active_trade.json"

def force_close_active_trade():
    # 1. Check if an active trade exists
    if not os.path.exists(ACTIVE_JSON):
        print("No active trade found to close.")
        return

    try:
        with open(ACTIVE_JSON, "r") as f:
            active_trade = json.load(f)
    except Exception as e:
        print(f"Error reading active trade: {e}")
        return

    # 2. Extract live trade details
    symbol = active_trade.get("symbol", "ETH_USDT")
    option_type = active_trade.get("option_type", "PUT")
    entry_price = float(active_trade.get("entry_price", 0))
    # For force close, assume exit at current live stock/premium price
    exit_price = float(active_trade.get("live_price", entry_price)) 
    stop_loss = float(active_trade.get("stop_loss", 0))
    target = float(active_trade.get("target", 0))
    quantity = int(active_trade.get("quantity", 1))
    
    entry_time = active_trade.get("entry_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    exit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exit_reason = "MANUAL_FORCE_CLOSE"

    # 3. Calculate Realized PnL (For PUT: (Entry - Exit) * Qty or Premium difference)
    entry_prem = float(active_trade.get("entry_premium", entry_price))
    exit_prem = float(active_trade.get("live_premium", exit_price))
    
    if option_type == "PUT":
        pnl = (entry_prem - exit_prem) * quantity * 10 # Adjust multiplier as per lot size
    else:
        pnl = (exit_prem - entry_prem) * quantity * 10

    # Read current capital balance (default to 100022.50 if missing)
    prev_balance = 100022.50
    if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
        with open(CSV_FILE, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                try:
                    prev_balance = float(rows[-1].get("Capital_Balance", 100022.50))
                except:
                    pass

    new_capital_balance = prev_balance + pnl

    # 4. Write Header if missing
    file_exists = os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0
    with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Entry_Time", "Exit_Time", "Symbol", "Option_Type",
                "Entry_Price", "Exit_Price", "Stop_Loss", "Target",
                "Quantity", "Exit_Reason", "PnL", "Capital_Balance"
            ])
        
        # Write Dynamic Live Trade Result
        writer.writerow([
            entry_time,
            exit_time,
            symbol,
            option_type,
            f"{entry_price:.2f}",
            f"{exit_price:.2f}",
            f"{stop_loss:.2f}",
            f"{target:.2f}",
            str(quantity),
            exit_reason,
            f"{pnl:.2f}",
            f"{new_capital_balance:.2f}"
        ])

    # 5. Clear active trade json so dashboard clears the active position view
    if os.path.exists(ACTIVE_JSON):
        os.remove(ACTIVE_JSON)
    
    print(f"✅ Trade Force Closed Successfully! PnL: {pnl:.2f}, New Balance: {new_capital_balance:.2f}")

if __name__ == "__main__":
    force_close_active_trade()