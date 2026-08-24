# square_off_now.py - Force Close Today's Trade
import os
import json
import yfinance as yf
import pandas as pd
from paper_broker import PaperBroker

def force_close_trade():
    broker = PaperBroker(initial_capital=100000)
    
    ACTIVE_JSON = "active_trade.json"
    if os.path.exists(ACTIVE_JSON):
        with open(ACTIVE_JSON, "r", encoding="utf-8") as f:
            active_data = json.load(f)
            
        if active_data.get("status") == "ACTIVE":
            sym = active_data["symbol"]
            e_price = active_data["entry_price"]
            e_stock_p = active_data["entry_stock_price"]
            qty = active_data["qty"]
            
            df = yf.download("RELIANCE.NS", period="1d", interval="1m", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            last_stock_p = float(df['Close'].iloc[-1])
            stock_diff = last_stock_p - e_stock_p
            exit_premium = max(1.0, round(e_price + (stock_diff * 0.5), 2))
            
            broker.position = {
                "entry_time": active_data["entry_time"],
                "symbol": sym,
                "type": active_data["type"],
                "entry_price": e_price,
                "qty": qty,
                "stop_loss": active_data["stop_loss"],
                "target": active_data["target"]
            }
            
            pnl = round((exit_premium - e_price) * qty, 2)
            broker.capital = 100000 + pnl
            broker._log_trade(exit_premium, "MARKET_CLOSE_SQUARE_OFF", pnl, "2026-08-14 15:15:00")
            broker._clear_active_json()
            print(f"✅ இன்றைய Reliance டிரேட் வெற்றிகரமாக க்ளோஸ் செய்யப்பட்டது! PnL: +₹{pnl:.2f}")

if __name__ == "__main__":
    force_close_trade()