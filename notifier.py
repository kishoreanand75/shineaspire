import requests
import os
try:
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
except Exception:
    TELEGRAM_BOT_TOKEN = ""
    TELEGRAM_CHAT_ID = ""

def send_telegram_alert(message_html: str) -> bool:
    """Send immediate Telegram push alert with sound"""
    if not message_html or not message_html.strip():
        return False
        
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message_html,
        "parse_mode": "HTML",
        "disable_notification": False  # LOUD SOUND ALERT!
    }
    try:
        res = requests.post(url, json=payload, timeout=5)
        if res.status_code == 200:
            return True
        # Plain text fallback
        clean_text = message_html.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", "").replace("<code>", "").replace("</code>", "")
        fallback_res = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": clean_text, "disable_notification": False}, timeout=5)
        return fallback_res.status_code == 200
    except Exception as e:
        print(f"Telegram Alert Error: {e}")
        return False

def send_copilot_order_ticket_alert(symbol: str, action: str, price: float, target: float, stop_loss: float, ai_conf: float, usdt_amount: float = 5.00) -> bool:
    """Sends the Exact 15-Second Binance Copy-Paste Order Sheet to Telegram"""
    clean_sym = f"{symbol}/USDT" if "/" not in symbol else symbol
    if "BITCOIN" in symbol: clean_sym = "BTC/USDT"
    elif "ETHEREUM" in symbol: clean_sym = "ETH/USDT"
    elif "SOLANA" in symbol: clean_sym = "SOL/USDT"
    elif "BNB" in symbol: clean_sym = "BNB/USDT"
    elif "XRP" in symbol: clean_sym = "XRP/USDT"

    msg = f"""🎯 <b>ANTONY AI CO-PILOT — BINANCE SPOT ORDER SHEET</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• <b>Symbol / Pair</b>     : <code>{clean_sym}</code>
• <b>Action</b>            : <b>{action} LIMIT ORDER</b> 🟢

📍 <b>PRICE FIELD</b>      ➔ Type: <code>${price:,.2f}</code>
📍 <b>TOTAL FIELD</b>      ➔ Type: <code>{usdt_amount:.2f} USDT</code>

☑️ <b>Check [x] TP/SL Box on Binance:</b>
🎯 <b>TAKE PROFIT (TP)</b> ➔ Type: <code>${target:,.2f}</code> (+6.0% Gain)
🛡️ <b>STOP LOSS (SL)</b>   ➔ Type: <code>${stop_loss:,.2f}</code> (-3.0% Risk Cap)

🤖 <b>AI CONFIDENCE</b>    ➔ <b>{ai_conf:.1f}%</b> (Institutional Rules Passed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<i>15-Second Copy-Paste Execution | Zero API Stress</i>"""
    return send_telegram_alert(msg)


def send_formatted_signal_alert(symbol: str, engine_result: dict, name: str = None) -> bool:
    """
    Send a complete Telegram alert for a non-HOLD signal coming straight out
    of signal_engine.generate_signal()/decide_from_row(). Includes Direction,
    Entry, Stop Loss, Take Profit (1:2 RRR), AI Confidence %, and 1H HTF trend
    -- the fields that were previously missing/partial in the plain alert.
    """
    signal = engine_result.get("signal", "HOLD")
    if signal == "HOLD":
        return False

    direction = "🟢 BUY / CALL" if signal == "BUY_CALL" else "🔴 SELL / PUT"
    display_symbol = name or symbol

    confidence = engine_result.get("confidence")
    conf_source = engine_result.get("confidence_source", "RULE_ONLY")
    conf_text = f"{confidence * 100:.1f}%" if confidence is not None else "N/A (rule-based)"

    entry_price = engine_result.get("entry_price")
    stop_loss = engine_result.get("stop_loss")
    take_profit = engine_result.get("take_profit")
    rrr = engine_result.get("rrr", 2.0)
    htf_trend = engine_result.get("htf_trend", "UNKNOWN")
    reason = engine_result.get("reason", "")

    price_lines = ""
    if entry_price is not None:
        price_lines = f"""📍 <b>Entry Price</b>       ➔ <code>${entry_price:,.2f}</code>
🛡️ <b>Stop Loss</b>        ➔ <code>${stop_loss:,.2f}</code>
🎯 <b>Take Profit (1:2)</b> ➔ <code>${take_profit:,.2f}</code>
⚖️ <b>Risk:Reward</b>      ➔ <b>1 : {rrr:.1f}</b>
"""

    msg = f"""🚨 <b>AI TRADE SIGNAL — {display_symbol}</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• <b>Direction</b>   : {direction}
{price_lines}🤖 <b>AI Confidence</b> ➔ <b>{conf_text}</b> ({conf_source})
📈 <b>1H HTF Trend</b> ➔ <b>{htf_trend}</b>
📝 <b>Reason</b>      ➔ {reason}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<i>Single-source-of-truth signal (signal_engine.py)</i>"""
    return send_telegram_alert(msg)