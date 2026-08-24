"""
ANTONY QUANT AI ALGO TERMINAL - BROKER INTEGRATOR V3.0
Streamlit Cloud Native Multi-Endpoint Binance API Integrator (Bypasses US Geofences)
"""

import os
import hmac
import hashlib
import time
import requests
import streamlit as st

import config
import trade_logger


def check_live_trading_readiness():
    """
    Gate that decides whether 'REAL' execution mode is allowed at all, based on
    objective evidence — not vibes. Checks:
      1. At least MIN_PAPER_TRADING_DAYS of paper-trade history exists.
        2. A backtest report file exists with the required win rate, drawdown,
            and profit factor.

    Returns dict: {"ready": bool, "reasons": [str, ...]} — "reasons" lists
    every unmet condition so the UI can show exactly what's missing.
    """
    reasons = []

    # --- Condition 1: paper trading history span ---
    all_trades = trade_logger.load_trades()
    if not all_trades:
        reasons.append(f"No paper trades logged yet (need {config.MIN_PAPER_TRADING_DAYS} days of history).")
    else:
        dates = sorted(set(t.get("date", "") for t in all_trades if t.get("date")))
        if len(dates) < config.MIN_PAPER_TRADING_DAYS:
            reasons.append(
                f"Only {len(dates)} distinct paper-trading day(s) logged "
                f"(need {config.MIN_PAPER_TRADING_DAYS})."
            )

    # --- Condition 2: backtest evidence ---
    backtest_report = None
    if os.path.exists("backtest_report.json"):
        try:
            import json
            with open("backtest_report.json", "r") as f:
                backtest_report = json.load(f)
        except Exception:
            backtest_report = None

    if backtest_report is None:
        reasons.append("No backtest_report.json found — run backtester.py against historical data first.")
    else:
        metrics = backtest_report.get("metrics", {})
        win_rate = metrics.get("win_rate", 0.0)
        max_dd = abs(metrics.get("max_drawdown_pct", 100.0))
        profit_factor = metrics.get("profit_factor", 0.0)
        if isinstance(profit_factor, str):
            profit_factor = 0.0

        try:
            win_rate = float(win_rate)
            max_dd = float(max_dd)
            profit_factor = float(profit_factor)
        except (TypeError, ValueError):
            reasons.append("Backtest metrics contain non-numeric values.")
            win_rate, max_dd, profit_factor = 0.0, 100.0, 0.0

        if win_rate < config.MIN_BACKTEST_WIN_RATE:
            reasons.append(
                f"Backtest win rate {win_rate}% is below the required {config.MIN_BACKTEST_WIN_RATE}%."
            )
        if max_dd > config.MAX_BACKTEST_DRAWDOWN_PCT:
            reasons.append(
                f"Backtest max drawdown {max_dd}% exceeds the allowed {config.MAX_BACKTEST_DRAWDOWN_PCT}%."
            )
        if profit_factor < config.MIN_PAPER_PROFIT_FACTOR:
            reasons.append(
                f"Backtest profit factor {profit_factor} is below the required {config.MIN_PAPER_PROFIT_FACTOR}."
            )

    return {"ready": len(reasons) == 0, "reasons": reasons}

def get_saved_binance_keys():
    """Auto-loads Binance API Keys from Secrets/.env so refresh NEVER clears them!"""
    # st.secrets raises StreamlitSecretNotFoundError if no secrets.toml exists
    # at all (even via .get()), so this must be wrapped -- not just defaulted.
    try:
        api_key = st.secrets.get("BINANCE_API_KEY", "")
        secret_key = st.secrets.get("BINANCE_SECRET_KEY", "")
    except Exception:
        api_key = ""
        secret_key = ""

    if not api_key:
        api_key = os.getenv("BINANCE_API_KEY", st.session_state.get("binance_api_key", ""))
    if not secret_key:
        secret_key = os.getenv("BINANCE_SECRET_KEY", st.session_state.get("binance_secret_key", ""))
    
    # Fallback to direct .env reading if running locally
    if not api_key or not secret_key:
        if os.path.exists(".env"):
            try:
                with open(".env", "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("BINANCE_API_KEY="):
                            api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        elif line.startswith("BINANCE_SECRET_KEY="):
                            secret_key = line.split("=", 1)[1].strip().strip('"').strip("'")
            except Exception:
                pass
                
    return str(api_key).strip(), str(secret_key).strip()


def test_binance_connection_streamlit_cloud(api_key: str, secret_key: str):
    """
    Direct Multi-Endpoint Binance API Verification for Streamlit Cloud.
    Cycles through api1, api2, api3, api4.binance.com to bypass US IP blocks!
    """
    if not api_key or not secret_key:
        return False, "API Key or Secret cannot be empty!", 0.0

    endpoints = [
        "https://api1.binance.com",
        "https://api2.binance.com",
        "https://api3.binance.com",
        "https://api4.binance.com"
    ]

    for base_url in endpoints:
        try:
            timestamp = int(time.time() * 1000)
            query_string = f"timestamp={timestamp}&recvWindow=60000"
            signature = hmac.new(
                secret_key.strip().encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            full_url = f"{base_url}/api/v3/account?{query_string}&signature={signature}"
            headers = {"X-MBX-APIKEY": api_key.strip()}

            res = requests.get(full_url, headers=headers, timeout=4)
            
            if res.status_code == 200:
                data = res.json()
                usdt_balance = 0.0
                for item in data.get('balances', []):
                    if item.get('asset') == 'USDT':
                        usdt_balance = float(item.get('free', 0.0))
                        break
                return True, "VERIFIED", usdt_balance
            elif res.status_code == 401:
                return False, "❌ BINANCE REJECTED KEY: Invalid API Key or Secret!", 0.0
        except Exception:
            continue

    # Honest failure: every endpoint failed/timed out. Do NOT report success
    # or invent a balance — the caller must treat this as "unverified".
    return False, "⚠️ COULD NOT VERIFY: All Binance endpoints unreachable or timed out. Credentials NOT confirmed valid.", 0.0

test_binance_connection = test_binance_connection_streamlit_cloud

def get_binance_spot_usdt_balance(api_key, secret_key):
    """
    Dynamically fetch real Binance Spot USDT balance.
    Returns None on failure — caller must show 'unavailable', never a guessed number.
    """
    success, msg, bal = test_binance_connection_streamlit_cloud(api_key, secret_key)
    return bal if success else None

def verify_and_save_binance_credentials(api_key: str, secret_key: str):
    success, msg, bal = test_binance_connection_streamlit_cloud(api_key, secret_key)
    return success, bal


def render_broker_integrator_tab():
    st.subheader("🔑 Streamlit Cloud Native Binance API Integrator")

    saved_key, saved_sec = get_saved_binance_keys()

    # --- LIVE-TRADING READINESS GATE ---
    # REAL mode is only offered as a choice if there's objective evidence
    # (enough paper-trading history + a passing backtest). This replaces the
    # earlier behavior where REAL mode could be picked from day one.
    readiness = check_live_trading_readiness()

    if not readiness["ready"]:
        st.error("🚫 **Real-money execution is locked.** Requirements not yet met:")
        for r in readiness["reasons"]:
            st.markdown(f"- {r}")
        st.info("Paper Trading Simulator is the only available mode until these are satisfied.")
        st.session_state['execution_mode'] = "PAPER"
        st.toggle("🟡 Paper  ⟷  🟢 Live", value=False, disabled=True, key="exec_mode_toggle_locked")
        st.caption("Toggle unlocks once the readiness conditions above are met.")
        st.markdown("---")
        return

    # 1. EXECUTION MODE TOGGLE (left = Paper, right = Live)
    # Off (left) = PAPER, always the default on a fresh session -- REAL is
    # something the user must deliberately switch to, never something they
    # land on by accident just because the readiness gate happened to pass.
    current_mode = st.session_state.get('execution_mode', 'PAPER')
    toggle_default = (current_mode == 'REAL')

    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_a:
        st.markdown("🟡 **Paper**")
    with col_b:
        is_live = st.toggle(
            "Execution Mode",
            value=toggle_default,
            key="execution_mode_toggle",
            label_visibility="collapsed"
        )
    with col_c:
        st.markdown("**Live** 🟢")

    selected_mode = "🟢 Binance Live Real Money (Real Spot Balance)" if is_live \
        else "🟡 Paper Trading Simulator ($100,000.00 Virtual)"

    if "Real Money" in selected_mode:
        st.error(
            "⚠️ **You're about to arm REAL-MONEY execution.** Orders placed in this "
            "mode use your actual Binance balance. This cannot be undone once a "
            "trade fills."
        )
        confirm_real = st.checkbox(
            "I understand this will trade with real funds and accept the risk.",
            key="confirm_real_mode_checkbox"
        )
        if confirm_real:
            st.session_state['execution_mode'] = "REAL"
            st.success("⚡ **STATUS: LIVE BINANCE REAL-MONEY EXECUTION ACTIVE!**")
        else:
            # Dropdown shows "Real Money" but nothing is armed until confirmed.
            st.session_state['execution_mode'] = "PAPER"
            st.warning("🧪 **STATUS: PAPER SIMULATOR ACTIVE** (confirm the checkbox above to arm REAL mode).")
    else:
        st.session_state['execution_mode'] = "PAPER"
        st.warning("🧪 **STATUS: PAPER SIMULATOR ACTIVE.**")

    st.markdown("---")

    # 2. BINANCE LIVE API CREDENTIALS FORM
    st.subheader("🟡 Binance Spot Crypto Live API Credentials")

    api_key_input = st.text_input("Binance API Key:", value=saved_key, type="password", key="b_key_input_st_cloud")
    secret_key_input = st.text_input("Binance API Secret:", value=saved_sec, type="password", key="b_sec_input_st_cloud")

    # Auto-fetch balance on page load if keys exist. No fabricated default —
    # if verification fails/times out, we show "Unavailable", not a fake number.
    live_usdt = None
    verify_failed_msg = None
    if saved_key and saved_sec:
        success, msg, free_usdt = test_binance_connection_streamlit_cloud(saved_key, saved_sec)
        if success:
            live_usdt = free_usdt
            st.session_state['binance_live_usdt_balance'] = free_usdt
        else:
            verify_failed_msg = msg

    if live_usdt is not None:
        st.info(f"💰 **Detected Live Binance Spot USDT Balance:** `${live_usdt:.2f} USDT`")
    else:
        st.warning(
            "⚠️ **Balance Unavailable** — could not verify Binance credentials yet."
            + (f" ({verify_failed_msg})" if verify_failed_msg else " Enter and verify your API keys below.")
        )

    # 3. VERIFY & SAVE BUTTON
    if st.button("💾 Verify & Save Binance Credentials"):
        with st.spinner("Connecting to Binance Mirror Endpoints..."):
            success, msg, free_usdt = test_binance_connection_streamlit_cloud(api_key_input, secret_key_input)

            if success:
                st.session_state['binance_api_key'] = api_key_input.strip()
                st.session_state['binance_secret_key'] = secret_key_input.strip()
                st.session_state['binance_live_usdt_balance'] = free_usdt
                # Do NOT set execution_mode to REAL here. Verifying credentials
                # only proves the keys work -- it is not consent to trade real
                # money. REAL mode must only be armed via the toggle + explicit
                # "I understand this will trade with real funds" checkbox above.
                st.toast(f"🎉 Binance API Verified! Spot Balance: ${free_usdt:.2f} USDT", icon="🟢")
                st.info("Credentials saved. Use the Paper/Live toggle above and confirm the checkbox to arm real-money trading.")
            else:
                # Do NOT flip execution_mode to REAL and do NOT show a fake balance.
                st.error(f"❌ Verification failed — credentials NOT saved. {msg}")

        st.rerun()