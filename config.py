# ================================================================================
# ANTONY QUANT AI TERMINAL - CONFIGURATION ENGINE (5-LAYER INSTITUTIONAL EDITION)
# ================================================================================
import os

# --- CORE TRADING MODE ---
PRIMARY_MODE = "BTC_SPOT"
DEFAULT_SYMBOL = "BTCUSDT"
BTC_START_CAPITAL_USD = 20.00
TIMEFRAME = "5m"
TRADE_TIMEFRAME = "5m"
SUPPORTED_TIMEFRAMES = ("1m", "5m", "15m", "1h", "4h", "1d")

# --- RISK KILL-SWITCH & LIVE-TRADING READINESS GATE ---
MAX_DAILY_LOSS_PCT = 3.0            # Auto-halt trading if today's loss exceeds 3% of capital
MAX_DAILY_TRADES = 3                # Hard cap on trades per day regardless of signals
RISK_PER_TRADE_PCT = 0.005          # Risk at most 0.5% of current capital per trade
PAPER_MAX_DAILY_TRADES = 999999     # Paper data collection is not limited by the live cap
MIN_PAPER_TRADING_DAYS = 30         # Minimum days of paper-trade history required before REAL mode unlocks
MIN_BACKTEST_WIN_RATE = 55.0        # Backtest win rate threshold (%) required before REAL mode unlocks
MAX_BACKTEST_DRAWDOWN_PCT = 15.0    # Backtest max drawdown threshold (%) - REAL mode blocked above this
MIN_ACTIONABLE_CONFIDENCE = 0.55    # Actionable signal requires a 55% directional model score
PAPER_TRADING_MODE = True           # Paper-only data collection; real mode remains gated
PAPER_MIN_ACTIONABLE_CONFIDENCE = 0.55  # Match the actionable directional-confidence gate
PAPER_EXPERIMENTAL_SIGNALS = False      # Do not execute near-random experimental signals
PAPER_MIN_DIRECTIONAL_PROBABILITY = 0.55
PAPER_SCHEDULED_TRADES_ENABLED = False  # Scheduled entries bypass signal quality and are disabled
PAPER_SCHEDULED_INTERVAL_MINUTES = 5
PAPER_SCHEDULED_MAX_CONCURRENT_POSITIONS = 4
MIN_PAPER_PROFIT_FACTOR = 1.0      # Paper execution requires a positive validated edge
MAX_PAPER_DRAWDOWN_PCT = 15.0      # Paper execution is blocked beyond this drawdown
BINANCE_TAKER_FEE = 0.00075        # Standard Binance spot taker fee per side
ALLOW_UNVALIDATED_PAPER_TRADING = True  # Collect paper evidence before real-mode review
AUTO_RETRAIN_ENABLED = True             # Retrain the paper model in the background
AUTO_RETRAIN_INTERVAL_HOURS = 24

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")