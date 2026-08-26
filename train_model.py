# train_model.py - Retrained to match multi_strategy.py feature set EXACTLY (fixes feature_names mismatch)
import os
import pandas as pd
import numpy as np
import ta
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score
import config
import data_feed
from signal_engine import get_timeframe_filters


def detect_candlestick_patterns(df):
    open_p, high_p, low_p, close_p = df['Open'], df['High'], df['Low'], df['Close']
    body = abs(close_p - open_p)
    candle_range = (high_p - low_p).replace(0, 0.001)

    df['Pattern_Doji'] = (body <= (candle_range * 0.1)).astype(int)
    df['Pattern_Marubozu'] = (body >= (candle_range * 0.85)).astype(int)
    lower_shadow = np.minimum(open_p, close_p) - low_p
    df['Pattern_Hammer'] = ((lower_shadow >= (body * 2)) & (body > 0)).astype(int)
    upper_shadow = high_p - np.maximum(open_p, close_p)
    df['Pattern_ShootingStar'] = ((upper_shadow >= (body * 2)) & (body > 0)).astype(int)
    df['Pattern_BullishEngulfing'] = ((close_p > open_p) & (close_p.shift(1) < open_p.shift(1)) & (close_p >= open_p.shift(1))).astype(int)
    df['Pattern_BearishEngulfing'] = ((close_p < open_p) & (close_p.shift(1) > open_p.shift(1)) & (close_p <= open_p.shift(1))).astype(int)
    return df


def calculate_daily_reset_vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
    vol = df['Volume'].replace(0, np.nan).fillna(1)
    vol_price = typical_price * vol
    dates = df.index.date
    cum_vol_price = vol_price.groupby(dates).cumsum()
    cum_vol = vol.groupby(dates).cumsum()
    vwap = cum_vol_price / cum_vol
    return vwap.fillna(typical_price)


def calculate_garman_klass_volatility(df: pd.DataFrame, window: int = 14) -> pd.Series:
    log_hl = np.log(df['High'] / df['Low']) ** 2
    log_co = np.log(df['Close'] / df['Open']) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return np.sqrt(gk.rolling(window).mean()).fillna(0)


# This MUST stay in sync with FEATURE_COLUMNS in signal_engine.py (the single
# source of truth used at inference time, live and backtest). If you
# add/remove/rename a feature in one file, mirror it in the other and retrain,
# or you will hit the same "feature_names mismatch" error again.
FEATURE_COLUMNS = [
    'RSI', 'EMA_Diff', 'ADX', 'ATR_Pct', 'GK_Volatility', 'VWAP_Diff',
    'BB_Width', 'BB_Pband', 'Return_1', 'Return_3',
    'Is_Morning_Open', 'Is_Lunch_Chop', 'Is_Power_Hour',
    'Pattern_Doji', 'Pattern_Marubozu', 'Pattern_Hammer',
    'Pattern_ShootingStar', 'Pattern_BullishEngulfing', 'Pattern_BearishEngulfing',
    'Body_Pct', 'Upper_Wick_Pct', 'Lower_Wick_Pct',
    'Vol_Ratio', 'Vol_Delta',
    'log_ret_1', 'log_ret_5', 'volatility_15m', 'volatility_60m',
    'Return_lag1', 'Return_lag2', 'Return_lag3',
    'Body_Pct_lag1', 'Body_Pct_lag2',
    'Vol_Delta_lag1', 'Vol_Delta_lag2',
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Builds the exact same feature set that scan_all_assets() in multi_strategy.py uses at inference time."""
    df = df.copy()

    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['EMA_9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA_21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['EMA_Diff'] = (df['EMA_9'] - df['EMA_21']) / df['EMA_21']

    adx_ind = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
    df['ADX'] = adx_ind.adx()

    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100.0
    df['GK_Volatility'] = calculate_garman_klass_volatility(df, window=14)

    df['VWAP'] = calculate_daily_reset_vwap(df)
    df['VWAP_Diff'] = (df['Close'] - df['VWAP']) / df['VWAP']

    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Width'] = bb.bollinger_wband()
    df['BB_Pband'] = bb.bollinger_pband()

    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute
    df['Is_Morning_Open'] = ((df['Hour'] == 9) & (df['Minute'] >= 15) | (df['Hour'] == 10)).astype(int)
    df['Is_Lunch_Chop'] = ((df['Hour'] >= 11) & (df['Hour'] <= 13)).astype(int)
    df['Is_Power_Hour'] = ((df['Hour'] == 14) | ((df['Hour'] == 15) & (df['Minute'] <= 15))).astype(int)

    df['Return_1'] = df['Close'].pct_change(1)
    df['Return_3'] = df['Close'].pct_change(3)

    df = detect_candlestick_patterns(df)

    # --- Candle anatomy + volume (continuous, mirrors signal_engine.py) ---
    candle_range = (df['High'] - df['Low']).replace(0, np.nan)
    body = (df['Close'] - df['Open']).abs()
    upper_wick = df['High'] - df[['Open', 'Close']].max(axis=1)
    lower_wick = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['Body_Pct'] = (body / candle_range).fillna(0)
    df['Upper_Wick_Pct'] = (upper_wick / candle_range).fillna(0)
    df['Lower_Wick_Pct'] = (lower_wick / candle_range).fillna(0)

    vol_ma20 = df['Volume'].rolling(20).mean()
    df['Vol_Ratio'] = (df['Volume'] / vol_ma20.replace(0, np.nan)).fillna(1.0)
    df['Vol_Delta'] = df['Volume'].diff().fillna(0)

    # --- Momentum speed (log returns) + multi-window realized volatility ---
    close = df['Close'].replace(0, np.nan)
    df['log_ret_1'] = np.log(close / close.shift(1)).fillna(0)
    df['log_ret_5'] = np.log(close / close.shift(5)).fillna(0)
    df['volatility_15m'] = df['log_ret_1'].rolling(15).std().fillna(0)
    df['volatility_60m'] = df['log_ret_1'].rolling(60).std().fillna(0)

    # --- Multi-bar sequence lags (past 3 bars) ---
    for lag in (1, 2, 3):
        df[f'Return_lag{lag}'] = df['Return_1'].shift(lag).fillna(0)
    for lag in (1, 2):
        df[f'Body_Pct_lag{lag}'] = df['Body_Pct'].shift(lag).fillna(0)
        df[f'Vol_Delta_lag{lag}'] = df['Vol_Delta'].shift(lag).fillna(0)

    return df


def train_institutional_ai():
    print("==================================================")
    print("🤖 TRAINING AI MODEL (feature set synced with multi_strategy.py) 🤖")
    print("==================================================")

    print(f"1. Downloading 365 days of {config.DEFAULT_SYMBOL} historical data ({config.TRADE_TIMEFRAME})...")
    df = data_feed.fetch_btc_historical_data(
        config.DEFAULT_SYMBOL, config.TRADE_TIMEFRAME, days=365
    )
    df = df.rename(columns={column: column.title() for column in df.columns})
    if df.empty:
        raise RuntimeError("Could not fetch BTC training data from Binance.")

    print("2. Computing indicators (matching live scan feature set)...")
    df = build_features(df)

    # --- Target matches the configured trade rule, not an arbitrary return. ---
    # The model must learn the same ATR target/stop outcome that the backtester
    # and live signal path use, otherwise confidence describes a different trade.
    # This simulates the SAME TP/SL/hold-time rule bar-by-bar to build labels
    # that match what the model is actually used for at inference time.
    TP_ATR_MULT = config.ATR_TP_MULTIPLIER
    SL_ATR_MULT = config.ATR_SL_MULTIPLIER
    MAX_HOLD_BARS = 20

    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    atr = df['ATR'].values
    n = len(df)
    target = np.ones(n, dtype=int)  # default HOLD

    print(f"2b. Simulating {TP_ATR_MULT}x/{SL_ATR_MULT}x ATR TP/SL over max "
          f"{MAX_HOLD_BARS} bars to build labels (matches real trade rule)...")
    for idx in range(n - 1):
        if np.isnan(atr[idx]) or atr[idx] <= 0:
            continue
        entry = close[idx]
        tp_long = entry + atr[idx] * TP_ATR_MULT
        sl_long = entry - atr[idx] * SL_ATR_MULT
        tp_short = entry - atr[idx] * TP_ATR_MULT
        sl_short = entry + atr[idx] * SL_ATR_MULT

        end = min(idx + 1 + MAX_HOLD_BARS, n)
        long_result = None
        short_result = None
        for j in range(idx + 1, end):
            if long_result is None:
                if low[j] <= sl_long:
                    long_result = -1
                elif high[j] >= tp_long:
                    long_result = 1
            if short_result is None:
                if high[j] >= sl_short:
                    short_result = -1
                elif low[j] <= tp_short:
                    short_result = 1
            if long_result is not None and short_result is not None:
                break

        # A bar is labeled CALL only if going long would have hit TP before SL,
        # and PUT only if going short would have hit TP before SL. If neither
        # (or both -- choppy/ambiguous), it stays HOLD.
        if long_result == 1 and short_result != 1:
            target[idx] = 2  # CALL
        elif short_result == 1 and long_result != 1:
            target[idx] = 0  # PUT

    df['Target'] = target
    df.iloc[-MAX_HOLD_BARS:, df.columns.get_loc('Target')] = 1  # tail has no full lookahead window -> HOLD

    # Train on the same minimum-quality bars that reach model scoring in the
    # deployed signal path. Including weak candles and low-volatility chop
    # teaches the classifier about rows that can never become trades.
    filters = get_timeframe_filters(df)
    candle_range = df['High'] - df['Low'] + 1e-6
    body_ratio = (df['Close'] - df['Open']).abs() / candle_range
    eligible = (
        (body_ratio >= filters['body'])
        & (df['ADX'] >= filters['adx'])
        & (df['ATR'] >= df['Close'] * filters['atr'])
    )
    df = df.loc[eligible].copy()

    df.dropna(subset=FEATURE_COLUMNS, inplace=True)

    X = df[FEATURE_COLUMNS]
    y = df['Target']

    print("\n2b. Class balance check (0=PUT, 1=HOLD, 2=CALL):")
    class_counts = y.value_counts().sort_index()
    for cls, count in class_counts.items():
        print(f"    class {cls}: {count} samples ({count/len(y)*100:.1f}%)")
    if class_counts.get(1, 0) / len(y) > 0.70:
        print("    WARNING: HOLD (class 1) is over 70% of labels. A model can score")
        print("    high 'accuracy' by mostly predicting HOLD -- accuracy alone is")
        print("    not a reliable metric here. Consider a smaller Future_Return")
        print("    threshold (currently 0.0015) to get more balanced CALL/PUT labels,")
        print("    or evaluate with per-class precision/recall instead of raw accuracy.")

    print(f"3. Training AI model on {len(X)} samples with {len(FEATURE_COLUMNS)} features...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Auto class-weighting: if HOLD (or any class) dominates the training split,
    # give the minority CALL/PUT classes proportionally more weight so the model
    # doesn't just learn to predict HOLD for a cheap accuracy score. This is the
    # automatic fix for the imbalance the check above only warned about.
    train_counts = y_train.value_counts()
    n_samples = len(y_train)
    n_classes = train_counts.shape[0]
    class_weight = {
        cls: n_samples / (n_classes * count) for cls, count in train_counts.items()
    }
    sample_weight = y_train.map(class_weight).values
    print("    Class weights applied (higher = rarer class, boosted more):")
    for cls, w in sorted(class_weight.items()):
        print(f"      class {cls}: weight {w:.3f}")

    model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.03,
        random_state=int(os.getenv("TRAINING_SEED", "42")),
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    directional_mask = np.isin(y_pred, [0, 2])
    directional_precision = (
        precision_score(y_test[directional_mask], y_pred[directional_mask], labels=[0, 2],
                        average="weighted", zero_division=0)
        if directional_mask.any() else 0.0
    )
    test_probabilities = model.predict_proba(X_test)
    predicted_confidence = test_probabilities.max(axis=1)
    directional_confidence = (
        float(predicted_confidence[directional_mask].mean())
        if directional_mask.any() else 0.0
    )
    print(f"\n✅ Training Complete! Test Accuracy: {accuracy*100:.2f}%")
    print(f"Directional win rate (CALL/PUT predictions): {directional_precision*100:.2f}%")
    print(f"Average confidence on directional predictions: {directional_confidence*100:.2f}%")
    print("\nPer-class precision/recall (0=PUT, 1=HOLD, 2=CALL):")
    print(classification_report(y_test, y_pred, target_names=["PUT", "HOLD", "CALL"], zero_division=0))
    print("⚠️  Note: accuracy alone doesn't tell you if this is profitable - check backtester.py metrics too.")
    print("⚠️  What matters most: precision on CALL and PUT (when the model DOES fire a")
    print("    signal, how often is it right) -- not overall accuracy, which HOLD dominates.\n")

    if directional_precision < config.MIN_VALIDATED_WIN_RATE:
        raise RuntimeError(
            f"Model rejected: directional win rate {directional_precision:.2%} is below "
            f"the required {config.MIN_VALIDATED_WIN_RATE:.2%}."
        )
    if directional_confidence < config.MIN_VALIDATED_CONFIDENCE:
        raise RuntimeError(
            f"Model rejected: average directional confidence {directional_confidence:.2%} is below "
            f"the required {config.MIN_VALIDATED_CONFIDENCE:.2%}."
        )

    model.save_model("xgboost_model.json")
    print("✅ Model saved as 'xgboost_model.json' - feature set now matches multi_strategy.py exactly.")


if __name__ == "__main__":
    train_institutional_ai()