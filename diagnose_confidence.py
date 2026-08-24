# diagnose_confidence.py
# Run this to see WHY 0 trades are firing: prints the distribution of the
# model's max confidence across every bar, so you can see whether 0.70 is
# reachable at all, or whether the threshold needs to change / model needs
# retraining.
#
# Usage:  python diagnose_confidence.py

import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import config
from signal_engine import build_features, FEATURE_COLUMNS

print(f"Fetching {config.DEFAULT_SYMBOL} ({config.TIMEFRAME}, 60 days)...")
df = yf.download(tickers=config.DEFAULT_SYMBOL, period="60d", interval=config.TIMEFRAME, progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [c[0] for c in df.columns]
df.dropna(inplace=True)
print(f"Got {len(df)} bars.")

model = XGBClassifier()
model.load_model("xgboost_model.json")

df_feat = build_features(df)

confidences = []
preds = []
for i in range(25, len(df_feat)):
    row = df_feat.iloc[i]
    try:
        features = pd.DataFrame([{col: row[col] for col in FEATURE_COLUMNS}]).fillna(0)
        probs = model.predict_proba(features)[0]
        confidences.append(float(np.max(probs)))
        preds.append(int(np.argmax(probs)))
    except Exception as e:
        print(f"  skipped bar {i}: {e}")

confidences = np.array(confidences)
preds = np.array(preds)

print("\n=== MODEL CONFIDENCE DISTRIBUTION ===")
print(f"  bars scored:     {len(confidences)}")
print(f"  min confidence:  {confidences.min():.3f}")
print(f"  max confidence:  {confidences.max():.3f}")
print(f"  mean confidence: {confidences.mean():.3f}")
print(f"  median:          {np.median(confidences):.3f}")
for pct in [50, 70, 90, 95, 99]:
    print(f"  {pct}th percentile: {np.percentile(confidences, pct):.3f}")

print("\n=== PREDICTED CLASS BREAKDOWN (0=PUT, 1=HOLD, 2=CALL) ===")
for cls in [0, 1, 2]:
    count = (preds == cls).sum()
    print(f"  class {cls}: {count} bars ({count/len(preds)*100:.1f}%)")

print("\n=== HOW MANY BARS WOULD FIRE AT DIFFERENT THRESHOLDS ===")
for thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    fires = ((confidences >= thresh) & (preds != 1)).sum()
    print(f"  threshold {thresh:.2f}: {fires} bars would fire a signal ({fires/len(confidences)*100:.2f}%)")