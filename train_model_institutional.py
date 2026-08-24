"""
ANTONY QUANT AI ALGO TERMINAL - INSTITUTIONAL ML ENSEMBLE ENGINE V3.0
Includes:
- XGBoost + LightGBM + CatBoost + Random Forest Quad-Voting Classifier
- Isotonic Probability Calibration (PAVA Algorithm)
- Purged & Embargoed Walk-Forward Cross-Validation (TimeSeriesSplit based)

HONESTY NOTE (fixed this pass): earlier versions of this docstring claimed
purged & embargoed walk-forward CV was implemented here. It wasn't --
`TimeSeriesSplit` was imported but never called, and `train_and_get_ensemble_model`
fit the ensemble on the ENTIRE dataset with zero holdout, then calibrated
probabilities on that same model's in-sample predictions (which makes
calibration look better than it really is, since a model is always
over-confident on data it was trained on). Both are now actually implemented
below -- see `run_purged_embargoed_cv` and `train_and_get_ensemble_model`.
"""

import numpy as np
import pandas as pd
import ta
import logging
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Attempt CatBoost import, fallback gracefully if not installed
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)

# 70% AI Win-Rate Confidence Threshold
AI_CONFIDENCE_THRESHOLD = 0.70

FEATURES = ['ADX', 'DMI_Plus', 'DMI_Minus', 'ATR_Ratio', 'BB_Width', 'Body_Ratio', 'Vol_Ratio', 'EMA_Slope']


def compute_advanced_institutional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Institutional Technical Indicators Calculation"""
    df = df.copy()
    df_cols = {col.lower(): col for col in df.columns}
    h_col = df_cols.get('high', 'High')
    l_col = df_cols.get('low', 'Low')
    c_col = df_cols.get('close', 'Close')
    o_col = df_cols.get('open', 'Open')
    v_col = df_cols.get('volume', 'Volume')
    
    # 1. ADX (Trend Strength > 25 Filter)
    adx_ind = ta.trend.ADXIndicator(high=df[h_col], low=df[l_col], close=df[c_col], window=14)
    df['ADX'] = adx_ind.adx()
    df['DMI_Plus'] = adx_ind.adx_pos()
    df['DMI_Minus'] = adx_ind.adx_neg()

    # 2. ATR Ratio (Volatility Expansion)
    df['ATR'] = ta.volatility.average_true_range(df[h_col], df[l_col], df[c_col], window=14)
    df['ATR_Ratio'] = df['ATR'] / df[c_col]

    # 3. Bollinger Bands Squeeze & Width
    bb = ta.volatility.BollingerBands(df[c_col], window=20, window_dev=2)
    df['BB_Width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()

    # 4. Ezekiel Chew Body Range Ratio
    df['Body_Ratio'] = abs(df[c_col] - df[o_col]) / (df[h_col] - df[l_col] + 1e-6)

    # 5. Volume MA Ratio (>= 1.2x)
    df['Vol_MA20'] = df[v_col].rolling(20).mean()
    df['Vol_Ratio'] = df[v_col] / (df['Vol_MA20'] + 1e-6)

    # 6. EMA 9/21 Slope Alignment
    df['EMA9'] = ta.trend.ema_indicator(df[c_col], window=9)
    df['EMA21'] = ta.trend.ema_indicator(df[c_col], window=21)
    df['EMA_Slope'] = (df['EMA9'] - df['EMA21']) / df['EMA21']

    return df.dropna()


def build_labeled_features(df: pd.DataFrame):
    """Shared feature/label construction used by both CV and final training,
    so the walk-forward numbers you see are computed on the exact same
    features/labels the deployed model is trained on."""
    df_feat = compute_advanced_institutional_features(df)
    df_cols = {col.lower(): col for col in df_feat.columns}
    c_col = df_cols.get('close', 'Close')
    # NOTE: this 0.3% next-bar-return label does not match the ATR-based
    # TP/SL/hold-time rule the bot actually trades with (see train_model.py,
    # which fixed this exact mismatch for the primary model). Left as-is
    # here since re-deriving the label is a separate fix from the validation
    # harness this pass addresses -- flagging it so it isn't mistaken for
    # resolved.
    df_feat['Target'] = np.where(df_feat[c_col].shift(-1) > df_feat[c_col] * 1.003, 1, 0)
    X = df_feat[FEATURES]
    y = df_feat['Target']
    return X, y


def purged_embargoed_splits(n_samples: int, n_splits: int = 5, embargo_bars: int = 5):
    """
    Purged & embargoed walk-forward splits, built on top of sklearn's
    TimeSeriesSplit (expanding-window, chronological -- never shuffled).

    - PURGE: the last `embargo_bars` samples of each training fold are
      dropped. These are the training rows closest to the test boundary,
      and since labels here are built from forward-looking price data
      (shift(-1)), a training label that close to the boundary may have
      been computed using price action that falls inside the test period.
    - EMBARGO: the first `embargo_bars` samples of each test fold are also
      dropped, so the test set doesn't start immediately adjacent to
      information the model may have partially seen.

    Yields (train_idx, test_idx) arrays, chronological, non-overlapping.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tscv.split(np.arange(n_samples)):
        if len(train_idx) > embargo_bars:
            train_idx = train_idx[:-embargo_bars]
        if len(test_idx) > embargo_bars:
            test_idx = test_idx[embargo_bars:]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        yield train_idx, test_idx


def run_purged_embargoed_cv(df: pd.DataFrame, n_splits: int = 5, embargo_bars: int = 5) -> dict:
    """
    Actually runs the purged & embargoed walk-forward CV the module docstring
    claims. Reports metrics PER FOLD (not just averaged) so you can see
    whether any edge is consistent across periods or only exists in one
    lucky fold -- same principle as backtester.walk_forward_validate.

    Returns {"folds": [...], "summary": {...}} or {"error": ...} if there
    isn't enough data for the requested split/embargo configuration.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    X, y = build_labeled_features(df)
    n = len(X)
    min_needed = (n_splits + 1) * (embargo_bars * 2 + 10)
    if n < min_needed:
        return {"error": f"Not enough data for {n_splits} purged/embargoed folds "
                          f"(need ~{min_needed} usable bars after feature warm-up, got {n})"}

    folds = []
    for fold_num, (train_idx, test_idx) in enumerate(purged_embargoed_splits(n, n_splits, embargo_bars), start=1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            # Degenerate fold (all one class) -- skip rather than report a
            # meaningless precision/recall number.
            continue

        model_tuple = train_institutional_ensemble(X_train, y_train, X_calib=None, y_calib=None)
        voting_clf, _ = model_tuple
        y_pred = voting_clf.predict(X_test)

        folds.append({
            "fold": fold_num,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        })

    if not folds:
        return {"error": "All folds were degenerate (single-class) -- cannot compute CV metrics on this data."}

    accs = [f["accuracy"] for f in folds]
    precs = [f["precision"] for f in folds]
    summary = {
        "num_valid_folds": len(folds),
        "accuracy_mean": round(float(np.mean(accs)), 4),
        "accuracy_std": round(float(np.std(accs)), 4),
        "precision_mean": round(float(np.mean(precs)), 4),
        "precision_std": round(float(np.std(precs)), 4),
    }
    return {"folds": folds, "summary": summary}


def train_institutional_ensemble(X_train: pd.DataFrame, y_train: pd.Series,
                                  X_calib: pd.DataFrame = None, y_calib: pd.Series = None):
    """
    Trains Quad-Model Heterogeneous Ensemble with Isotonic Calibration.

    FIX: the calibrator used to be fit on the ensemble's predictions on its
    OWN training data (voting_clf.predict_proba(X_train) then calibrate
    against y_train) -- a model is always over-confident on data it was
    trained on, so that made the calibration curve look better than it
    actually is. If X_calib/y_calib are provided (a held-out, chronologically
    later slice), the calibrator is fit on those out-of-sample predictions
    instead, which is what isotonic calibration is supposed to be validated
    against. Falls back to in-sample calibration (with a warning) only if no
    calibration set is given, e.g. when a caller has too little data to
    afford a 3-way split.
    """
    xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.03, random_state=42, eval_metric='logloss')
    lgb = LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.03, random_state=42, verbose=-1)
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    
    estimators = [('xgb', xgb), ('lgb', lgb), ('rf', rf)]
    
    if CATBOOST_AVAILABLE:
        cb = CatBoostClassifier(iterations=100, depth=4, learning_rate=0.03, verbose=0, random_seed=42)
        estimators.append(('cb', cb))
        
    voting_clf = VotingClassifier(estimators=estimators, voting='soft')
    voting_clf.fit(X_train, y_train)

    if X_calib is not None and y_calib is not None and len(X_calib) > 0:
        calib_probs = voting_clf.predict_proba(X_calib)[:, 1]
        calib_targets = y_calib
    else:
        logger.warning(
            "train_institutional_ensemble: no held-out calibration set provided -- "
            "calibrating on in-sample predictions, which will look more confident "
            "than the model actually is out-of-sample."
        )
        calib_probs = voting_clf.predict_proba(X_train)[:, 1]
        calib_targets = y_train

    iso_calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso_calibrator.fit(calib_probs, calib_targets)
    
    return voting_clf, iso_calibrator


def predict_calibrated_win_probability(model_tuple, feature_vector: pd.DataFrame) -> float:
    """
    Predicts recalibrated physical win probability (0.0 to 1.0).
    """
    try:
        voting_clf, iso_calibrator = model_tuple
        raw_prob = voting_clf.predict_proba(feature_vector)[:, 1][0]
        calibrated_prob = iso_calibrator.predict([raw_prob])[0]
        return float(np.clip(calibrated_prob, 0.0, 1.0))
    except Exception:
        # Previously silent. A caller receiving 0.55 with no log has no way
        # to tell "the model genuinely scored this near coin-flip" apart
        # from "the model call broke and this is a made-up fallback."
        logger.warning("predict_calibrated_win_probability failed; returning fallback 0.55", exc_info=True)
        return 0.55  # Fallback baseline probability -- see warning above


def train_and_get_ensemble_model(df: pd.DataFrame, run_validation: bool = True,
                                  calib_fraction: float = 0.15, embargo_bars: int = 5):
    """
    XGBoost + LightGBM + RandomForest Ensemble Classifier Training with Calibration.

    FIX: this used to fit on the entire dataset with no holdout whatsoever --
    the "walk-forward validation" the module docstring claimed didn't exist
    anywhere in the code. Now:
      1. (if run_validation) runs real purged & embargoed walk-forward CV
         across the whole df and returns per-fold metrics, so you can see
         whether there's a consistent signal or not BEFORE trusting the
         final model.
      2. Trains the final deployed model on a chronological train/calibration
         split (not the full dataset), with an embargo gap purged out of the
         training tail, and calibrates on the held-out calibration slice
         instead of in-sample.

    Returns (model_tuple, features, validation_report). validation_report is
    None if run_validation=False or there wasn't enough data to run it.
    """
    df_feat_check = compute_advanced_institutional_features(df)
    if len(df_feat_check) < 50:
        return None, None, None

    validation_report = None
    if run_validation:
        validation_report = run_purged_embargoed_cv(df, n_splits=5, embargo_bars=embargo_bars)
        if "error" in validation_report:
            logger.warning("Walk-forward CV skipped: %s", validation_report["error"])

    X, y = build_labeled_features(df)
    n = len(X)

    # Chronological train/calibration split with an embargo gap purged from
    # the training tail (same reasoning as purged_embargoed_splits above).
    calib_size = max(int(n * calib_fraction), 1)
    split_point = n - calib_size
    train_idx = np.arange(0, max(split_point - embargo_bars, 1))
    calib_idx = np.arange(split_point, n)

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_calib, y_calib = X.iloc[calib_idx], y.iloc[calib_idx]

    if y_train.nunique() < 2:
        logger.error("train_and_get_ensemble_model: training split has only one class, cannot fit.")
        return None, None, validation_report

    model_tuple = train_institutional_ensemble(
        X_train, y_train,
        X_calib=X_calib if y_calib.nunique() > 1 else None,
        y_calib=y_calib if y_calib.nunique() > 1 else None,
    )
    return model_tuple, FEATURES, validation_report


def predict_signal_probability(model_tuple, features_list, current_row_df):
    """Predicts calibrated probability Score P(Win)"""
    if model_tuple is None:
        return 0.50
    try:
        if isinstance(model_tuple, tuple):
            X_curr = current_row_df[features_list]
            return predict_calibrated_win_probability(model_tuple, X_curr)
        else:
            X_curr = current_row_df[features_list]
            prob_win = model_tuple.predict_proba(X_curr)[0][1]
            return round(float(prob_win), 4)
    except Exception as e:
        logger.error(f"Prediction Error: {e}", exc_info=True)
        return 0.50