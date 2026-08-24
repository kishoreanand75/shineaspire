import unittest

import numpy as np
import pandas as pd

from signal_engine import build_features, decide_from_row


class FakeModel:
    def predict_proba(self, features):
        return np.array([[0.05, 0.10, 0.85]])


class SignalGateTests(unittest.TestCase):
    def test_confidence_above_threshold_should_signal_without_extra_structure_gate(self):
        rows = []
        base = 100.0
        for i in range(80):
            close = base + np.sin(i / 5.0) * 2.0 + i * 0.02
            rows.append(
                {
                    "Open": close - 0.3,
                    "High": close + 0.9,
                    "Low": close - 0.8,
                    "Close": close,
                    "Volume": 1000,
                }
            )

        df = pd.DataFrame(rows)
        df.index = pd.date_range("2024-01-01", periods=len(df), freq="1h")
        df = build_features(df)

        result = decide_from_row(df, len(df) - 1, asset_symbol="BTCUSDT", model=FakeModel(), htf_trend="BULLISH")

        self.assertEqual(result["signal"], "BUY_CALL", result)
        self.assertGreaterEqual(result["confidence"], 0.55, result)


if __name__ == "__main__":
    unittest.main()
