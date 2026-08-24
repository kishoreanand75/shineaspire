"""Backward-compatible Bitcoin signal entry point."""

from signal_engine import generate_signal as _generate_signal


def generate_signal(df):
    """Generate a BTC signal using the shared live/backtest signal engine."""
    df = df.rename(columns={column: column.title() for column in df.columns})
    result = _generate_signal(df, asset_symbol="BTCUSDT")
    return result["signal"], float(df.iloc[-1]["Close"])
