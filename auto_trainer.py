"""Background daily model retraining for paper-data collection."""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import config


STATE_FILE = "auto_training_state.json"
LOG_FILE = "auto_training.log"
MODEL_FILE = "xgboost_model.json"
_process = None


def _load_state():
    try:
        with open(STATE_FILE, encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def _save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


def maybe_retrain():
    """Start at most one daily background training job and report its state."""
    global _process
    if not config.AUTO_RETRAIN_ENABLED:
        return "disabled"
    if _process is not None and _process.poll() is None:
        return "running"
    state = _load_state()
    now = datetime.now(timezone.utc)
    last_started = _parse_datetime(state.get("last_started"))
    retrying = state.get("status") in ("failed", "error")
    wait_seconds = (
        config.AUTO_RETRAIN_RETRY_MINUTES * 60
        if retrying else config.AUTO_RETRAIN_INTERVAL_HOURS * 3600
    )
    if last_started and (now - last_started).total_seconds() < wait_seconds:
        return "waiting"
    try:
        log = open(LOG_FILE, "a", encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        attempt = int(state.get("attempt", 0)) + 1
        environment["TRAINING_SEED"] = str(41 + attempt)
        _process = subprocess.Popen(
            [sys.executable, "train_model.py"], stdout=log, stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.abspath(__file__)), env=environment,
        )
        _save_state({"last_started": now.isoformat(), "status": "running", "pid": _process.pid, "attempt": attempt})
        return "started"
    except OSError as exc:
        _save_state({"last_started": now.isoformat(), "status": "error", "error": str(exc)})
        return "error"


def training_finished():
    """Refresh state after a background job exits; caller can reload the model."""
    global _process
    if _process is None or _process.poll() is None:
        return False
    code = _process.returncode
    state = _load_state()
    state.update({"status": "completed" if code == 0 else "failed", "returncode": code})
    if code == 0 and os.path.exists(MODEL_FILE):
        state["model_updated_at"] = datetime.fromtimestamp(
            os.path.getmtime(MODEL_FILE), timezone.utc
        ).isoformat()
    _save_state(state)
    _process = None
    return True


def _parse_datetime(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None