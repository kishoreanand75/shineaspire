"""cPanel Passenger entry point for the live Bitcoin dashboard."""

import os
from pathlib import Path

from flask import Flask, jsonify, send_from_directory


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

import live_server  # noqa: E402


application = Flask(__name__)


@application.get("/")
def dashboard():
    return send_from_directory(ROOT, "live_dashboard.html")


@application.get("/api/snapshot")
def snapshot():
    try:
        return jsonify(live_server.get_snapshot())
    except Exception:
        return jsonify({
            "status": "connecting",
            "message": "Live analysis service is temporarily unavailable.",
        }), 503


@application.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    application.run()