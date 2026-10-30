"""Flask API for scoring payments and reviewing fraud alerts."""

import hmac
import math
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from reasons import risk_factors


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "fraud.db"
MODEL_PATH = BASE_DIR / "model.joblib"
MERCHANT_TYPES = {"grocery", "fuel", "food", "shopping", "travel", "gaming", "crypto"}
CHANNELS = {"UPI", "card", "wallet"}
ALERT_ACTIONS = {"Approved", "Blocked", "Under review"}

if not DATABASE_PATH.exists():
    raise FileNotFoundError(f"{DATABASE_PATH} does not exist. Run train.py first.")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"{MODEL_PATH} does not exist. Run train.py first.")

model_bundle = joblib.load(MODEL_PATH)
app = Flask(__name__)

REQUIRE_REVIEW_KEY = os.getenv("FRAUD_REQUIRE_REVIEW_KEY", "").lower() == "true"
REVIEW_API_KEY = os.getenv("FRAUD_REVIEW_API_KEY")


@contextmanager
def get_connection():
    con = sqlite3.connect(DATABASE_PATH)
    con.row_factory = sqlite3.Row
    try:
        with con:
            yield con
    finally:
        con.close()


def number_value(payload, name, minimum=None, maximum=None, integer=False):
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if integer and int(value) != value:
        raise ValueError(f"{name} must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")
    return int(value) if integer else float(value)


def transaction_features(payload):
    required = {
        "amount",
        "hour",
        "merchant_type",
        "channel",
        "location_mismatch",
        "new_device",
        "txns_last_hour",
        "is_new_payee",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}.")

    merchant_type = payload["merchant_type"]
    channel = payload["channel"]
    if not isinstance(merchant_type, str) or merchant_type not in MERCHANT_TYPES:
        raise ValueError("merchant_type must be one of: " + ", ".join(sorted(MERCHANT_TYPES)) + ".")
    if not isinstance(channel, str) or channel not in CHANNELS:
        raise ValueError("channel must be one of: " + ", ".join(sorted(CHANNELS)) + ".")

    features = {
        "amount": number_value(payload, "amount", minimum=0.01),
        "hour": number_value(payload, "hour", minimum=0, maximum=23, integer=True),
        "merchant_type": merchant_type,
        "channel": channel,
        "location_mismatch": number_value(
            payload, "location_mismatch", minimum=0, maximum=1, integer=True
        ),
        "new_device": number_value(payload, "new_device", minimum=0, maximum=1, integer=True),
        "txns_last_hour": number_value(payload, "txns_last_hour", minimum=0, integer=True),
        "is_new_payee": number_value(payload, "is_new_payee", minimum=0, maximum=1, integer=True),
    }

    if "amount_ratio" in payload:
        features["amount_ratio"] = number_value(payload, "amount_ratio", minimum=0.01)
    elif "usual_amount" in payload:
        usual_amount = number_value(payload, "usual_amount", minimum=0.01)
        features["amount_ratio"] = round(features["amount"] / usual_amount, 2)
    else:
        raise ValueError("Provide either amount_ratio or usual_amount.")

    customer_id = payload.get("customer_id")
    if customer_id is not None:
        features["customer_id"] = number_value(payload, "customer_id", minimum=1, integer=True)
    else:
        features["customer_id"] = None
    return features


def score_level(score):
    if score >= 70:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


@app.post("/score")
def score_transaction():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    try:
        features = transaction_features(payload)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    raw_features = pd.DataFrame(
        [{key: value for key, value in features.items() if key != "customer_id"}]
    )
    encoded = pd.get_dummies(raw_features, columns=["merchant_type", "channel"], dtype=int)
    encoded = encoded.reindex(columns=model_bundle["columns"], fill_value=0)
    score = round(float(model_bundle["model"].predict_proba(encoded)[0, 1]) * 100, 1)
    level = score_level(score)
    reasons = risk_factors(features)

    with get_connection() as con:
        cursor = con.execute(
            """
            INSERT INTO transactions (
                customer_id, amount, hour, merchant_type, channel, location_mismatch,
                new_device, txns_last_hour, amount_ratio, is_new_payee, is_fraud
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                features["customer_id"],
                features["amount"],
                features["hour"],
                features["merchant_type"],
                features["channel"],
                features["location_mismatch"],
                features["new_device"],
                features["txns_last_hour"],
                features["amount_ratio"],
                features["is_new_payee"],
            ),
        )
        txn_id = cursor.lastrowid
        con.execute(
            "INSERT INTO risk_scores (txn_id, risk_score, risk_level, reasons) VALUES (?, ?, ?, ?)",
            (txn_id, score, level, reasons),
        )
        if level == "High":
            con.execute(
                "INSERT INTO alerts (txn_id, status) VALUES (?, 'Pending review')",
                (txn_id,),
            )

    return jsonify(
        {
            "txn_id": txn_id,
            "risk_score": score,
            "risk_level": level,
            "reasons": reasons,
        }
    ), 201


@app.get("/alerts")
def pending_alerts():
    with get_connection() as con:
        rows = con.execute(
            """
            SELECT a.alert_id, a.txn_id, a.status, t.customer_id, t.amount,
                   t.merchant_type, t.channel, r.risk_score, r.risk_level, r.reasons
            FROM alerts a
            JOIN transactions t ON t.txn_id = a.txn_id
            JOIN risk_scores r ON r.txn_id = a.txn_id
            WHERE a.status = 'Pending review'
            ORDER BY r.risk_score DESC, a.alert_id
            """
        ).fetchall()
    return jsonify([dict(row) for row in rows])


@app.get("/health")
def health_check():
    with get_connection() as con:
        con.execute("SELECT 1").fetchone()
    return jsonify({"status": "ok"})


@app.post("/alerts/<int:alert_id>/action")
def action_alert(alert_id):
    if REQUIRE_REVIEW_KEY and not REVIEW_API_KEY:
        return jsonify({"error": "Alert review is not configured on this server."}), 503
    if REVIEW_API_KEY and not hmac.compare_digest(
        request.headers.get("X-Reviewer-Key", ""), REVIEW_API_KEY
    ):
        return jsonify({"error": "A valid reviewer key is required."}), 401

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    action = payload.get("action")
    reviewer = payload.get("reviewed_by")
    if not isinstance(action, str) or action not in ALERT_ACTIONS:
        return jsonify({"error": "action must be Approved, Blocked, or Under review."}), 400
    if not isinstance(reviewer, str) or not reviewer.strip():
        return jsonify({"error": "reviewed_by must be a non-empty string."}), 400

    with get_connection() as con:
        cursor = con.execute(
            """
            UPDATE alerts
            SET status = ?, reviewed_by = ?, action_time = CURRENT_TIMESTAMP
            WHERE alert_id = ?
            """,
            (action, reviewer.strip(), alert_id),
        )
        if cursor.rowcount == 0:
            return jsonify({"error": "Alert not found."}), 404
        row = con.execute(
            "SELECT alert_id, txn_id, status, reviewed_by, action_time FROM alerts WHERE alert_id = ?",
            (alert_id,),
        ).fetchone()
    return jsonify(dict(row))


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )
