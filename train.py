"""Train fraud models, evaluate them, and initialize the SQLite database."""

import sqlite3
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from reasons import risk_factors


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "transactions.csv"
DATABASE_PATH = BASE_DIR / "fraud.db"
MODEL_PATH = BASE_DIR / "model.joblib"


def risk_level(score):
    """Map a 0-100 score to the project's standard risk bands."""
    if score >= 70:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def create_database(df, scored):
    """Replace the prototype's generated data and scores in SQLite."""
    cities = np.random.default_rng(42).choice(
        ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad", "Pune"],
        size=1500,
    )
    customer_averages = df.groupby("customer_id")["amount"].mean()

    with sqlite3.connect(DATABASE_PATH) as con:
        con.executescript(
            """
            DROP TABLE IF EXISTS alerts;
            DROP TABLE IF EXISTS risk_scores;
            DROP TABLE IF EXISTS transactions;
            DROP TABLE IF EXISTS customers;

            CREATE TABLE customers (
                customer_id INTEGER PRIMARY KEY,
                average_amount REAL NOT NULL,
                home_city TEXT NOT NULL
            );
            CREATE TABLE transactions (
                txn_id INTEGER PRIMARY KEY,
                customer_id INTEGER,
                amount REAL NOT NULL,
                hour INTEGER NOT NULL,
                merchant_type TEXT NOT NULL,
                channel TEXT NOT NULL,
                location_mismatch INTEGER NOT NULL,
                new_device INTEGER NOT NULL,
                txns_last_hour INTEGER NOT NULL,
                amount_ratio REAL NOT NULL,
                is_new_payee INTEGER NOT NULL,
                is_fraud INTEGER
            );
            CREATE TABLE risk_scores (
                txn_id INTEGER PRIMARY KEY REFERENCES transactions(txn_id),
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                reasons TEXT NOT NULL
            );
            CREATE TABLE alerts (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                txn_id INTEGER NOT NULL REFERENCES transactions(txn_id),
                status TEXT NOT NULL,
                reviewed_by TEXT,
                action_time TEXT
            );
            CREATE INDEX idx_alerts_status ON alerts(status);
            """
        )

        con.executemany(
            "INSERT INTO customers (customer_id, average_amount, home_city) VALUES (?, ?, ?)",
            [
                (int(customer_id), float(average), str(cities[int(customer_id) - 1]))
                for customer_id, average in customer_averages.items()
            ],
        )
        con.executemany(
            """
            INSERT INTO transactions VALUES (
                :txn_id, :customer_id, :amount, :hour, :merchant_type, :channel,
                :location_mismatch, :new_device, :txns_last_hour, :amount_ratio,
                :is_new_payee, :is_fraud
            )
            """,
            df.to_dict(orient="records"),
        )
        con.executemany(
            "INSERT INTO risk_scores (txn_id, risk_score, risk_level, reasons) VALUES (?, ?, ?, ?)",
            list(scored[["txn_id", "risk_score", "risk_level", "reasons"]].itertuples(index=False, name=None)),
        )
        con.executemany(
            "INSERT INTO alerts (txn_id, status) VALUES (?, 'Pending review')",
            [(int(txn_id),) for txn_id in scored.loc[scored["risk_level"] == "High", "txn_id"]],
        )


def main():
    df = pd.read_csv(DATA_PATH)
    feature_columns = ["txn_id", "customer_id", "is_fraud"]
    X = pd.get_dummies(
        df.drop(columns=feature_columns),
        columns=["merchant_type", "channel"],
        dtype=int,
    )
    y = df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42
        ),
    }

    results = []
    predictions = {}
    best_model = None
    best_auc = -1
    best_name = None
    for name, model in models.items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        predicted_labels = model.predict(X_test)
        auc = roc_auc_score(y_test, probabilities)
        results.append(
            {
                "model": name,
                "precision": precision_score(y_test, predicted_labels, zero_division=0),
                "recall": recall_score(y_test, predicted_labels, zero_division=0),
                "f1": f1_score(y_test, predicted_labels, zero_division=0),
                "roc_auc": auc,
            }
        )
        predictions[name] = probabilities
        print(
            f"{name}: precision={results[-1]['precision']:.3f}, "
            f"recall={results[-1]['recall']:.3f}, "
            f"F1={results[-1]['f1']:.3f}, ROC-AUC={auc:.3f}"
        )
        if auc > best_auc:
            best_model, best_auc, best_name = model, auc, name

    metrics = pd.DataFrame(results)
    metrics.to_csv(BASE_DIR / "model_metrics.csv", index=False)
    print(f"Best model by ROC-AUC: {best_name}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        best_model, X_test, y_test, display_labels=["Legitimate", "Fraud"], ax=ax
    )
    ax.set_title(f"Confusion Matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(BASE_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, probabilities in predictions.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
        auc = metrics.loc[metrics["model"] == name, "roc_auc"].iloc[0]
        ax.plot(false_positive_rate, true_positive_rate, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curve",
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(BASE_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)

    bundle = {"model": best_model, "columns": list(X.columns), "name": best_name}
    joblib.dump(bundle, MODEL_PATH)

    scores = (best_model.predict_proba(X)[:, 1] * 100).round(1)
    scored = df[["txn_id", "customer_id"]].copy()
    scored["risk_score"] = scores
    scored["risk_level"] = scored["risk_score"].map(risk_level)
    scored["reasons"] = df.apply(risk_factors, axis=1)
    create_database(df, scored)
    print("\nRisk levels:")
    print(scored["risk_level"].value_counts().to_string())
    print(f"Saved model, evaluation charts, and database under {BASE_DIR}")


if __name__ == "__main__":
    main()
