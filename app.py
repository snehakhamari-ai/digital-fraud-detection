"""Streamlit analyst dashboard for the fraud-scoring prototype."""

import os
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "fraud.db"
API_URL = os.getenv("FRAUD_API_URL", "http://127.0.0.1:5000").rstrip("/")
if not API_URL.startswith(("http://", "https://")):
    API_URL = f"https://{API_URL}"
API_TIMEOUT = float(os.getenv("FRAUD_API_TIMEOUT_SECONDS", "5"))
REVIEW_API_KEY = os.getenv("FRAUD_REVIEW_API_KEY")
REQUIRE_LOGIN = os.getenv("FRAUD_REQUIRE_LOGIN", "").lower() == "true"
MERCHANT_TYPES = ["grocery", "fuel", "food", "shopping", "travel", "gaming", "crypto"]
CHANNELS = ["UPI", "card", "wallet"]
RISK_LEVELS = ["Low", "Medium", "High"]

st.set_page_config(
    page_title="Fraud Risk Command Center",
    page_icon="💳",
    layout="wide",
)
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: #17263a;
        border: 1px solid #263c57;
        padding: 1rem 1.2rem;
        border-radius: 12px;
    }
    .risk-box {
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border: 1px solid;
        margin: .5rem 0 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not DATABASE_PATH.exists():
    st.error("fraud.db is missing. Run `python train.py` before starting the dashboard.")
    st.stop()

connection = sqlite3.connect(DATABASE_PATH)
try:
    data = pd.read_sql_query(
        """
        SELECT t.*, r.risk_score, r.risk_level, r.reasons
        FROM transactions t
        JOIN risk_scores r ON r.txn_id = t.txn_id
        """,
        connection,
    )
    model_path = BASE_DIR / "model_metrics.csv"
    model_metrics = pd.read_csv(model_path) if model_path.exists() else pd.DataFrame()
finally:
    connection.close()

if data.empty:
    st.error("No scored transactions are available. Run `python train.py` to initialize the data.")
    st.stop()

with st.sidebar:
    st.title("Analyst workspace")
    if "analyst" not in st.session_state:
        st.session_state.analyst = None

    if st.session_state.analyst:
        st.success(f"Signed in as {st.session_state.analyst}")
        if st.button("Log out", width="stretch"):
            st.session_state.analyst = None
            st.rerun()
    else:
        with st.form("analyst_login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_clicked = st.form_submit_button("Analyst login", width="stretch")
        if login_clicked:
            expected_username = os.getenv(
                "FRAUD_ANALYST_USERNAME", "" if REQUIRE_LOGIN else "analyst"
            )
            expected_password = os.getenv(
                "FRAUD_ANALYST_PASSWORD", "" if REQUIRE_LOGIN else "fraud-demo"
            )
            if username == expected_username and password == expected_password:
                st.session_state.analyst = username
                st.rerun()
            else:
                st.error("Invalid username or password.")

    st.divider()
    st.subheader("Transaction filters")
    medium_threshold = st.slider("Medium risk starts at", 1, 99, 40)
    high_threshold = st.slider(
        "High risk starts at",
        medium_threshold + 1,
        100,
        max(70, medium_threshold + 1),
    )
    risk_filter = st.multiselect("Risk level", RISK_LEVELS, default=RISK_LEVELS)
    merchant_filter = st.multiselect(
        "Merchant type", MERCHANT_TYPES, default=MERCHANT_TYPES
    )
    channel_filter = st.multiselect("Channel", CHANNELS, default=CHANNELS)
    amount_min = float(data["amount"].min())
    amount_max = float(data["amount"].max())
    amount_range = st.slider(
        "Amount range",
        min_value=amount_min,
        max_value=amount_max,
        value=(amount_min, amount_max),
        step=max((amount_max - amount_min) / 100, 0.01),
        format="Rs %.2f",
    )

data["display_level"] = data["risk_score"].map(
    lambda score: "High"
    if score >= high_threshold
    else "Medium"
    if score >= medium_threshold
    else "Low"
)
filtered = data[
    data["display_level"].isin(risk_filter)
    & data["merchant_type"].isin(merchant_filter)
    & data["channel"].isin(channel_filter)
    & data["amount"].between(amount_range[0], amount_range[1])
]
st.sidebar.metric(
    "High-risk transactions in view",
    int((filtered["display_level"] == "High").sum()),
)

st.title("Digital Payment Fraud Detection")
st.caption("Payment monitoring · Risk scoring · Analyst review")
dashboard_tab, model_tab = st.tabs(["Dashboard", "Model results"])

with dashboard_tab:
    card1, card2, card3 = st.columns(3)
    card1.metric("Total transactions", f"{len(data):,}")
    card2.metric(
        "High-risk alerts",
        f"{int((data['display_level'] == 'High').sum()):,}",
    )
    card3.metric("Average risk score", f"{data['risk_score'].mean():.1f} / 100")

    chart_column, table_column = st.columns([1, 2])
    with chart_column:
        st.subheader("Risk levels")
        level_counts = (
            filtered["display_level"]
            .value_counts()
            .reindex(RISK_LEVELS, fill_value=0)
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(
            level_counts.index,
            level_counts.values,
            color=["#2ebd85", "#f0ad4e", "#ef5350"],
        )
        ax.set_ylabel("Transactions")
        ax.spines[["top", "right"]].set_visible(False)
        ax.bar_label(bars, padding=3)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with table_column:
        st.subheader("Top 25 riskiest transactions")
        top = filtered.sort_values("risk_score", ascending=False).head(25)
        display_columns = [
            "txn_id",
            "amount",
            "merchant_type",
            "channel",
            "risk_score",
            "display_level",
            "reasons",
        ]

        def badge_color(value):
            colors = {
                "High": "background-color: #7f2528; color: #fff;",
                "Medium": "background-color: #765117; color: #fff;",
                "Low": "background-color: #18543e; color: #fff;",
            }
            return colors.get(value, "")

        styled_top = top[display_columns].style.apply(
            lambda column: column.map(badge_color),
            subset=["risk_score", "display_level"],
        ).format({"amount": "Rs {:,.2f}", "risk_score": "{:.1f}"})
        st.dataframe(styled_top, width="stretch", hide_index=True)

    flagged = filtered[filtered["display_level"] != "Low"]
    st.download_button(
        "Download flagged transactions as CSV",
        data=flagged.to_csv(index=False).encode("utf-8"),
        file_name="flagged_transactions.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Pending alerts")
    try:
        alerts_response = requests.get(
            f"{API_URL}/alerts", timeout=API_TIMEOUT
        )
        alerts_response.raise_for_status()
        alerts = alerts_response.json()
    except (requests.RequestException, ValueError) as error:
        alerts = []
        st.error(f"Could not load alerts from the Flask API: {error}")

    if not alerts:
        st.info("There are no pending alerts.")
    for alert in alerts:
        with st.container(border=True):
            info, approve_col, block_col, review_col = st.columns([5, 1, 1, 1])
            info.markdown(
                f"**Alert {alert['alert_id']} · Transaction {alert['txn_id']}**  \n"
                f"Score **{alert['risk_score']:.1f}** · {alert['merchant_type']} · "
                f"{alert['channel']} · Rs {alert['amount']:,.2f}  \n"
                f"{alert['reasons']}"
            )
            actions = [
                (approve_col, "Approve", "Approved"),
                (block_col, "Block", "Blocked"),
                (review_col, "Under review", "Under review"),
            ]
            for column, label, action in actions:
                if column.button(
                    label,
                    key=f"alert_{alert['alert_id']}_{action}",
                    disabled=not st.session_state.analyst,
                    width="stretch",
                ):
                    try:
                        response = requests.post(
                            f"{API_URL}/alerts/{alert['alert_id']}/action",
                            json={
                                "action": action,
                                "reviewed_by": st.session_state.analyst,
                            },
                            headers={"X-Reviewer-Key": REVIEW_API_KEY or ""},
                            timeout=API_TIMEOUT,
                        )
                        response.raise_for_status()
                        st.success(f"Alert {alert['alert_id']} marked {action}.")
                        st.rerun()
                    except (requests.RequestException, ValueError) as error:
                        st.error(f"Could not update alert {alert['alert_id']}: {error}")
    if not st.session_state.analyst:
        st.caption("Sign in from the sidebar to review or update alerts.")

    st.divider()
    st.subheader("Check a new transaction")
    with st.form("score_transaction"):
        amount_col, activity_col, context_col = st.columns(3)
        amount = amount_col.number_input(
            "Amount (Rs)", min_value=0.01, max_value=500000.0, value=500.0
        )
        usual_amount = amount_col.number_input(
            "Customer's usual amount (Rs)",
            min_value=0.01,
            max_value=500000.0,
            value=500.0,
        )
        hour = amount_col.slider("Hour of day", 0, 23, 14)
        txns_last_hour = activity_col.slider("Payments in the last hour", 0, 20, 1)
        merchant_type = activity_col.selectbox("Merchant type", MERCHANT_TYPES)
        channel = activity_col.selectbox("Channel", CHANNELS)
        new_device = context_col.checkbox("New device")
        location_mismatch = context_col.checkbox("Location mismatch")
        is_new_payee = context_col.checkbox("First payment to this payee")
        submitted = st.form_submit_button("Get risk score", type="primary", width="stretch")

    if submitted:
        transaction = {
            "amount": amount,
            "usual_amount": usual_amount,
            "hour": hour,
            "txns_last_hour": txns_last_hour,
            "merchant_type": merchant_type,
            "channel": channel,
            "new_device": int(new_device),
            "location_mismatch": int(location_mismatch),
            "is_new_payee": int(is_new_payee),
        }
        try:
            response = requests.post(
                f"{API_URL}/score", json=transaction, timeout=API_TIMEOUT
            )
            response.raise_for_status()
            result = response.json()
            colors = {
                "High": ("#7f2528", "#ef5350"),
                "Medium": ("#765117", "#f0ad4e"),
                "Low": ("#18543e", "#2ebd85"),
            }
            background, accent = colors[result["risk_level"]]
            st.markdown(
                f"""
                <div class="risk-box" style="background:{background}; border-color:{accent};">
                  <strong>{result['risk_level']} risk</strong>
                  &nbsp;·&nbsp; Score <strong>{result['risk_score']:.1f} / 100</strong>
                  <br><br>{result['reasons']}
                </div>
                """,
                unsafe_allow_html=True,
            )
        except (requests.RequestException, ValueError, KeyError) as error:
            st.error(f"Could not score this transaction: {error}")

with model_tab:
    st.subheader("Model comparison")
    if model_metrics.empty:
        st.error("Model results are missing. Run `python train.py`.")
    else:
        st.dataframe(
            model_metrics.style.format(
                {"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}", "roc_auc": "{:.3f}"}
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption("The model with the highest ROC-AUC is used by the scoring API.")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        confusion_path = BASE_DIR / "confusion_matrix.png"
        if confusion_path.exists():
            st.image(str(confusion_path), caption="Confusion matrix for the selected model")
        else:
            st.info("Confusion matrix not found. Run `python train.py`.")
    with chart_right:
        roc_path = BASE_DIR / "roc_curve.png"
        if roc_path.exists():
            st.image(str(roc_path), caption="ROC curves on the stratified test set")
        else:
            st.info("ROC curve not found. Run `python train.py`.")
