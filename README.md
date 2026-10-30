# Digital Payment Fraud Detection & Risk Scoring

A prototype fraud analyst dashboard for synthetic UPI, card, and wallet transactions. It trains Logistic Regression and Random Forest models, chooses the better ROC-AUC model, serves scores through Flask, and provides an analyst review dashboard in Streamlit.

This web-based machine-learning prototype complements the repository's earlier Review I rule-based fraud-pattern project; the earlier project history remains in Git.

## Setup and run

From this directory, install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Generate the synthetic dataset, train the models, and initialize the SQLite database:

```powershell
python generate_data.py
python train.py
```

Start the Flask API in a first terminal:

```powershell
python api.py
```

Start the dashboard in a second terminal:

```powershell
python -m streamlit run app.py
```

The dashboard is available at `http://localhost:8501`; the API listens on `http://127.0.0.1:5000`. Run `train.py` again whenever you regenerate the dataset.

## Generated files

- `transactions.csv`: 20,000 synthetic transactions and fraud labels.
- `fraud.db`: `customers`, `transactions`, `risk_scores`, and `alerts` tables.
- `model.joblib`: selected model and the exact encoded feature-column list.
- `model_metrics.csv`: precision, recall, F1, and ROC-AUC for both models.
- `confusion_matrix.png` and `roc_curve.png`: test-set evaluation charts.

Fraud labels in new API-scored transactions are unknown and stored as `NULL`. Newly scored transactions with a High risk level are added to the pending alerts queue.

## API

`POST /score` accepts JSON using the dataset's feature names. Supply either `amount_ratio` or `usual_amount`; the API calculates the ratio when `usual_amount` is supplied. `customer_id` is optional for newly scored transactions.

```json
{
  "customer_id": 42,
  "amount": 4500,
  "usual_amount": 500,
  "hour": 2,
  "merchant_type": "crypto",
  "channel": "UPI",
  "location_mismatch": 1,
  "new_device": 1,
  "txns_last_hour": 4,
  "is_new_payee": 1
}
```

The response contains `txn_id`, `risk_score` (0-100), `risk_level`, and plain-English `reasons`. Risk levels are Low below 40, Medium from 40 to below 70, and High from 70 upward.

- `GET /alerts` returns alerts still awaiting review.
- `POST /alerts/<alert_id>/action` accepts `{"action":"Approved","reviewed_by":"analyst"}`. Actions are `Approved`, `Blocked`, and `Under review`.

## Analyst login

The prototype dashboard defaults to username `analyst` and password `fraud-demo`. Set `FRAUD_ANALYST_USERNAME` and `FRAUD_ANALYST_PASSWORD` in the environment to change them. These prototype credentials are not a production authentication system.

Set `FRAUD_API_URL` if the Flask API is hosted somewhere other than `http://127.0.0.1:5000`.

## Public demo deployment (Render)

The root `render.yaml` configures two free Render web services: one for the Flask API and one for the Streamlit dashboard. Each service generates its own synthetic dataset and model during the build. Free services can sleep when idle, and their local SQLite databases are ephemeral; demo transactions and alert decisions can be lost when a service restarts or redeploys. This setup is for a synthetic-data prototype, not real bank transactions.

1. Create a **new, separate public GitHub repository** for this project. Do not publish the existing `house-price-multi-agent-project` repository; the fraud project is currently inside that unrelated local repository.
2. Add the project source files, `.streamlit/config.toml`, `requirements.txt`, and `render.yaml` to the new repository. Do not upload `.env` files, virtual environments, SQLite databases, or generated model/data files; `.gitignore` excludes these.
3. In Render, choose **New → Blueprint**, connect the new repository, and apply the `render.yaml` configuration.
4. When prompted for secret environment values, set `FRAUD_ANALYST_USERNAME` and `FRAUD_ANALYST_PASSWORD` for `fraud-dashboard`. Set `FRAUD_REVIEW_API_KEY` on **both** `fraud-api` and `fraud-dashboard` to the same long, random value. These values must be entered as Render secrets, not committed to GitHub. Hosted login has no default demo credentials.
5. Once both services are live, share the `https://fraud-dashboard.onrender.com` URL. The API is used by the dashboard; it does not serve the UI.

Anyone with the dashboard link can view the synthetic demo and submit transactions for scoring. Only analysts who know the login credentials can use the dashboard's alert-review buttons. The API requires the separately configured review key for alert actions. Free-tier services may take a little while to wake after inactivity.
