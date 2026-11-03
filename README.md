# Cloud Based Fraud Detection System

Knowledge Demonstration project - B.Tech Information Technology, Excel Engineering College (Autonomous), Komarapalayam.

A machine-learning platform that scores payment transactions for fraud in real time, applies business rules,
raises alerts into an analyst case queue, and reports model performance on a dashboard. It implements every
module described in the project report: data ingestion, preprocessing, feature engineering with class-imbalance
handling (SMOTE), model training and comparison (Logistic Regression, Random Forest, gradient boosting,
Isolation Forest), a REST scoring API, a rules engine, alerting and case management, and analytics.

> **Read this first - the data is synthetic.** Real transaction data cannot be redistributed, so the project
> ships a generator (`fraud_detection/data.py`) that produces a labelled, imbalanced, deliberately messy stream of
> card/UPI transactions. The numbers below show that the pipeline works end to end; they are **not** the accuracy
> a bank would see. To use real data, replace `generate_transactions()` with a loader that returns the same columns.

## Quick start

```bash
pip install -r requirements.txt        # numpy, pandas, scikit-learn, joblib, threadpoolctl
python -m fraud_detection.train        # generate data, train, evaluate, save model + reports/metrics.json (~20 s)
python -m fraud_detection.benchmark    # measure scoring latency -> reports/latency.json
python -m fraud_detection.report       # build reports/dashboard.html (open it in a browser)
python -m unittest discover -s tests   # 15 tests
python -m fraud_detection.api          # start the API on http://127.0.0.1:8000
```

Optional: `pip install xgboost` - XGBoost is then trained and compared automatically; without it,
scikit-learn's `HistGradientBoostingClassifier` takes its place (this is what produced the results below).

## Using the API

```bash
# score a transaction (customer history is built up from earlier requests)
curl -X POST localhost:8000/score -H "Content-Type: application/json" -d '{
  "customer_id": "C00010", "amount": 450000, "city": "Mumbai", "device_id": "BRAND-NEW",
  "merchant_category": "jewellery", "channel": "online", "timestamp": "2026-11-03T03:05:00"}'
# -> fraud_probability, risk_level (low/medium/high), decision (approve/review/block), reasons, case_id

curl "localhost:8000/cases?status=open"                                        # analyst case queue
curl -X POST localhost:8000/cases/1/resolve -H "Content-Type: application/json" \
     -d '{"status": "confirmed_fraud", "analyst": "divya"}'                    # feedback for retraining
```

| Endpoint | Purpose | Roles |
|---|---|---|
| `GET /health` | liveness, model version | none |
| `POST /score` | score one transaction | customer, analyst, admin |
| `GET /cases`, `POST /cases/<id>/resolve` | case queue and analyst decisions | analyst, admin |
| `GET /dashboard` | analytics dashboard | analyst, admin |
| `GET /metrics` | training and evaluation metrics | admin |

Access control: `FRAUD_API_KEYS="adminkey:admin,analystkey:analyst,appkey:customer"` and send `X-API-Key`.
Without the variable the API runs in open development mode and refuses to listen on a non-loopback address.

## How it maps to the report

| Report module | Code |
|---|---|
| Data collection and ingestion | `data.py` (generator), `api.py` (`POST /score`) |
| Data preprocessing | `preprocess.py` (`clean_transactions`, scaling and one-hot encoding) |
| Feature engineering and imbalance handling | `features.py` (frequency, average spend, time since last, location and device change), `preprocess.py` (`smote`) |
| Model training and selection | `modeling.py` (model comparison, validation-based selection, model registry in `models/`) |
| Real-time fraud scoring | `scoring.py`, `api.py` |
| Rules and risk decision | `rules.py` |
| Alert and case management | `cases.py` (SQLite case queue, alert outbox) |
| Analytics dashboard and reports | `report.py` -> `reports/dashboard.html` |
| Cloud hosting | `Dockerfile`, `docs/DEPLOYMENT.md` |

## Results (synthetic data, test split)

80,240 raw transactions, 2.0 % fraud. Chronological split: train 47,897 / validation 11,974 / test 19,958
(376 fraud cases). Models were trained on SMOTE-balanced training data; the best model was chosen on the
validation split and thresholds were tuned on validation, so the test split was never used for any decision.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.987 | 0.753 | 0.447 | 0.561 | 0.959 |
| Random Forest | 0.993 | 0.879 | 0.734 | 0.800 | 0.982 |
| **Gradient Boosting (HistGB)** - selected | 0.993 | 0.884 | 0.729 | 0.799 | 0.977 |
| Isolation Forest (unsupervised) | 0.969 | 0.277 | 0.418 | 0.333 | 0.888 |
| Fixed rules (baseline) | 0.970 | 0.210 | 0.223 | 0.216 | 0.604 |

Selected model on the test split: 274 fraud cases caught, 102 missed, 36 false alarms among 19,582 genuine transactions.
Scoring latency (local, 2 CPUs, one process): 7.2 ms mean and 10.4 ms p95 over HTTP, 5.9 ms mean in process.

## Honest limitations

* **Synthetic data.** Fraud in the generator arrives in bursts and often at night, so velocity features
  (time since last transaction) dominate the model. Real fraud will be harder and less regular.
* **SMOTE helped only a little here.** Without it the same model scores F1 0.792 (precision 0.821, recall 0.766)
  against 0.799 with it (precision 0.884, recall 0.729); it traded some recall for precision. See the dashboard.
* **XGBoost was not run** to produce these results (it could not be installed in the build environment).
  The code uses it automatically when it is installed; re-run training if you want it in the comparison.
* **Latency is local**, from sequential requests on a small machine. A cloud deployment adds network time.
* **Not deployed to a cloud account.** The Dockerfile and deployment notes are provided but untested here.
* **Alerts are written to a file** (`runtime/alerts_outbox.jsonl`); sending real e-mail/SMS needs a provider.
* **Access control is API-key based** with three roles. JWT sessions, user registration and a multi-user web UI
  are future work, as are deep models (LSTM, autoencoders), graph-based fraud-ring detection, Kafka streaming and explainable AI.
* Customer profiles are held in memory and cases in SQLite, so run one instance (see `docs/DEPLOYMENT.md`).

## Project layout

```
fraud_detection/   features.py data.py preprocess.py modeling.py rules.py cases.py scoring.py api.py report.py benchmark.py train.py
tests/             unit and end-to-end tests (feature maths, SMOTE, time split, rules, case queue, API auth and flow)
reports/           metrics.json, curves.json, latency.json, dashboard.html  (committed so results can be checked)
docs/              DEPLOYMENT.md
data/ models/ runtime/   created when you run the project (not committed)
```
