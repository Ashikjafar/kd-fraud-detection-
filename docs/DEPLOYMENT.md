# Deploying to the cloud

The project is cloud-ready (stateless scoring process, container image, API-key access control),
but **it has not been deployed to a cloud account as part of this repository**, and the Docker image
below has not been built in the environment where the code was written. Treat these steps as a
checklist to run and verify yourself.

## 1. Build and run the container locally

```bash
docker build -t fraud-detection .
docker run -p 8000:8000 \
  -e FRAUD_API_KEYS="adminkey:admin,analystkey:analyst,appkey:customer" \
  fraud-detection
curl localhost:8000/health
```

The image trains the model while it is built (about a minute), so no model file has to be stored in Git.

## 2. Push to a managed container service

Any of these work with the same image (all expose port 8000):

| Platform | Service | Notes |
|---|---|---|
| Google Cloud | Cloud Run | `gcloud run deploy fraud-detection --source . --port 8000 --set-env-vars FRAUD_API_KEYS=...` |
| AWS | App Runner / ECS Fargate | push the image to ECR, create a service on port 8000 |
| Azure | Container Apps | push the image to ACR, create a container app with ingress on 8000 |

Store `FRAUD_API_KEYS` in the platform's secret manager rather than on the command line.

## 3. What to change for real production use

* **State.** Customer profiles live in process memory and cases in a local SQLite file, so run a single
  instance per case database. For several instances, move profiles to Redis and cases to PostgreSQL
  (`CustomerProfileStore` and `CaseStore` are the two classes to replace).
* **Alerts.** `OutboxNotifier` only writes to a file. Replace `notify()` with an SMTP / SMS-gateway call.
* **Authentication.** API keys with three roles are provided; use JWT / an identity provider and TLS
  termination (the platforms above provide HTTPS) for production.
* **Scaling.** Scoring is single-threaded per process by design (about 6 ms per transaction on 2 CPUs);
  scale by adding instances behind the platform's load balancer.
* **Retraining.** `CaseStore.feedback_labels()` exports analyst decisions to merge into the next training run.
