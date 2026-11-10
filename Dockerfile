# Container image for the scoring API (written for cloud deployment; see docs/DEPLOYMENT.md).
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Optional: RUN pip install --no-cache-dir xgboost

COPY fraud_detection ./fraud_detection

# Train at build time so the image contains a model, the reports and the warm-start data.
RUN python -m fraud_detection.train && python -m fraud_detection.benchmark --requests 300 && python -m fraud_detection.report

RUN useradd --create-home appuser && mkdir -p runtime && chown -R appuser /app
USER appuser

EXPOSE 8000
# Hosts such as Render set $PORT; it defaults to 8000 elsewhere.
# The API refuses to listen on 0.0.0.0 unless FRAUD_API_KEYS is set, e.g.
#   docker run -p 8000:8000 -e FRAUD_API_KEYS="adminkey:admin,analystkey:analyst,appkey:customer" <image>
CMD ["sh", "-c", "python -m fraud_detection.api --host 0.0.0.0 --port ${PORT:-8000}"]
