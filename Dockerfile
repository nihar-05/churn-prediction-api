# ── Stage 1: builder ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ───────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy app files
COPY app.py .
COPY churn_prediction_model.pkl .

# Non-root user for security
RUN useradd -m appuser
USER appuser

EXPOSE 8000

ENV MODEL_PATH=churn_prediction_model.pkl

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
