# 📉 Telco Customer Churn Prediction API

> A production-ready REST API that predicts whether a telecom customer will churn, powered by a scikit-learn ML pipeline and served via FastAPI.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model](#model)
- [Project Structure](#project-structure)
- [Quick Start (Local)](#quick-start-local)
- [API Reference](#api-reference)
- [Docker](#docker)
- [Deploy on AWS (Free Tier)](#deploy-on-aws-free-tier)
- [GitHub Setup](#github-setup)

---

## Overview

This project wraps a trained churn prediction model in a FastAPI backend that accepts customer feature data and returns:

- **Churn prediction** (`Yes` / `No`)
- **Churn probability** (0–1 float)
- **Risk level** (`Low` / `Medium` / `High`)

It supports both single-customer and batch prediction endpoints.

---

## Dataset

**IBM Telco Customer Churn** — publicly available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

| Property | Value |
|----------|-------|
| Rows | 7,043 |
| Features | 20 |
| Target | `Churn` (Yes / No) |
| Churn rate | ~26.5% |

**Key features:**

| Feature | Type | Description |
|---------|------|-------------|
| `tenure` | int | Months as a customer |
| `Contract` | categorical | Month-to-month / One year / Two year |
| `MonthlyCharges` | float | Monthly bill amount |
| `TotalCharges` | float | Cumulative charges |
| `InternetService` | categorical | DSL / Fiber optic / None |
| `PaymentMethod` | categorical | Payment type |
| `TechSupport`, `OnlineSecurity`, etc. | categorical | Add-on services |

---

## Model

The pickle file (`churn_prediction_model.pkl`) contains a **scikit-learn Pipeline** that includes:

1. **Preprocessing** — column transformers for categorical encoding and numerical scaling
2. **SMOTE** (via `imbalanced-learn`) / **RandomUnderSampler** — to handle the class imbalance (~73% No / ~26% Yes)
3. **Classifier** — trained binary classifier (e.g., Random Forest / Gradient Boosting)

The pipeline was trained on the IBM Telco dataset after standard EDA and feature engineering.

---

## Project Structure

```
churn-prediction-api/
├── app.py                        # FastAPI application
├── churn_prediction_model.pkl    # Trained sklearn pipeline
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container definition
├── .dockerignore
└── README.md
```

---

## Quick Start (Local)

### Prerequisites
- Python 3.11+
- pip

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/churn-prediction-api.git
cd churn-prediction-api

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the API
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Visit **http://localhost:8000/docs** for the interactive Swagger UI.

---

## API Reference

### `GET /`
Health check — returns a welcome message.

### `GET /health`
Returns model load status.

```json
{ "status": "ok", "model_loaded": true }
```

### `POST /predict`
Predict churn for a single customer.

**Request body:**
```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 358.20
}
```

**Response:**
```json
{
  "churn_prediction": "Yes",
  "churn_probability": 0.7421,
  "risk_level": "High"
}
```

### `POST /predict/batch`
Send up to 100 customer objects in an array. Returns an array of prediction responses.

---

## Docker

### Build the image

```bash
docker build -t churn-api:latest .
```

### Run the container

```bash
docker run -p 8000:8000 churn-api:latest
```

### Test it

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "DSL",
    "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85, "TotalCharges": 358.20
  }'
```

---

## Deploy on AWS (Free Tier)

We'll use **AWS App Runner** — the simplest zero-config hosting for containers with a free tier available.

> **AWS Free Tier**: App Runner gives 1 million requests/month free for 12 months. ECR gives 500 MB free storage.

### Prerequisites
- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) installed and configured (`aws configure`)
- [Docker](https://docs.docker.com/get-docker/) installed

---

### Step 1 — Push Docker image to Amazon ECR

```bash
# Set your variables
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO=churn-prediction-api

# Create ECR repository
aws ecr create-repository \
  --repository-name $ECR_REPO \
  --region $AWS_REGION

# Authenticate Docker to ECR
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS \
  --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Tag and push image
docker tag churn-api:latest \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest

docker push \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest
```

---

### Step 2 — Create an IAM Role for App Runner

```bash
# Create the trust policy file
cat > apprunner-trust.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "build.apprunner.amazonaws.com" },
    "Action": "sts:AssumeRole"
  }]
}
EOF

# Create role
aws iam create-role \
  --role-name AppRunnerECRAccessRole \
  --assume-role-policy-document file://apprunner-trust.json

# Attach managed policy for ECR access
aws iam attach-role-policy \
  --role-name AppRunnerECRAccessRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
```

---

### Step 3 — Deploy to App Runner

```bash
# Get the role ARN
ROLE_ARN=$(aws iam get-role \
  --role-name AppRunnerECRAccessRole \
  --query Role.Arn --output text)

IMAGE_URI=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest

# Create the App Runner service
aws apprunner create-service \
  --service-name churn-prediction-api \
  --source-configuration "{
    \"ImageRepository\": {
      \"ImageIdentifier\": \"$IMAGE_URI\",
      \"ImageConfiguration\": {
        \"Port\": \"8000\"
      },
      \"ImageRepositoryType\": \"ECR\"
    },
    \"AuthenticationConfiguration\": {
      \"AccessRoleArn\": \"$ROLE_ARN\"
    },
    \"AutoDeploymentsEnabled\": false
  }" \
  --instance-configuration '{
    "Cpu": "1 vCPU",
    "Memory": "2 GB"
  }' \
  --region $AWS_REGION
```

---

### Step 4 — Get your live URL

```bash
aws apprunner list-services \
  --region $AWS_REGION \
  --query "ServiceSummaryList[?ServiceName=='churn-prediction-api'].ServiceUrl" \
  --output text
```

Your API will be live at: `https://<random>.us-east-1.awsapprunner.com`

Swagger docs: `https://<your-url>/docs`

---

### Updating the deployment

```bash
# Rebuild, retag, push
docker build -t churn-api:latest .
docker tag churn-api:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest

# Trigger a new deployment
SERVICE_ARN=$(aws apprunner list-services \
  --region $AWS_REGION \
  --query "ServiceSummaryList[?ServiceName=='churn-prediction-api'].ServiceArn" \
  --output text)

aws apprunner start-deployment \
  --service-arn $SERVICE_ARN \
  --region $AWS_REGION
```

---

## GitHub Setup

### Step 1 — Initialize git locally

```bash
cd churn-prediction-api
git init
git add .
git commit -m "Initial commit: churn prediction FastAPI app"
```

### Step 2 — Create a `.gitignore`

```bash
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
.env
venv/
.venv/
*.egg-info/
dist/
.DS_Store
EOF

git add .gitignore
git commit -m "Add .gitignore"
```

### Step 3 — Create the GitHub repo (CLI)

Install the [GitHub CLI](https://cli.github.com/) if you haven't:

```bash
# macOS
brew install gh

# Ubuntu/Debian
sudo apt install gh
```

Then:

```bash
gh auth login          # follow prompts to log in
gh repo create churn-prediction-api \
  --public \
  --description "Telco customer churn prediction REST API using FastAPI + sklearn" \
  --source . \
  --push
```

### Step 4 — Or push manually via HTTPS

```bash
# Create a new repo at https://github.com/new first, then:
git remote add origin https://github.com/<your-username>/churn-prediction-api.git
git branch -M main
git push -u origin main
```

### Step 5 — Subsequent pushes

```bash
git add .
git commit -m "Your commit message"
git push
```

---

## License

MIT © 2024
