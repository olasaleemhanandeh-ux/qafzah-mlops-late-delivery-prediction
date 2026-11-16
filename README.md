# Qafza — MLOps Late Delivery Prediction Pipeline

An end-to-end **MLOps pipeline** for predicting late deliveries in e-commerce orders using the **Olist Brazilian E-Commerce Dataset**.

This project transforms a machine learning workflow into a structured, production-oriented pipeline covering data validation, feature engineering, model management, inference, API development, automated testing, and deployment.

**Tech Stack:** Python · Scikit-learn · Pandas · Great Expectations · MLflow · FastAPI · Pydantic · Docker · Docker Compose · DVC · Pytest · GitHub Actions

---

## Project Overview

The objective is to predict whether an e-commerce order will be delivered late or on time.

The target variable is:

| Value | Meaning |
|---|---|
| `0` | On-time delivery |
| `1` | Late delivery |

The project focuses on transforming a notebook-based machine learning workflow into a reusable and deployable MLOps pipeline.

## MLOps Pipeline

### 1. Data Management & Versioning
- Organized dataset artifacts under `data/artifacts/`.
- Used DVC to track data artifacts and support reproducibility.
- Centralized project configuration in `config/config.yaml`.

### 2. Data Validation
Implemented data validation using **Great Expectations**, including:
- Schema and column validation.
- Data type checks.
- Missing-value validation.
- Categorical value validation.
- Numerical range checks.
- Target value validation.

### 3. Feature Engineering
- Extracted reusable feature engineering logic from the notebooks.
- Created time-based features from order timestamps.
- Integrated geolocation features.
- Prepared numerical and categorical features.
- Maintained consistent feature preparation for inference.

### 4. Model Management
- Used a Scikit-learn Random Forest Classifier.
- Applied class weighting to address class imbalance.
- Used a saved model and preprocessing pipeline for inference.
- Integrated MLflow for experiment tracking and model registration.

### 5. Prediction Service
- Implemented a reusable prediction module.
- Loaded the saved model and preprocessor.
- Applied a configurable classification threshold of `0.51`.
- Returned prediction results and probabilities.

### 6. API Development
Developed a REST API using **FastAPI** and **Pydantic**.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | API health check |
| `POST` | `/predict` | Predicts delivery status |

Interactive API documentation is available through Swagger UI at:

`http://localhost:8000/docs`

### 7. Automated Testing
Implemented automated tests using **Pytest**, covering:
- Feature engineering.
- Prediction functionality.
- API endpoints.

**Latest recorded test result:** 13 passed.

### 8. Deployment & CI/CD
- Dockerfile for containerization.
- Docker Compose for service orchestration.
- GitHub Actions workflow for automated project checks.

---

## Model Evaluation

The recorded F1 scores from the existing model evaluation:

| Evaluation Set | F1 Score |
|---|---:|
| Validation | 0.1909 |
| Test | 0.0582 |

---

## Getting Started

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Qafzah_Task2
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**Linux / macOS**
```bash
source .venv/bin/activate
```

**Windows**
```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

For development:

```bash
pip install -r requirements/dev.txt
```

For runtime:

```bash
pip install -r requirements/runtime.txt
```

### 4. Run Data Validation

```bash
python -m src.validation.great_expectations
```

### 5. Run Automated Tests

```bash
pytest -v
```

### 6. Start the API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Access the API documentation:

`http://localhost:8000/docs`

### 7. Run with Docker Compose

```bash
docker compose up -d --build
```

To stop the services:

```bash
docker compose down
```

---

## Project Structure

```text
Qafzah_Task2/
│
├── .dvc/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── artifacts/
│   ├── .gitignore
│   └── artifacts.dvc
│
├── notebooks/
│   ├── 01_read_join.ipynb
│   ├── 02_create_labels.ipynb
│   ├── 03_split.ipynb
│   ├── 04_eda.ipynb
│   ├── 05_feature_engineering.ipynb
│   └── 06_modeling.ipynb
│
├── requirements/
│   ├── dev.txt
│   └── runtime.txt
│
├── src/
│   ├── validation/
│   │   ├── __init__.py
│   │   └── great_expectations.py
│   │
│   ├── __init__.py
│   ├── api.py
│   ├── feature_engineering.py
│   ├── predict.py
│   └── register_model.py
│
├── tests/
│   ├── test_api.py
│   ├── test_feature_engineering.py
│   └── test_predictor.py
│
├── .dockerignore
├── .dvcignore
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
└── README.md
```

---

## Key Learning Outcomes

- Transforming notebook-based ML workflows into reusable Python modules.
- Building an end-to-end inference pipeline.
- Implementing data validation with Great Expectations.
- Managing data artifacts using DVC.
- Integrating MLflow for experiment tracking and model management.
- Developing REST APIs using FastAPI and Pydantic.
- Writing automated tests with Pytest.
- Containerizing applications with Docker.
- Implementing CI/CD workflows using GitHub Actions.

---

## Acknowledgment

Developed as part of **Qafza Training**, applying MLOps concepts and practices to build an end-to-end machine learning pipeline.
