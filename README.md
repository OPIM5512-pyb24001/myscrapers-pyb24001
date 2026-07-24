# 🚗 End-to-End Production Machine Learning Pipeline for Craigslist Vehicle Price Prediction

An end-to-end production-grade machine learning pipeline that continuously scrapes Craigslist vehicle listings, transforms raw text into structured datasets using automated ETL workflows, trains and evaluates predictive models, and deploys automated predictions through Google Cloud Platform and GitHub Actions.

This project demonstrates production-ready Data Engineering, Machine Learning, MLOps, Cloud Computing, and Explainable AI practices.

---

# Overview

The objective of this project is to build an automated machine learning system capable of:

- Scraping vehicle listings from Craigslist
- Extracting structured information from raw listing text
- Building a cumulative historical dataset
- Training predictive machine learning models
- Generating price predictions for newly scraped listings
- Explaining model behavior using Explainable AI techniques
- Automatically synchronizing model artifacts back to GitHub

The entire pipeline executes automatically using Google Cloud Functions, Cloud Scheduler, Cloud Storage, and GitHub Actions.

---

# Architecture

```
Craigslist Listings
        │
        ▼
Web Scraper
(Python + BeautifulSoup)
        │
        ▼
Google Cloud Storage
Raw TXT Listings
        │
        ▼
Regex Extraction
Cloud Function
        │
        ▼
Structured JSON Records
        │
        ▼
Materialization Pipeline
        │
        ▼
Master Training Dataset (CSV)
        │
        ▼
Feature Engineering
        │
        ▼
Random Forest Model
        │
        ▼
Predictions
Feature Importance
Partial Dependence Plots
        │
        ▼
Google Cloud Storage
        │
        ▼
GitHub Actions
        │
        ▼
Repository Results
```

---

# Features

- Automated Craigslist vehicle scraping
- Cloud-native ETL pipeline
- Incremental dataset creation
- Automated feature engineering
- Historical model training
- Hourly prediction generation
- Explainable AI
- Cloud deployment
- CI/CD using GitHub Actions
- Fully reproducible pipeline

---

# Tech Stack

## Programming

- Python

## Machine Learning

- Scikit-learn
- Random Forest Regression

## Cloud

- Google Cloud Platform
- Cloud Functions Gen2
- Cloud Storage
- Cloud Scheduler

## Automation

- GitHub Actions

## Data Processing

- Pandas
- NumPy

## Visualization

- Matplotlib

## Web Scraping

- BeautifulSoup
- Requests

---

# Project Structure

```
cloud_function/
│
├── extractor/
│
├── materialize/
│
├── train-dt/
│
├── extractor-llm/
│
├── materialize-llm/
│
└── train-dt-llm/
│
.github/
└── workflows/
    ├── deploy-extractor.yml
    ├── deploy-materialize.yml
    ├── deploy-train.yml
    └── sync-results.yml

results/

README.md
```

---

# Workflow

## Step 1 — Data Collection

Vehicle listings are automatically scraped from Craigslist using Python and BeautifulSoup.

The scraper:

- Collects listing URLs
- Downloads complete listing pages
- Stores raw text in Google Cloud Storage

---

## Step 2 — ETL

Cloud Functions automatically process raw listing text.

Information extracted includes:

- Price
- Make
- Model
- Year
- Mileage
- Fuel Type
- Transmission
- Drive Type
- Title Status

Structured JSON records are written back to Cloud Storage.

---

## Step 3 — Dataset Materialization

All structured JSON files are merged into a continuously growing master dataset.

Duplicate listings are removed while preserving the newest version.

---

## Step 4 — Feature Engineering

Engineered features include:

- Vehicle Age
- Log Mileage
- Miles Per Year

Categorical variables are encoded using One-Hot Encoding.

Missing values are automatically imputed.

---

## Step 5 — Model Training

A Random Forest Regressor is trained using historical listings.

Model configuration:

- 200 Trees
- Maximum Depth: 20
- Parallel Training
- Random State: 42

The model predicts prices for newly scraped listings while training only on historical data.

---

## Step 6 — Model Evaluation

Performance is evaluated using:

- Mean Absolute Error (MAE)

Additional model interpretation includes:

- Permutation Feature Importance
- Partial Dependence Plots (PDP)

---

## Step 7 — Automation

Google Cloud Scheduler automatically executes:

1. Scraping
2. ETL
3. Dataset Materialization
4. Model Training
5. Prediction Generation

GitHub Actions automatically:

- Downloads artifacts
- Commits predictions
- Publishes feature importance
- Uploads explainability visualizations

---

# Results

Example pipeline outputs include:

- Predictions CSV
- Feature Importance
- Partial Dependence Plots

The production pipeline generated:

- **3,411+ vehicle price predictions**
- **23 automated production runs**
- **Hourly model retraining**
- **Best observed MAE of approximately \$3.37K**

---

# Explainable AI

To improve transparency, the project includes:

- Permutation Feature Importance
- Partial Dependence Plots

These techniques help interpret:

- Most influential features
- Feature interactions
- Model behavior
- Prediction sensitivity

---

# MLOps Components

- Cloud Functions Gen2
- Cloud Scheduler
- GitHub Actions
- Continuous Model Retraining
- Automated Predictions
- Automated Artifact Publishing

---

# Future Improvements

- Hyperparameter optimization using Optuna
- XGBoost and LightGBM benchmarking
- Time-series cross-validation
- Model Registry
- Vertex AI deployment
- Real-time prediction API
- Drift detection and monitoring
- LLM-assisted feature extraction

---

# Learning Outcomes

This project demonstrates practical experience in:

- Machine Learning
- Feature Engineering
- Data Engineering
- ETL Pipelines
- Cloud Computing
- MLOps
- CI/CD
- Explainable AI
- GitHub Automation
- Production Model Deployment

---

# Author

**Vikram Krishnareddy**

Master of Science in Data Science

University of Connecticut

GitHub: https://github.com/Vikramgk47

LinkedIn: https://linkedin.com/in/vikram-krishnareddy
