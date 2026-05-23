# Customer Churn Prediction AI System

End-to-end machine learning system for customer churn prediction with model deployment, monitoring, authentication, and interactive user interface.

---

## Project Overview

This project predicts whether a customer is likely to churn based on customer behavioural data and provides:

- Churn prediction
- Churn probability score
- Risk level classification
- API response latency
- Model version tracking
- Interactive prediction UI
- Real-time monitoring dashboard

## Live Demo

### Customer Churn Prediction UI

UI Demo:

http://churn-customer.duckdns.org:7860

Features:
- Customer churn prediction
- Probability score
- Risk level indicator
- API latency information
- Real-time prediction

---

### API Documentation

Swagger UI:

http://churn-customer.duckdns.org:8001/docs

Features:
- Interactive API testing
- Endpoint documentation
- Request/response examples

---

### Monitoring Dashboard

Grafana Dashboard:

http://churn-customer.duckdns.org:3000

Features:
- Total API calls
- Prediction requests
- Success/Error rate
- CPU usage
- RAM usage
- API latency
- Request rate monitoring

The project follows a production-style ML workflow including model serving, monitoring, containerization, and deployment.

---

## System Architecture

User Interface (Gradio)
        ↓
FastAPI Prediction API
        ↓
Authentication Layer
        ↓
Machine Learning Model
        ↓
Prometheus Metrics Collection
        ↓
Grafana Monitoring Dashboard
        ↓
AWS EC2 Deployment

---

## Features

### Machine Learning
- Customer churn prediction model
- Probability prediction
- Risk level indicator:
    - Low Risk 🟢
    - Medium Risk 🟡
    - High Risk 🔴

### API Features
- FastAPI REST API
- API key authentication
- Request rate limiting
- Model versioning
- Health check endpoint
- Swagger API documentation

### Monitoring
- Prometheus metrics collection
- Grafana monitoring dashboard
- API latency tracking
- Request rate monitoring
- Error rate monitoring
- CPU monitoring
- RAM monitoring
- Prediction request tracking

### Deployment
- Docker containerization
- AWS EC2 deployment
- CI/CD with GitHub Actions

### User Interface
- Interactive Gradio UI
- Real-time prediction results
- Customer input form

---

## Tech Stack

| Component | Technology |
|------------|------------|
| Language | Python |
| ML Model | Scikit-learn |
| API | FastAPI |
| Monitoring | Prometheus |
| Visualization | Grafana |
| UI | Gradio |
| Containerization | Docker |
| Deployment | AWS EC2 |
| CI/CD | GitHub Actions |

---

## API Endpoints

### Home

GET

```bash
/
