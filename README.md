# 💧 Water Quality Monitoring System

A real-time water quality monitoring system using IoT sensor simulation and Machine Learning classifiers, deployed on Streamlit Cloud.

## 🔬 Project Overview

Periodic manual water testing fails to capture sudden contamination events. This project integrates simulated IoT sensor data with ML classifiers to create a continuous, real-time monitoring system capable of immediately detecting and alerting authorities to hazardous changes in water quality.

## 🚀 Live Demo

👉 **[Open Live App](https://your-app-link.streamlit.app](https://water-quality-monitor-gasf6pdpgdjywecrzmbx5t.streamlit.app/))** ← replace after deploying

## 📊 Features

- **Manual Test Tab** — Enter water parameters via sliders → instant Safe/Contaminated prediction
- **Sensor Simulation Tab** — Simulates 30 IoT readings with live progress, line chart, and alert log
- **Model Sidebar** — Shows classifier used and WHO safety thresholds

## 🤖 ML Models Trained

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | ~60% | ~0.45 |
| Random Forest | ~67% | ~0.55 |
| SVM | ~65% | ~0.50 |

**Best Model: Random Forest Classifier**

## 📁 Dataset

[Kaggle — Water Potability Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
- 3,276 samples · 9 features · Binary classification (Potable / Not Potable)

## ⚙️ Tech Stack

- Python, scikit-learn, pandas, numpy
- Streamlit (dashboard)
- joblib (model serialization)
- Deployed on: Streamlit Cloud

## 🗂️ Project Structure

```
├── app.py                  # Streamlit dashboard
├── best_model.pkl          # Trained Random Forest model
├── scaler.pkl              # StandardScaler
├── requirements.txt        # Python dependencies
└── notebooks/
    ├── eda.ipynb           # Exploratory Data Analysis
    └── models.ipynb        # Model training & comparison
```

## 🏃 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/water-quality-monitor
cd water-quality-monitor
pip install -r requirements.txt
streamlit run app.py
```

---
**Mini Project | T.E. AI & DS | Cloud Computing**
