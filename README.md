# Crop Yield Prediction & Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange?style=flat-square&logo=scikit-learn)
![Flask](https://img.shields.io/badge/Flask-2.2-black?style=flat-square&logo=flask)
![License](https://img.shields.io/badge/License-Academic-green?style=flat-square)

**A machine learning system for Bangladesh agriculture**
that recommends suitable crops and predicts yield
using soil nutrients, weather, and environmental data.

*Course: CSE-4889 (Section-D) | United International University*

</div>

---

## Table of Contents

- [Overview](#overview)
- [Team](#team)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Pipeline](#pipeline)
- [Models & Results](#models--results)
- [Data Augmentation](#data-augmentation)
- [Features](#features)
- [How to Run](#how-to-run)
- [Web Application](#web-application)

---

## Overview

Agriculture is the backbone of Bangladesh's economy, yet most farmers make cropping decisions based on experience alone. This project bridges that gap using machine learning — giving farmers data-driven recommendations on what to grow and how much to expect.

The system solves two distinct problems:

| Task | Type | Goal |
|---|---|---|
| **Task A** | Regression | Predict crop production yield (metric tons) given soil & weather inputs |
| **Task B** | Classification | Recommend the most suitable crop for a given location and season |

Three models are trained and compared for each task: **Random Forest**, **Gradient Boosting**, and **Deep Neural Network (MLP)**.

---

## Team

**Team Name:** Light Seekers

| Name | Student ID |
|---|---|
| Md. Najmul Hossain Nur | 0112230536 |
| Md. Farhan Sadik Shafin | 0112230546 |
| Tanjil Hassan Sawan | 0112230556 |
| Asif Mustoba Sazzad | 0112230236 |
| Md. Arefin Iqram | 0112230926 |

**Supervised by:** Ms. Sadia Islam, Assistant Professor, Dept. of CSE
**Submission Date:** April 28, 2026

---

## Project Structure

```
MLLL/
│
├── Data/
│   ├── agumnetation/
│   │   ├── train_augmented.csv          # Augmented training set (21,139 rows)
│   │   └── test_original.csv            # Original held-out test set (373 rows)
│   │
│   ├── dataset/
│   │   ├── SPAS-Dataset-BD.csv          # Primary: Bangladesh district-level crop data
│   │   ├── Crop_recommendation.csv      # Soil nutrient features (N, P, K, pH)
│   │   └── 65 Years of Weather Data Bangladesh.csv  # Historical rainfall by station
│   │
│   └── Marge/
│       └── merged_dataset.csv           # Final merged dataset (4,607 rows, 20 features)
│
├── RF/
│   ├── Preprocesse-for-RF.py            # Cleaning → Encoding → Scaling → Split
│   ├── Rf-Model-training.py             # Random Forest + Gradient Boosting training
│   ├── model_comparison.png             # Side-by-side model metric charts
│   └── feature_importance_re...png      # Feature importance visualization
│
├── DNN/
│   ├── Augmentation.py                  # 3-technique data augmentation pipeline
│   ├── dnn_preprocess.py                # Scale augmented data for DNN input
│   ├── dnn_training.py                  # MLP training (regression + classification)
│   └── dnn_results.png                  # Loss curves + metric charts
│
├── preprocess/
│   ├── RF/                              # Scaled train/test splits for RF & GB
│   └── dnn/                             # Scaled train/val/test splits for DNN
│
├── Marge.py                             # Dataset merging script
├── app.py                               # Flask web application
└── README.md
```

---

## Datasets

Three public datasets were merged to create a comprehensive feature set:

| Dataset | Source | Rows | Key Features Added |
|---|---|---|---|
| SPAS-Dataset-BD | [Mendeley Data](https://data.mendeley.com/datasets/cphdw4z5kw/2) | 4,607 | Area, Production, Temp, Humidity, Season, District |
| Crop Recommendation | [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) | 2,200 | N, P, K, pH, Rainfall |
| 65 Years Bangladesh Weather | [Kaggle](https://www.kaggle.com/datasets/emonreza/65-years-of-weather-data-bangladesh-preprocessed) | — | District-wise historical Rainfall |

Datasets were merged using `Crop Name` and `District` as join keys, filling missing soil and rainfall features through crop-class averaging and station mapping respectively.

---

## Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  STEP 1 │ Marge.py                                      │
│          │ Merge 3 datasets → merged_dataset.csv         │
│          │ Output: 4,607 rows × 20 features              │
└──────────┴──────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 2 │ RF/Preprocesse-for-RF.py                      │
│          │ Drop leaky columns → Remove outliers (IQR)    │
│          │ Label encode → Log-transform → Scale → Split  │
│          │ Output: preprocess/RF/ (train 80%, test 20%)  │
└──────────┴──────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 3 │ RF/Rf-Model-training.py                       │
│          │ Train Random Forest + Gradient Boosting        │
│          │ Output: model_comparison.png                   │
│          │         feature_importance_regression.png      │
└──────────┴──────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 4 │ DNN/Augmentation.py                           │
│          │ Gaussian Noise + Interpolation + Seasonal Shift│
│          │ Output: train_augmented.csv (21,139 rows)      │
└──────────┴──────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 5 │ DNN/dnn_preprocess.py                         │
│          │ StandardScale augmented data → Train/Val split │
│          │ Output: preprocess/dnn/ (train 90%, val 10%)   │
└──────────┴──────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 6 │ DNN/dnn_training.py                           │
│          │ Train MLP (Regression + Classification)        │
│          │ Output: dnn_results.png                        │
└──────────┴──────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 7 │ app.py                                        │
│          │ Launch Flask web app → http://localhost:5000   │
└──────────┴──────────────────────────────────────────────┘
```

---

## Models & Results

### Task A — Yield Prediction (Regression)

| Model | Test R² | RMSE | MAE | Notes |
|---|---|---|---|---|
| Random Forest | 0.8952 | 0.6085 | 0.4247 | Strong baseline |
| **Gradient Boosting** | **0.9110** | **0.5607** | **0.4013** | ✅ Best regression model |
| DNN (MLP) | 0.8505 | 0.7267 | 0.5100 | Competitive with less data |

### Task B — Crop Recommendation (Classification)

| Model | Test Accuracy | F1 Score (Weighted) | Notes |
|---|---|---|---|
| Random Forest | 85.79% | 0.8572 | Strong baseline |
| Gradient Boosting | 85.25% | 0.8538 | Comparable to RF |
| **DNN (MLP)** | **91.15%** | **0.8922** | ✅ Best classification model |

> **Key Finding:** Gradient Boosting excels at yield prediction (R² = 0.91), while DNN leads in crop recommendation (Accuracy = 91.15%) — benefiting from the 5.7× augmented training set.

---

## Model Architectures

**Random Forest**
```
n_estimators = 150 | max_depth = 20
min_samples_leaf = 2 | class_weight = balanced
```

**Gradient Boosting (HistGradientBoosting)**
```
max_iter = 150 | learning_rate = 0.05 | max_depth = 6
```

**DNN — Regression (MLP)**
```
Input(15) → Dense(256, ReLU) → Dense(128, ReLU)
          → Dense(64, ReLU)  → Dense(32, ReLU)
          → Output(1, Linear)

Optimizer : Adam   | Learning Rate : 0.001
Loss      : MSE    | Batch Size    : 256
Early Stop: patience = 20
```

**DNN — Classification (MLP)**
```
Input(9) → Dense(256, ReLU) → Dense(128, ReLU)
         → Dense(64, ReLU)  → Output(72, Softmax)

Optimizer : Adam          | Learning Rate : 0.001
Loss      : Cross-Entropy | Batch Size    : 256
Early Stop: patience = 20 | Classes       : 72 crops
```

---

## Data Augmentation

The original dataset contained only **3,729 rows** — insufficient for training a deep neural network reliably. Three complementary augmentation techniques were applied exclusively to the training set:

| Technique | Method | Rows Added |
|---|---|---|
| Gaussian Noise | Add ±3% random noise to all numeric features | +11,187 |
| Class Interpolation | Linear interpolation between same-crop sample pairs (SMOTE-style) | +7,458 |
| Seasonal Variation | Apply Bangladesh-specific seasonal temperature and humidity shifts | +3,729 |
| **Final Training Set** | | **21,139 rows** |

> ⚠️ **Important:** The test set (`test_original.csv`) contains **only original, unmodified data** to ensure honest evaluation.

---

## Features

**Task A — Regression (15 input features)**

| Feature | Description |
|---|---|
| Area | Cultivated land area (hectares) |
| N, P, K | Soil nitrogen, phosphorus, potassium (mg/kg) |
| ph | Soil pH level |
| Avg / Min / Max Temp | Temperature readings (°C) |
| Avg / Min / Max Humidity | Relative humidity (%) |
| Rainfall | Annual rainfall (mm) |
| Season_enc | Encoded season (Kharif 1/2, Rabi) |
| District_enc | Encoded district (64 districts) |
| Crop_enc | Encoded crop type (72 crops) |

**Task B — Classification (9 input features)**
`N, P, K, ph, Avg Temp, Avg Humidity, Rainfall, Season_enc, District_enc`

**Target Variables**
- Task A → `Production_log` (log₁₊ₓ transformed yield)
- Task B → `Crop_enc` (72 crop classes across 64 Bangladesh districts)

---

## How to Run

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn flask
```

### Step-by-step

```bash
# Step 1 — Merge datasets
python Marge.py

# Step 2 — Preprocess for Random Forest & Gradient Boosting
python RF/Preprocesse-for-RF.py

# Step 3 — Train RF + Gradient Boosting models
python RF/Rf-Model-training.py

# Step 4 — Augment data for DNN
python DNN/Augmentation.py

# Step 5 — Preprocess augmented data for DNN
python DNN/dnn_preprocess.py

# Step 6 — Train DNN (MLP)
python DNN/dnn_training.py

# Step 7 — Launch web application
python app.py
# Open browser → http://localhost:5000
```

---

## Web Application

The Flask web app (`app.py`) provides an interactive interface with two sections:

**Farmer Dashboard**
Enter soil nutrients (N, P, K, pH), temperature, humidity, rainfall, and district to receive an instant crop recommendation powered by the best classification model.

**Model Comparison Dashboard**
Interactive visualization comparing all three models (RF, Gradient Boosting, DNN) across all evaluation metrics — R², RMSE, MAE, Accuracy, and F1 Score.

---

## Requirements

```
pandas >= 1.5.0
numpy >= 1.23.0
scikit-learn >= 1.1.0
matplotlib >= 3.6.0
seaborn >= 0.12.0
flask >= 2.2.0
```

---

*Developed for CSE-4889 Machine Learning Course | United International University, Bangladesh*