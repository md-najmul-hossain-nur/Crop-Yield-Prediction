# 🌾 Crop Yield Prediction & Recommendation System
### Team: Light Seekers | Course: CSE-4889 (Section-D)

> A machine learning system for Bangladesh agriculture that recommends suitable crops and predicts yield using soil, weather, and environmental data.

---

## 👥 Team Members

| Name | Student ID |
|---|---|
| Md. Najmul Hossain Nur | 0112230536 |
| Md. Farhan Sadik Shafin | 0112230546 |
| Tanjil Hassan Sawan | 0112230556 |
| Asif Mustoba Sazzad | 0112230236 |
| Md. Arefin Iqram | 0112230926 |

**Submitted to:** Ms. Sadia Islam (Assistant Professor, Dept. of CSE)
**Submitted:** April 28, 2026

---

## 📌 Project Overview

This project addresses two key agricultural problems in Bangladesh:

- **Task A — Yield Prediction (Regression):** Predict how much a crop will produce given soil and weather conditions
- **Task B — Crop Recommendation (Classification):** Recommend the best crop to grow given local conditions

---

## 📁 Project Structure

```
MLLL/
│
├── Crop-Yield-Prediction/           # Git repository root
│
├── Data/
│   ├── agumnetation/
│   │   ├── train_augmented.csv      # Augmented training data (~21,000 rows)
│   │   └── test_original.csv        # Original test data (373 rows)
│   │
│   ├── dataset/
│   │   ├── SPAS-Dataset-BD.csv      # Bangladesh crop data (primary)
│   │   ├── Crop_recommendation.csv  # NPK + soil features
│   │   └── 65 Years of Weather Data Bangladesh.csv
│   │
│   └── Marge/
│       └── merged_dataset.csv       # Final merged dataset
│
├── DNN/
│   ├── Augmentation.py              # Data augmentation pipeline
│   ├── dnn_preprocess.py            # Preprocessing for DNN
│   ├── dnn_training.py              # DNN (MLP) model training
│   └── dnn_results.png              # DNN result charts
│
├── preprocess/
│   ├── dnn/                         # Scaled data for DNN training
│   └── RF/                          # Scaled data for RF training
│
├── RF/
│   ├── Preprocesse-for-RF.py        # Preprocessing for RF + GB models
│   ├── Rf-Model-training.py         # Random Forest + Gradient Boosting
│   ├── model_comparison.png         # Model comparison charts
│   └── feature_importance_re...png  # Feature importance chart
│
├── Marge.py                         # Step 1: Merge 3 datasets
└── app.py                           # Web Application (Flask)
```

---

## 🗄️ Datasets Used

| Dataset | Source | Purpose |
|---|---|---|
| SPAS-Dataset-BD | Mendeley Data | Primary Bangladesh crop data (4,607 rows, 64 districts) |
| Crop Recommendation | Kaggle | N, P, K, pH, Rainfall features |
| 65 Years Weather Bangladesh | Kaggle | District-wise historical rainfall |

---

## ⚙️ Pipeline

```
Step 1: Marge.py
        ↓  Merge 3 datasets by Crop Name & District
        Data/Marge/merged_dataset.csv  (4,607 rows, 20 features)

Step 2: RF/Preprocesse-for-RF.py
        ↓  Clean → Encode → Scale → Split (80/20)
        preprocess/RF/ folder

Step 3: RF/Rf-Model-training.py
        ↓  Train Random Forest + Gradient Boosting
        RF/model_comparison.png
        RF/feature_importance_regression.png

Step 4: DNN/Augmentation.py
        ↓  Gaussian Noise + Interpolation + Seasonal Shift
        Data/agumnetation/train_augmented.csv  (21,139 rows)

Step 5: DNN/dnn_preprocess.py
        ↓  Scale augmented data for DNN
        preprocess/dnn/ folder

Step 6: DNN/dnn_training.py
        ↓  Train DNN (MLP — 4 hidden layers)
        DNN/dnn_results.png

Step 7: app.py
        ↓  Run web application
        http://localhost:5000
```

---

## 🤖 Models

### Task A — Yield Prediction (Regression)

| Model | Test R² | RMSE | MAE |
|---|---|---|---|
| Random Forest | 0.8952 | 0.6085 | 0.4247 |
| **Gradient Boosting** | **0.9110** | **0.5607** | **0.4013** |
| DNN (MLP) | 0.8505 | 0.7267 | 0.5100 |

### Task B — Crop Recommendation (Classification)

| Model | Test Accuracy | F1 Score |
|---|---|---|
| Random Forest | 0.8579 | 0.8572 |
| Gradient Boosting | 0.8525 | 0.8538 |
| **DNN (MLP)** | **0.9115** | **0.8922** |

---

## 🧠 Model Architectures

**Random Forest**
- 150 estimators, max depth 20, class_weight=balanced

**Gradient Boosting (HistGradientBoosting)**
- 150 iterations, learning rate 0.05, max depth 6

**DNN — Regression**
```
Input(15) → Dense(256, ReLU) → Dense(128, ReLU)
          → Dense(64, ReLU)  → Dense(32, ReLU)
          → Output(1, Linear)
Optimizer: Adam | LR: 0.001 | Batch: 256 | Early Stopping: patience=20
```

**DNN — Classification**
```
Input(9) → Dense(256, ReLU) → Dense(128, ReLU)
         → Dense(64, ReLU)  → Output(72, Softmax)
Optimizer: Adam | LR: 0.001 | Batch: 256 | Early Stopping: patience=20
```

---

## 📊 Data Augmentation

Original dataset had only **3,729 rows** — too small for Deep Learning.

| Technique | Description | New Rows Added |
|---|---|---|
| Gaussian Noise | 3% random noise on numeric features | +11,187 |
| Class Interpolation | SMOTE-style same-crop mixing | +7,458 |
| Seasonal Variation | Realistic temp/humidity shift by season | +3,729 |
| **Total** | | **21,139 rows (5.7x increase)** |

> ⚠️ Test set contains **only original data** — no augmented rows.

---

## 🔧 Features

**Regression (15 features):**
`Area, N, P, K, ph, Avg Temp, Min Temp, Max Temp, Avg Humidity, Min Humidity, Max Humidity, Rainfall, Season_enc, District_enc, Crop_enc`

**Classification (9 features):**
`N, P, K, ph, Avg Temp, Avg Humidity, Rainfall, Season_enc, District_enc`

**Target Variables:**
- Regression → `Production_log` (log-transformed yield)
- Classification → `Crop_enc` (72 crop classes, 64 districts)

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn flask

# 2. Merge datasets
python Marge.py

# 3. Preprocess for RF + GB
python RF/Preprocesse-for-RF.py

# 4. Train RF + Gradient Boosting
python RF/Rf-Model-training.py

# 5. Augment data for DNN
python DNN/Augmentation.py

# 6. Preprocess for DNN
python DNN/dnn_preprocess.py

# 7. Train DNN
python DNN/dnn_training.py

# 8. Run web app
python app.py
```

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
matplotlib>=3.6.0
seaborn>=0.12.0
flask>=2.2.0
```

---

## 🌐 Web Application

The web app (`app.py`) provides two main features:

- **🌱 Farmer Dashboard** — Input soil & weather conditions → get crop recommendation instantly
- **📊 Model Comparison** — Visual comparison of RF, Gradient Boosting, and DNN performance

---

## 📄 License

This project was developed for academic purposes — Course CSE-4889, United International University, Bangladesh.
