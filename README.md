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
├── Data/
│   ├── agumnetation/
│   │   ├── train_augmented.csv       # Augmented training data (~21,000 rows)
│   │   └── test_original.csv         # Original test data (373 rows)
│   │
│   ├── dataset/
│   │   ├── SPAS-Dataset-BD.csv       # Bangladesh crop data (primary)
│   │   ├── Crop_recommendation.csv   # NPK + soil features
│   │   └── 65 Years of Weather Data Bangladesh.csv
│   │
│   └── Marge/
│       └── merged_dataset.csv        # Final merged dataset
│
├── preprocess/
│   ├── dataset_cleaned.csv
│   ├── encoding_crop.csv
│   ├── encoding_district.csv
│   ├── encoding_season.csv
│   ├── X_reg_train.csv / X_reg_test.csv
│   ├── y_reg_train.csv / y_reg_test.csv
│   ├── X_cls_train.csv / X_cls_test.csv
│   ├── y_cls_train.csv / y_cls_test.csv
│   └── dnn/                          # Augmented + scaled data for DNN
│       ├── X_reg_train.csv / X_reg_val.csv / X_reg_test.csv
│       ├── y_reg_train.csv / y_reg_val.csv / y_reg_test.csv
│       ├── X_cls_train.csv / X_cls_val.csv / X_cls_test.csv
│       └── y_cls_train.csv / y_cls_val.csv / y_cls_test.csv
│
├── Marge.py                          # Step 1: Merge 3 datasets
├── Preprocessed.py                   # Step 2: Clean + Encode + Scale + Split
├── Augmentation.py                   # Step 3: Data Augmentation
├── dnn_preprocess.py                 # Step 4: Preprocessing for DNN
├── model_training.py                 # Step 5: RF + Gradient Boosting
├── dnn_training.py                   # Step 6: DNN (MLP)
└── app.py                            # Web Application (Flask)
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
        merged_dataset.csv  (4,607 rows, 20 features)

Step 2: Preprocessed.py
        ↓  Clean → Encode → Scale → Split (80/20)
        preprocess/ folder

Step 3: Augmentation.py
        ↓  Gaussian Noise + Interpolation + Seasonal Shift
        train_augmented.csv  (21,139 rows — 5.7x increase)

Step 4: dnn_preprocess.py
        ↓  Scale augmented data for DNN
        preprocess/dnn/ folder

Step 5: model_training.py
        ↓  Train RF + Gradient Boosting
        model_comparison.png, feature_importance_regression.png

Step 6: dnn_training.py
        ↓  Train DNN (MLP — 4 hidden layers)
        dnn_results.png
```

---

## 🤖 Models

### Task A — Yield Prediction (Regression)

| Model | Val R² | Test R² | RMSE | MAE |
|---|---|---|---|---|
| Random Forest | - | 0.8952 | 0.6085 | 0.4247 |
| **Gradient Boosting** | - | **0.9110** | **0.5607** | **0.4013** |
| DNN (MLP) | - | 0.8505 | 0.7267 | 0.5100 |

### Task B — Crop Recommendation (Classification)

| Model | Val Acc | Test Acc | F1 Score |
|---|---|---|---|
| Random Forest | - | 0.8579 | 0.8572 |
| Gradient Boosting | - | 0.8525 | 0.8538 |
| **DNN (MLP)** | - | **0.9115** | **0.8922** |

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

| Technique | Description | New Rows |
|---|---|---|
| Gaussian Noise | 3% noise on numeric features | +11,187 |
| Class Interpolation | SMOTE-style same-class mixing | +7,458 |
| Seasonal Variation | Realistic temp/humidity shift | +3,729 |
| **Total** | | **21,139 rows** |

> ⚠️ Test set contains **only original data** — no augmented rows.

---

## 🔧 Features

**Regression (15 features):**
`Area, N, P, K, ph, Avg Temp, Min Temp, Max Temp, Avg Humidity, Min Humidity, Max Humidity, Rainfall, Season_enc, District_enc, Crop_enc`

**Classification (9 features):**
`N, P, K, ph, Avg Temp, Avg Humidity, Rainfall, Season_enc, District_enc`

**Target:**
- Regression → `Production_log` (log-transformed yield)
- Classification → `Crop_enc` (72 crop classes)

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn flask

# 2. Merge datasets
python Marge.py

# 3. Preprocess
python Preprocessed.py

# 4. Augment data
python Augmentation.py

# 5. Preprocess for DNN
python dnn_preprocess.py

# 6. Train RF + Gradient Boosting
python model_training.py

# 7. Train DNN
python dnn_training.py

# 8. Run web app
python app.py
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
flask
```

---

## 🌐 Web Application

The web app (`app.py`) provides:
- **Farmer Dashboard** — Input soil & weather conditions → get crop recommendation
- **Model Comparison** — Visual comparison of all 3 models (RF, GB, DNN)

---

## 📄 License

This project was developed for academic purposes — Course CSE-4889, United International University.
