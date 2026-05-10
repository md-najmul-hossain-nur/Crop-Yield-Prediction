"""
=============================================================
  PREPROCESSING FOR DNN — Crop Yield Prediction
  Team: Light Seekers | Course: CSE-4889

  Input  : Data/agumnetation/train_augmented.csv
           Data/agumnetation/val_augmented.csv
           Data/agumnetation/test_original.csv
  Output : preprocess/dnn/ folder এ scaled files
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('preprocess/dnn', exist_ok=True)

# ─────────────────────────────────────────────
# STEP 0: Load — তিনটা file আলাদা আলাদা
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 0: Data Load")
print("=" * 60)

train = pd.read_csv('Data/agumnetation/train_augmented.csv')
val   = pd.read_csv('Data/agumnetation/val_augmented.csv')
test  = pd.read_csv('Data/agumnetation/test_original.csv')

print(f"Train (augmented) : {train.shape[0]:,} rows")
print(f"Val   (augmented) : {val.shape[0]:,} rows")
print(f"Test  (original)  : {test.shape[0]:,} rows")

# ─────────────────────────────────────────────
# Features
# ─────────────────────────────────────────────
REG_FEATURES = [
    'Area', 'N', 'P', 'K', 'ph',
    'Avg Temp', 'Min Temp', 'Max Temp',
    'Avg Humidity', 'Min Relative Humidity', 'Max Relative Humidity',
    'Rainfall', 'Season_enc', 'District_enc', 'Crop_enc'
]

CLS_FEATURES = [
    'N', 'P', 'K', 'ph',
    'Avg Temp', 'Avg Humidity', 'Rainfall',
    'Season_enc', 'District_enc'
]

# ─────────────────────────────────────────────
# STEP 1: Scale — Regression
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: StandardScaler — Regression Features")
print("=" * 60)

scaler_r = StandardScaler()

X_train_r = pd.DataFrame(scaler_r.fit_transform(train[REG_FEATURES]), columns=REG_FEATURES)
X_val_r   = pd.DataFrame(scaler_r.transform(val[REG_FEATURES]),       columns=REG_FEATURES)
X_test_r  = pd.DataFrame(scaler_r.transform(test[REG_FEATURES]),      columns=REG_FEATURES)

y_train_r = train['Production_log'].reset_index(drop=True)
y_val_r   = val['Production_log'].reset_index(drop=True)
y_test_r  = test['Production_log'].reset_index(drop=True)

print(f"X_train_r : {X_train_r.shape}")
print(f"X_val_r   : {X_val_r.shape}")
print(f"X_test_r  : {X_test_r.shape}")

# ─────────────────────────────────────────────
# STEP 2: Scale — Classification
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: StandardScaler — Classification Features")
print("=" * 60)

scaler_c = StandardScaler()

X_train_c = pd.DataFrame(scaler_c.fit_transform(train[CLS_FEATURES]), columns=CLS_FEATURES)
X_val_c   = pd.DataFrame(scaler_c.transform(val[CLS_FEATURES]),       columns=CLS_FEATURES)
X_test_c  = pd.DataFrame(scaler_c.transform(test[CLS_FEATURES]),      columns=CLS_FEATURES)

y_train_c = train['Crop_enc'].reset_index(drop=True)
y_val_c   = val['Crop_enc'].reset_index(drop=True)
y_test_c  = test['Crop_enc'].reset_index(drop=True)

print(f"X_train_c : {X_train_c.shape}")
print(f"X_val_c   : {X_val_c.shape}")
print(f"X_test_c  : {X_test_c.shape}")

# ─────────────────────────────────────────────
# STEP 3: Save → preprocess/dnn/
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Files Save")
print("=" * 60)

# Regression
X_train_r.to_csv('preprocess/dnn/X_reg_train.csv', index=False)
X_val_r.to_csv('preprocess/dnn/X_reg_val.csv',     index=False)
X_test_r.to_csv('preprocess/dnn/X_reg_test.csv',   index=False)
y_train_r.to_csv('preprocess/dnn/y_reg_train.csv', index=False)
y_val_r.to_csv('preprocess/dnn/y_reg_val.csv',     index=False)
y_test_r.to_csv('preprocess/dnn/y_reg_test.csv',   index=False)

# Classification
X_train_c.to_csv('preprocess/dnn/X_cls_train.csv', index=False)
X_val_c.to_csv('preprocess/dnn/X_cls_val.csv',     index=False)
X_test_c.to_csv('preprocess/dnn/X_cls_test.csv',   index=False)
y_train_c.to_csv('preprocess/dnn/y_cls_train.csv', index=False)
y_val_c.to_csv('preprocess/dnn/y_cls_val.csv',     index=False)
y_test_c.to_csv('preprocess/dnn/y_cls_test.csv',   index=False)

print("✅ Saved — preprocess/dnn/:")
print("   Regression     : X_reg_train, X_reg_val, X_reg_test")
print("                    y_reg_train, y_reg_val, y_reg_test")
print("   Classification : X_cls_train, X_cls_val, X_cls_test")
print("                    y_cls_train, y_cls_val, y_cls_test")

print("\n" + "=" * 60)
print("✅ PREPROCESSING COMPLETE")
print("=" * 60)
print(f"  Train rows   : {len(X_train_r):,}  (augmented + scaled)")
print(f"  Val rows     : {len(X_val_r):,}  (augmented + scaled)")
print(f"  Test rows    : {len(X_test_r):,}  (original only)")
print(f"  Reg features : {len(REG_FEATURES)}")
print(f"  Cls features : {len(CLS_FEATURES)}")
print("=" * 60)
print("\nএখন dnn_training.py চালাও!")