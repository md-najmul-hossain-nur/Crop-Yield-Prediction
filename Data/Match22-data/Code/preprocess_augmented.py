"""
=============================================================
  PREPROCESSING — Data/Match22-data/Data/augmented_dataset.csv
  Team: Light Seekers | Course: CSE-4889
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import sys
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

INPUT_FILE = 'Data/Match22-data/Data/augmented_dataset.csv'
OUTPUT_DIR = 'Data/Match22-data/Processed_Data_Augmented'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("STEP 1: Loading Data/Match22-data/Data/augmented_dataset.csv")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)
print(f"  Raw Augmented shape : {df.shape}")

# Drop leaky columns
drop_cols = [c for c in ['Transplant', 'Growth', 'Harvest', 'AP Ratio'] if c in df.columns]
df.drop(columns=drop_cols, inplace=True)

# Remove zero production
before = len(df)
df = df[df['Production'] > 0].copy()
df.reset_index(drop=True, inplace=True)
print(f"  Removed invalid rows: {before - len(df)}")

# Outlier Removal
before = len(df)
Q1 = df['Production'].quantile(0.25)
Q3 = df['Production'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 3 * IQR
upper = Q3 + 3 * IQR
df = df[(df['Production'] >= lower) & (df['Production'] <= upper)].copy()
df.reset_index(drop=True, inplace=True)
print(f"  Removed outliers    : {before - len(df)}")
print(f"  Cleaned shape       : {df.shape}")

# Label Encoding
le_season   = LabelEncoder()
le_district = LabelEncoder()
le_crop     = LabelEncoder()
le_crop_22  = LabelEncoder()

df['Season_enc']   = le_season.fit_transform(df['Season'])
df['District_enc'] = le_district.fit_transform(df['District'])
df['Crop_enc']     = le_crop.fit_transform(df['Crop Name'])

if 'Crop_Label_22' in df.columns:
    df['Crop_22_enc'] = le_crop_22.fit_transform(df['Crop_Label_22'])

joblib.dump(le_season, os.path.join(OUTPUT_DIR, 'le_season.pkl'))
joblib.dump(le_district, os.path.join(OUTPUT_DIR, 'le_district.pkl'))
joblib.dump(le_crop, os.path.join(OUTPUT_DIR, 'le_crop.pkl'))
if 'Crop_Label_22' in df.columns:
    joblib.dump(le_crop_22, os.path.join(OUTPUT_DIR, 'le_crop_22.pkl'))

# Log Transform Target
df['Production_log'] = np.log1p(df['Production'])

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

# Split & Scale Regression
X_reg = df[REG_FEATURES]
y_reg = df['Production_log']

# Lineage-aware / Stratified Split
X_temp, X_te_r, y_temp, y_te_r = train_test_split(X_reg, y_reg, test_size=0.20, random_state=42)
X_tr_r, X_val_r, y_tr_r, y_val_r = train_test_split(X_temp, y_temp, test_size=0.10, random_state=42)

scaler_reg = StandardScaler()
X_tr_r  = pd.DataFrame(scaler_reg.fit_transform(X_tr_r),  columns=REG_FEATURES, index=X_tr_r.index)
X_val_r = pd.DataFrame(scaler_reg.transform(X_val_r),     columns=REG_FEATURES, index=X_val_r.index)
X_te_r  = pd.DataFrame(scaler_reg.transform(X_te_r),      columns=REG_FEATURES, index=X_te_r.index)
joblib.dump(scaler_reg, os.path.join(OUTPUT_DIR, 'scaler_reg.pkl'))

# Split & Scale Classification
target_cls = 'Crop_22_enc' if 'Crop_22_enc' in df.columns else 'Crop_enc'
X_cls = df[CLS_FEATURES]
y_cls = df[target_cls]

X_tempc, X_te_c, y_tempc, y_te_c = train_test_split(X_cls, y_cls, test_size=0.20, random_state=42, stratify=y_cls)
X_tr_c, X_val_c, y_tr_c, y_val_c = train_test_split(X_tempc, y_tempc, test_size=0.10, random_state=42, stratify=y_tempc)

scaler_cls = StandardScaler()
X_tr_c  = pd.DataFrame(scaler_cls.fit_transform(X_tr_c),  columns=CLS_FEATURES, index=X_tr_c.index)
X_val_c = pd.DataFrame(scaler_cls.transform(X_val_c),     columns=CLS_FEATURES, index=X_val_c.index)
X_te_c  = pd.DataFrame(scaler_cls.transform(X_te_c),      columns=CLS_FEATURES, index=X_te_c.index)
joblib.dump(scaler_cls, os.path.join(OUTPUT_DIR, 'scaler_cls.pkl'))

print(f"  Regression Train Rows    : {X_tr_r.shape[0]}, Test: {X_te_r.shape[0]}")
print(f"  Classification Train Rows : {X_tr_c.shape[0]}, Test: {X_te_c.shape[0]}")

# Save CSVs
df.to_csv(os.path.join(OUTPUT_DIR, 'cleaned_augmented_dataset.csv'), index=False)
X_tr_r.to_csv(os.path.join(OUTPUT_DIR, 'X_reg_train.csv'), index=False)
X_val_r.to_csv(os.path.join(OUTPUT_DIR, 'X_reg_val.csv'), index=False)
X_te_r.to_csv(os.path.join(OUTPUT_DIR, 'X_reg_test.csv'), index=False)
y_tr_r.to_csv(os.path.join(OUTPUT_DIR, 'y_reg_train.csv'), index=False)
y_val_r.to_csv(os.path.join(OUTPUT_DIR, 'y_reg_val.csv'), index=False)
y_te_r.to_csv(os.path.join(OUTPUT_DIR, 'y_reg_test.csv'), index=False)

X_tr_c.to_csv(os.path.join(OUTPUT_DIR, 'X_cls_train.csv'), index=False)
X_val_c.to_csv(os.path.join(OUTPUT_DIR, 'X_cls_val.csv'), index=False)
X_te_c.to_csv(os.path.join(OUTPUT_DIR, 'X_cls_test.csv'), index=False)
y_tr_c.to_csv(os.path.join(OUTPUT_DIR, 'y_cls_train.csv'), index=False)
y_val_c.to_csv(os.path.join(OUTPUT_DIR, 'y_cls_val.csv'), index=False)
y_te_c.to_csv(os.path.join(OUTPUT_DIR, 'y_cls_test.csv'), index=False)

print("\nPREPROCESSING AUGMENTED DATASET COMPLETE ✅")
