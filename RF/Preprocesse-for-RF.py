
"""
=============================================================
  preprocess/RFING PIPELINE — Crop Yield Prediction
  Team: Light Seekers | Course: CSE-4889
=============================================================
"""

print("DEBUG: Script started")
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Create preprocess/RF folder if it doesn't exist
os.makedirs('preprocess/RF', exist_ok=True)

# ─────────────────────────────────────────────
# STEP 0: Load Dataset
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 0: Loading Dataset")
print("=" * 60)

df = pd.read_csv('Data/Marge/merged_dataset.csv')
print(f"✅ Loaded shape: {df.shape}")
print(df.head(3).to_string())


# ─────────────────────────────────────────────
# STEP 1: Drop Unnecessary Columns
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: Dropping Unnecessary Columns")
print("=" * 60)

# Transplant, Growth, Harvest = text-based date ranges, not useful for ML
# AP Ratio = derived from Area & Production, causes data leakage
drop_cols = ['Transplant', 'Growth', 'Harvest', 'AP Ratio']
df.drop(columns=drop_cols, inplace=True)

print(f"✅ Dropped: {drop_cols}")
print(f"Remaining columns: {list(df.columns)}")


# ─────────────────────────────────────────────
# STEP 2: Handle Outliers in Production
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Handling Outliers (Production)")
print("=" * 60)

before = len(df)
# Remove Production = 0 (no yield recorded)
df = df[df['Production'] > 0]

# IQR method to remove extreme outliers
Q1 = df['Production'].quantile(0.25)
Q3 = df['Production'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 3 * IQR
upper = Q3 + 3 * IQR
df = df[(df['Production'] >= lower) & (df['Production'] <= upper)]

after = len(df)
print(f"✅ Rows before: {before} | After removing outliers: {after}")
print(f"   Removed {before - after} outlier rows")
print(f"   Production range now: {df['Production'].min():.0f} — {df['Production'].max():.0f}")


# ─────────────────────────────────────────────
# STEP 3: Encode Categorical Columns
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Encoding Categorical Columns")
print("=" * 60)

le_season   = LabelEncoder()
le_district = LabelEncoder()
le_crop     = LabelEncoder()

df['Season_enc']   = le_season.fit_transform(df['Season'])
df['District_enc'] = le_district.fit_transform(df['District'])
df['Crop_enc']     = le_crop.fit_transform(df['Crop Name'])

print("✅ Encoded columns:")
print(f"   Season     → {dict(zip(le_season.classes_, le_season.transform(le_season.classes_)))}")
print(f"   District   → 64 unique districts encoded (0–63)")
print(f"   Crop Name  → 72 unique crops encoded (0–71)")

# Save label encoder mappings for later use
season_map   = dict(zip(le_season.classes_,   le_season.transform(le_season.classes_)))
district_map = dict(zip(le_district.classes_, le_district.transform(le_district.classes_)))
crop_map     = dict(zip(le_crop.classes_,     le_crop.transform(le_crop.classes_)))

pd.DataFrame(list(season_map.items()),   columns=['Season','Code']).to_csv('preprocess/RF/encoding_season.csv',   index=False)
pd.DataFrame(list(district_map.items()), columns=['District','Code']).to_csv('preprocess/RF/encoding_district.csv', index=False)
pd.DataFrame(list(crop_map.items()),     columns=['Crop','Code']).to_csv('preprocess/RF/encoding_crop.csv',     index=False)
print("✅ Encoding maps saved as CSV files")


# ─────────────────────────────────────────────
# STEP 4: Log Transform Production (skewed)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Log Transform on Production (Target)")
print("=" * 60)

df['Production_log'] = np.log1p(df['Production'])
print(f"✅ Original Production  — skewness: {df['Production'].skew():.3f}")
print(f"   Log Production       — skewness: {df['Production_log'].skew():.3f}")
print("   (skewness কমলে model ভালো শেখে)")


# ─────────────────────────────────────────────
# STEP 5: Feature Selection
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Feature Selection")
print("=" * 60)

# ── Task A: Yield Prediction (Regression) ──
regression_features = [
    'Area', 'N', 'P', 'K', 'ph',
    'Avg Temp', 'Min Temp', 'Max Temp',
    'Avg Humidity', 'Min Relative Humidity', 'Max Relative Humidity',
    'Rainfall', 'Season_enc', 'District_enc', 'Crop_enc'
]
regression_target = 'Production_log'

# ── Task B: Crop Recommendation (Classification) ──
classification_features = [
    'N', 'P', 'K', 'ph',
    'Avg Temp', 'Avg Humidity', 'Rainfall',
    'Season_enc', 'District_enc'
]
classification_target = 'Crop_enc'

print(f"✅ Regression features   ({len(regression_features)}): {regression_features}")
print(f"✅ Classification target: Crop_enc (72 classes)")


# ─────────────────────────────────────────────
# STEP 6: Feature Scaling
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Feature Scaling")
print("=" * 60)

# For Regression
X_reg = df[regression_features].copy()
y_reg = df[regression_target].copy()

scaler_reg = StandardScaler()
X_reg_scaled = pd.DataFrame(
    scaler_reg.fit_transform(X_reg),
    columns=regression_features
)

# For Classification
X_cls = df[classification_features].copy()
y_cls = df[classification_target].copy()

scaler_cls = StandardScaler()
X_cls_scaled = pd.DataFrame(
    scaler_cls.fit_transform(X_cls),
    columns=classification_features
)

print("✅ StandardScaler applied to both regression & classification features")
print(f"   X_reg shape: {X_reg_scaled.shape}")
print(f"   X_cls shape: {X_cls_scaled.shape}")


# ─────────────────────────────────────────────
# STEP 7: Train / Test Split
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Train / Test Split (80% / 20%)")
print("=" * 60)

# Regression split
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg_scaled, y_reg, test_size=0.2, random_state=42
)

# Classification split
X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
    X_cls_scaled, y_cls, test_size=0.2, random_state=42
)

print(f"✅ Regression  — Train: {X_reg_train.shape} | Test: {X_reg_test.shape}")
print(f"✅ Classification — Train: {X_cls_train.shape} | Test: {X_cls_test.shape}")


# ─────────────────────────────────────────────
# STEP 8: Save preprocess/RFed Files
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Saving preprocess/RFed Files")
print("=" * 60)

# Save full cleaned dataframe
df.to_csv('preprocess/RF/dataset_cleaned.csv', index=False)

# Save regression splits
X_reg_train.to_csv('preprocess/RF/X_reg_train.csv', index=False)  # ← RF/RF ছিল, ঠিক করা হয়েছে
X_reg_test.to_csv('preprocess/RF/X_reg_test.csv',   index=False)
y_reg_train.to_csv('preprocess/RF/y_reg_train.csv', index=False)
y_reg_test.to_csv('preprocess/RF/y_reg_test.csv',   index=False)

# Save classification splits
X_cls_train.to_csv('preprocess/RF/X_cls_train.csv', index=False)
X_cls_test.to_csv('preprocess/RF/X_cls_test.csv',   index=False)
y_cls_train.to_csv('preprocess/RF/y_cls_train.csv', index=False)
y_cls_test.to_csv('preprocess/RF/y_cls_test.csv',   index=False)

print("✅ Saved files:")
print("   dataset_cleaned.csv")
print("   X_reg_train.csv, X_reg_test.csv, y_reg_train.csv, y_reg_test.csv")
print("   X_cls_train.csv, X_cls_test.csv, y_cls_train.csv, y_cls_test.csv")
print("   encoding_season.csv, encoding_district.csv, encoding_crop.csv")


# ─────────────────────────────────────────────
# STEP 9: Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ preprocess/RFING COMPLETE — SUMMARY")
print("=" * 60)
print(f"  Total rows after cleaning    : {len(df)}")
print(f"  Total features               : {len(regression_features)}")
print(f"  Task A (Regression)  target  : Production_log")
print(f"  Task B (Classification) target: Crop_enc (72 crops)")
print(f"  Train size                   : {len(X_reg_train)}")
print(f"  Test size                    : {len(X_reg_test)}")
print("=" * 60)