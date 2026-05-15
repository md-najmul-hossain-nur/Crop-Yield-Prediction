"""
=============================================================
  RF AUGMENTATION + PREPROCESSING + TRAINING
  Team: Light Seekers | CSE-4889

  নতুন Pipeline (তোমার plan অনুযায়ী):
  
  Step 1: merged_dataset.csv load
  Step 2: Augmentation (Gaussian + Interpolation + Seasonal)
  Step 3: Preprocessing (Clean + Encode + Scale + Split)
  Step 4: RF + Gradient Boosting Train
  Step 5: Result Comparison (আগের vs নতুন)

  Run: python RF_Augmentation.py
=============================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier
)
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score
)
warnings.filterwarnings('ignore')
np.random.seed(42)

os.makedirs('preprocess/RF_aug', exist_ok=True)

# ═══════════════════════════════════════════════
# STEP 1: Load Original Merged Dataset
# ═══════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Original Dataset Load")
print("=" * 60)

df = pd.read_csv('Data/Marge/merged_dataset.csv')

# Leaky columns বাদ দাও
drop_cols = ['Transplant', 'Growth', 'Harvest', 'AP Ratio']
df.drop(columns=drop_cols, inplace=True)

# Production = 0 বাদ দাও
df = df[df['Production'] > 0].copy()
df.reset_index(drop=True, inplace=True)

print(f"✅ Loaded shape : {df.shape}")
print(f"   Columns      : {list(df.columns)}")


# ═══════════════════════════════════════════════
# STEP 2: AUGMENTATION (আগে clean এর আগে)
# ═══════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Data Augmentation")
print("=" * 60)

NUMERIC_COLS = [
    'Area', 'N', 'P', 'K', 'ph',
    'Avg Temp', 'Min Temp', 'Max Temp',
    'Avg Humidity', 'Min Relative Humidity', 'Max Relative Humidity',
    'Rainfall', 'Production'
]

# ── Technique 1: Gaussian Noise ──
print("\n[1] Gaussian Noise Augmentation...")

def gaussian_noise(df, n_copies=2, noise_pct=0.03):
    parts = []
    for _ in range(n_copies):
        df_copy = df.copy()
        for col in NUMERIC_COLS:
            std = df[col].std()
            noise = np.random.normal(0, std * noise_pct, size=len(df))
            df_copy[col] = (df[col] + noise).clip(lower=0)
        parts.append(df_copy)
    return pd.concat(parts, ignore_index=True)

noise_data = gaussian_noise(df, n_copies=2, noise_pct=0.03)
print(f"   ✅ Gaussian rows: {len(noise_data):,}")

# ── Technique 2: Class Interpolation ──
print("[2] Class Interpolation (SMOTE-style)...")

def class_interpolation(df, n_per_row=1):
    rows = []
    for crop in df['Crop Name'].unique():
        crop_df = df[df['Crop Name'] == crop].reset_index(drop=True)
        if len(crop_df) < 2:
            continue
        for _ in range(n_per_row * len(crop_df)):
            i, j = np.random.choice(len(crop_df), 2, replace=False)
            alpha = np.random.uniform(0.3, 0.7)
            new_row = crop_df.iloc[i].copy()
            for col in NUMERIC_COLS:
                new_row[col] = alpha * crop_df.iloc[i][col] + (1 - alpha) * crop_df.iloc[j][col]
            new_row[col] = max(0, new_row[col])
            rows.append(new_row)
    return pd.DataFrame(rows).reset_index(drop=True)

interp_data = class_interpolation(df, n_per_row=1)
print(f"   ✅ Interpolation rows: {len(interp_data):,}")

# ── Technique 3: Seasonal Variation ──
print("[3] Seasonal Variation Augmentation...")

SEASON_SHIFTS = {
    'Kharif 1': {'Avg Temp': +1.5, 'Avg Humidity': +3.0, 'Rainfall': +15.0},
    'Kharif 2': {'Avg Temp': +1.0, 'Avg Humidity': +2.0, 'Rainfall': +10.0},
    'Rabi':     {'Avg Temp': -1.5, 'Avg Humidity': -3.0, 'Rainfall': -8.0},
}

def seasonal_variation(df):
    rows = []
    for _, row in df.iterrows():
        shifts = SEASON_SHIFTS.get(row['Season'], {})
        new_row = row.copy()
        for col, shift in shifts.items():
            variation = np.random.uniform(-0.3, 0.3) * abs(shift)
            new_row[col] = max(0, row[col] + shift + variation)
        if 'Avg Temp' in shifts:
            delta = new_row['Avg Temp'] - row['Avg Temp']
            new_row['Min Temp'] = max(0, row['Min Temp'] + delta * 0.8)
            new_row['Max Temp'] = row['Max Temp'] + delta * 1.2
        if 'Avg Humidity' in shifts:
            delta = new_row['Avg Humidity'] - row['Avg Humidity']
            new_row['Min Relative Humidity'] = max(0, row['Min Relative Humidity'] + delta)
            new_row['Max Relative Humidity'] = min(100, row['Max Relative Humidity'] + delta)
        prod_noise = np.random.uniform(-0.05, 0.05) * row['Production']
        new_row['Production'] = max(1, row['Production'] + prod_noise)
        rows.append(new_row)
    return pd.DataFrame(rows).reset_index(drop=True)

seasonal_data = seasonal_variation(df)
print(f"   ✅ Seasonal rows: {len(seasonal_data):,}")

# ── Combine All ──
df_aug = pd.concat([df, noise_data, interp_data, seasonal_data], ignore_index=True)
df_aug = df_aug.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n   Original      : {len(df):,} rows")
print(f"   Gaussian      : {len(noise_data):,} rows")
print(f"   Interpolation : {len(interp_data):,} rows")
print(f"   Seasonal      : {len(seasonal_data):,} rows")
print(f"   ─────────────────────────")
print(f"   Total (Aug)   : {len(df_aug):,} rows  ✅")


# ═══════════════════════════════════════════════
# STEP 3: PREPROCESSING (augmented data এ)
# ═══════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Preprocessing on Augmented Data")
print("=" * 60)

before = len(df_aug)

# IQR Outlier Removal
Q1 = df_aug['Production'].quantile(0.25)
Q3 = df_aug['Production'].quantile(0.75)
IQR = Q3 - Q1
df_aug = df_aug[
    (df_aug['Production'] >= Q1 - 3 * IQR) &
    (df_aug['Production'] <= Q3 + 3 * IQR)
].copy()

after = len(df_aug)
print(f"✅ Outlier removal: {before:,} → {after:,} rows (removed {before-after:,})")

# Label Encoding
le_season   = LabelEncoder()
le_district = LabelEncoder()
le_crop     = LabelEncoder()

df_aug['Season_enc']   = le_season.fit_transform(df_aug['Season'])
df_aug['District_enc'] = le_district.fit_transform(df_aug['District'])
df_aug['Crop_enc']     = le_crop.fit_transform(df_aug['Crop Name'])

print(f"✅ Encoding done — Seasons: {len(le_season.classes_)}, Districts: {len(le_district.classes_)}, Crops: {len(le_crop.classes_)}")

# Log Transform
df_aug['Production_log'] = np.log1p(df_aug['Production'])
print(f"✅ Log transform — skewness: {df_aug['Production'].skew():.3f} → {df_aug['Production_log'].skew():.3f}")

# Save encoding maps
pd.DataFrame(list(zip(le_season.classes_, le_season.transform(le_season.classes_))),
             columns=['Season','Code']).to_csv('preprocess/RF_aug/encoding_season.csv', index=False)
pd.DataFrame(list(zip(le_district.classes_, le_district.transform(le_district.classes_))),
             columns=['District','Code']).to_csv('preprocess/RF_aug/encoding_district.csv', index=False)
pd.DataFrame(list(zip(le_crop.classes_, le_crop.transform(le_crop.classes_))),
             columns=['Crop','Code']).to_csv('preprocess/RF_aug/encoding_crop.csv', index=False)
print("✅ Encoding maps saved → preprocess/RF_aug/")

# Feature Selection
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

# Scaling
X_reg = df_aug[REG_FEATURES].copy()
y_reg = df_aug['Production_log'].copy()
X_cls = df_aug[CLS_FEATURES].copy()
y_cls = df_aug['Crop_enc'].copy()

scaler_r = StandardScaler()
scaler_c = StandardScaler()
X_reg_scaled = pd.DataFrame(scaler_r.fit_transform(X_reg), columns=REG_FEATURES)
X_cls_scaled = pd.DataFrame(scaler_c.fit_transform(X_cls), columns=CLS_FEATURES)

# Train/Test Split (80/20)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg_scaled, y_reg, test_size=0.2, random_state=42
)
X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
    X_cls_scaled, y_cls, test_size=0.2, random_state=42
)

print(f"\n✅ Regression  — Train: {X_reg_train.shape} | Test: {X_reg_test.shape}")
print(f"✅ Classification — Train: {X_cls_train.shape} | Test: {X_cls_test.shape}")

# Save splits
X_reg_train.to_csv('preprocess/RF_aug/X_reg_train.csv', index=False)
X_reg_test.to_csv('preprocess/RF_aug/X_reg_test.csv',   index=False)
y_reg_train.to_csv('preprocess/RF_aug/y_reg_train.csv', index=False)
y_reg_test.to_csv('preprocess/RF_aug/y_reg_test.csv',   index=False)
X_cls_train.to_csv('preprocess/RF_aug/X_cls_train.csv', index=False)
X_cls_test.to_csv('preprocess/RF_aug/X_cls_test.csv',   index=False)
y_cls_train.to_csv('preprocess/RF_aug/y_cls_train.csv', index=False)
y_cls_test.to_csv('preprocess/RF_aug/y_cls_test.csv',   index=False)
print("✅ Splits saved → preprocess/RF_aug/")


# ═══════════════════════════════════════════════
# STEP 4: MODEL TRAINING
# ═══════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Model Training")
print("=" * 60)

def reg_report(name, y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    print(f"  {name:<14} R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")
    return r2, rmse, mae

def cls_report(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"  {name:<14} Accuracy={acc:.4f}  F1={f1:.4f}")
    return acc, f1

# Validation split (train থেকে 10%)
X_tr_r, X_val_r, y_tr_r, y_val_r = train_test_split(X_reg_train, y_reg_train, test_size=0.1, random_state=42)
X_tr_c, X_val_c, y_tr_c, y_val_c = train_test_split(X_cls_train, y_cls_train, test_size=0.1, random_state=42)

# ── Random Forest ──
print("\n[1] Random Forest Regressor (Augmented Data)")
rf_reg = RandomForestRegressor(n_estimators=150, max_depth=20, min_samples_leaf=2, n_jobs=-1, random_state=42)
rf_reg.fit(X_tr_r, y_tr_r)
reg_report("Validation", y_val_r, rf_reg.predict(X_val_r))
rf_r2, rf_rmse, rf_mae = reg_report("Test", y_reg_test, rf_reg.predict(X_reg_test))

print("\n[2] Random Forest Classifier (Augmented Data)")
rf_cls = RandomForestClassifier(n_estimators=150, max_depth=20, min_samples_leaf=2, class_weight='balanced', n_jobs=-1, random_state=42)
rf_cls.fit(X_tr_c, y_tr_c)
cls_report("Validation", y_val_c, rf_cls.predict(X_val_c))
rf_acc, rf_f1 = cls_report("Test", y_cls_test, rf_cls.predict(X_cls_test))

# ── Gradient Boosting ──
print("\n[3] Gradient Boosting Regressor (Augmented Data)")
gb_reg = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=6, random_state=42)
gb_reg.fit(X_tr_r, y_tr_r)
reg_report("Validation", y_val_r, gb_reg.predict(X_val_r))
gb_r2, gb_rmse, gb_mae = reg_report("Test", y_reg_test, gb_reg.predict(X_reg_test))

print("\n[4] Gradient Boosting Classifier (Augmented Data)")
gb_cls = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=6, random_state=42)
gb_cls.fit(X_tr_c, y_tr_c)
cls_report("Validation", y_val_c, gb_cls.predict(X_val_c))
gb_acc, gb_f1 = cls_report("Test", y_cls_test, gb_cls.predict(X_cls_test))


# ═══════════════════════════════════════════════
# STEP 5: BEFORE vs AFTER COMPARISON
# ═══════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Before vs After Comparison")
print("=" * 60)

# আগের result (original preprocessing থেকে)
OLD = {
    'RF_R2': 0.8952, 'RF_RMSE': 0.6085, 'RF_MAE': 0.4247,
    'RF_Acc': 0.8579, 'RF_F1': 0.8572,
    'GB_R2': 0.9110, 'GB_RMSE': 0.5607, 'GB_MAE': 0.4013,
    'GB_Acc': 0.8525, 'GB_F1': 0.8538,
}

print(f"\n  Task A — Regression")
print(f"  {'Model':<22} {'Old R²':>8} {'New R²':>8} {'Change':>8}")
print("  " + "-"*50)
print(f"  {'Random Forest':<22} {OLD['RF_R2']:>8.4f} {rf_r2:>8.4f} {rf_r2-OLD['RF_R2']:>+8.4f} {'✅' if rf_r2 > OLD['RF_R2'] else '❌'}")
print(f"  {'Gradient Boosting':<22} {OLD['GB_R2']:>8.4f} {gb_r2:>8.4f} {gb_r2-OLD['GB_R2']:>+8.4f} {'✅' if gb_r2 > OLD['GB_R2'] else '❌'}")

print(f"\n  Task B — Classification")
print(f"  {'Model':<22} {'Old Acc':>9} {'New Acc':>9} {'Change':>8}")
print("  " + "-"*50)
print(f"  {'Random Forest':<22} {OLD['RF_Acc']:>9.4f} {rf_acc:>9.4f} {rf_acc-OLD['RF_Acc']:>+8.4f} {'✅' if rf_acc > OLD['RF_Acc'] else '❌'}")
print(f"  {'Gradient Boosting':<22} {OLD['GB_Acc']:>9.4f} {gb_acc:>9.4f} {gb_acc-OLD['GB_Acc']:>+8.4f} {'✅' if gb_acc > OLD['GB_Acc'] else '❌'}")

# ── Chart: Before vs After ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Before vs After Augmentation — RF & GB\nTeam: Light Seekers | CSE-4889',
             fontsize=13, fontweight='bold')

models  = ['RF (Before)', 'RF (After)', 'GB (Before)', 'GB (After)']
colors  = ['#95a5a6', '#2ecc71', '#95a5a6', '#3498db']

# R² chart
ax = axes[0]
r2_vals = [OLD['RF_R2'], rf_r2, OLD['GB_R2'], gb_r2]
bars = ax.bar(models, r2_vals, color=colors, edgecolor='white', width=0.5)
ax.set_title('Task A — R² Score (Regression)', fontweight='bold')
ax.set_ylim(0.8, 1.0)
ax.set_ylabel('R²')
for b, v in zip(bars, r2_vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.002, f'{v:.4f}', ha='center', fontweight='bold', fontsize=10)
ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.4, label='0.90 line')
ax.legend()

# Accuracy chart
ax = axes[1]
acc_vals = [OLD['RF_Acc'], rf_acc, OLD['GB_Acc'], gb_acc]
bars = ax.bar(models, acc_vals, color=colors, edgecolor='white', width=0.5)
ax.set_title('Task B — Accuracy (Classification)', fontweight='bold')
ax.set_ylim(0.8, 1.0)
ax.set_ylabel('Accuracy')
for b, v in zip(bars, acc_vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.002, f'{v:.4f}', ha='center', fontweight='bold', fontsize=10)
ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.4, label='0.90 line')
ax.legend()

plt.tight_layout()
plt.savefig('RF_aug_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Chart saved: RF_aug_comparison.png")

print("\n" + "=" * 60)
print("✅ সব শেষ! RF_aug_comparison.png দেখো!")
print("=" * 60)
print(f"  Data: {len(df):,} original → {len(df_aug):,} augmented")
print(f"  RF  R² : {OLD['RF_R2']:.4f} → {rf_r2:.4f}")
print(f"  GB  R² : {OLD['GB_R2']:.4f} → {gb_r2:.4f}")
print(f"  RF  Acc: {OLD['RF_Acc']:.4f} → {rf_acc:.4f}")
print(f"  GB  Acc: {OLD['GB_Acc']:.4f} → {gb_acc:.4f}")
print("=" * 60)
