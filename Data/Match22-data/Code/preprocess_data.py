"""
=============================================================
  PREPROCESSING — Data/Match22-data/Data/merged_dataset.csv
  Team: Light Seekers | CSE-4889
  Match22-data Pipeline
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

# Enable UTF-8 encoding for Windows console output
sys.stdout.reconfigure(encoding='utf-8')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Define input and output paths
INPUT_FILE = 'Data/Match22-data/Data/merged_dataset.csv'
OUTPUT_DIR = 'Data/Match22-data/Processed_Data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1: Load Merged Dataset
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Data/Match22-data/Data/merged_dataset.csv")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)
print(f"  Raw shape      : {df.shape}")
print(f"  Columns        : {list(df.columns)}")
print(f"  Null values    : {df.isnull().sum().sum()} (total)")

# ─────────────────────────────────────────────
# STEP 2: Drop Leaky Columns
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Dropping Leaky Columns")
print("=" * 60)

drop_cols = ['Transplant', 'Growth', 'Harvest', 'AP Ratio']
existing_drop_cols = [c for c in drop_cols if c in df.columns]
df.drop(columns=existing_drop_cols, inplace=True)
print(f"  Dropped        : {existing_drop_cols}")
print(f"  Shape after    : {df.shape}")

# ─────────────────────────────────────────────
# STEP 3: Remove Zero/Negative Production Rows
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Cleaning Production <= 0 Rows")
print("=" * 60)

before = len(df)
df = df[df['Production'] > 0].copy()
df.reset_index(drop=True, inplace=True)
print(f"  Removed        : {before - len(df)} invalid rows")
print(f"  Shape after    : {df.shape}")

# ─────────────────────────────────────────────
# STEP 4: IQR Outlier Handling (Production, 3×IQR)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: IQR Outlier Removal (Production, 3×IQR)")
print("=" * 60)

before = len(df)
Q1  = df['Production'].quantile(0.25)
Q3  = df['Production'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 3 * IQR
upper = Q3 + 3 * IQR

df = df[(df['Production'] >= lower) & (df['Production'] <= upper)].copy()
df.reset_index(drop=True, inplace=True)

print(f"  Q1={Q1:.0f}, Q3={Q3:.0f}, IQR={IQR:.0f}")
print(f"  Lower bound    : {lower:.0f}")
print(f"  Upper bound    : {upper:.0f}")
print(f"  Removed        : {before - len(df)} outlier rows")
print(f"  Shape after    : {df.shape}")

# ─────────────────────────────────────────────
# STEP 5: Label Encoding (Season, District, Crop Name, Crop_Label_22)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Label Encoding Categorical Features")
print("=" * 60)

le_season   = LabelEncoder()
le_district = LabelEncoder()
le_crop     = LabelEncoder()
le_crop_22  = LabelEncoder()

df['Season_enc']   = le_season.fit_transform(df['Season'])
df['District_enc'] = le_district.fit_transform(df['District'])
df['Crop_enc']     = le_crop.fit_transform(df['Crop Name'])

if 'Crop_Label_22' in df.columns:
    df['Crop_22_enc'] = le_crop_22.fit_transform(df['Crop_Label_22'])
    print(f"  Crop 22  -> {len(le_crop_22.classes_)} classes : {list(le_crop_22.classes_)}")

print(f"  Season   -> {len(le_season.classes_)} classes : {list(le_season.classes_)}")
print(f"  District -> {len(le_district.classes_)} classes")
print(f"  Crop     -> {len(le_crop.classes_)} classes")

# Save Encodings to CSV
pd.DataFrame({
    'Season_label': le_season.classes_,
    'Season_enc': le_season.transform(le_season.classes_)
}).to_csv(os.path.join(OUTPUT_DIR, 'season_encoding.csv'), index=False)

pd.DataFrame({
    'Crop_label': le_crop.classes_,
    'Crop_enc': le_crop.transform(le_crop.classes_)
}).to_csv(os.path.join(OUTPUT_DIR, 'crop_encoding.csv'), index=False)

pd.DataFrame({
    'District_label': le_district.classes_,
    'District_enc': le_district.transform(le_district.classes_)
}).to_csv(os.path.join(OUTPUT_DIR, 'district_encoding.csv'), index=False)

if 'Crop_Label_22' in df.columns:
    pd.DataFrame({
        'Crop_22_label': le_crop_22.classes_,
        'Crop_22_enc': le_crop_22.transform(le_crop_22.classes_)
    }).to_csv(os.path.join(OUTPUT_DIR, 'crop_22_encoding.csv'), index=False)

# Save LabelEncoder objects using joblib
joblib.dump(le_season, os.path.join(OUTPUT_DIR, 'le_season.pkl'))
joblib.dump(le_district, os.path.join(OUTPUT_DIR, 'le_district.pkl'))
joblib.dump(le_crop, os.path.join(OUTPUT_DIR, 'le_crop.pkl'))
if 'Crop_Label_22' in df.columns:
    joblib.dump(le_crop_22, os.path.join(OUTPUT_DIR, 'le_crop_22.pkl'))

print(f"  Encoding CSVs and PKL objects saved to {OUTPUT_DIR}/")

# ─────────────────────────────────────────────
# STEP 6: Log Transform Production Target
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Log Transform Production → Production_log")
print("=" * 60)

df['Production_log'] = np.log1p(df['Production'])
print(f"  Production     min={df['Production'].min():.0f}  max={df['Production'].max():.0f}")
print(f"  Production_log min={df['Production_log'].min():.4f}  max={df['Production_log'].max():.4f}")

# ─────────────────────────────────────────────
# STEP 7: Define Feature Sets
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Defining Feature Sets")
print("=" * 60)

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

print(f"  Regression features ({len(REG_FEATURES)}) : {REG_FEATURES}")
print(f"  Classification features ({len(CLS_FEATURES)}) : {CLS_FEATURES}")

# ─────────────────────────────────────────────
# STEP 8: StandardScaler + Train/Val/Test Split
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Scaling & Train/Val/Test Split (72/8/20)")
print("=" * 60)

# Regression Split
X_reg = df[REG_FEATURES]
y_reg = df['Production_log']

X_temp, X_te_r, y_temp, y_te_r = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=42)
X_tr_r, X_val_r, y_tr_r, y_val_r = train_test_split(
    X_temp, y_temp, test_size=0.10, random_state=42)

scaler_reg = StandardScaler()
X_tr_r  = pd.DataFrame(scaler_reg.fit_transform(X_tr_r),  columns=REG_FEATURES, index=X_tr_r.index)
X_val_r = pd.DataFrame(scaler_reg.transform(X_val_r),     columns=REG_FEATURES, index=X_val_r.index)
X_te_r  = pd.DataFrame(scaler_reg.transform(X_te_r),      columns=REG_FEATURES, index=X_te_r.index)
joblib.dump(scaler_reg, os.path.join(OUTPUT_DIR, 'scaler_reg.pkl'))

# Classification Split (predicting Crop_22_enc if available, otherwise Crop_enc)
target_cls_col = 'Crop_22_enc' if 'Crop_22_enc' in df.columns else 'Crop_enc'
X_cls = df[CLS_FEATURES]
y_cls = df[target_cls_col]

X_tempc, X_te_c, y_tempc, y_te_c = train_test_split(
    X_cls, y_cls, test_size=0.20, random_state=42, stratify=y_cls)
X_tr_c, X_val_c, y_tr_c, y_val_c = train_test_split(
    X_tempc, y_tempc, test_size=0.10, random_state=42, stratify=y_tempc)

scaler_cls = StandardScaler()
X_tr_c  = pd.DataFrame(scaler_cls.fit_transform(X_tr_c),  columns=CLS_FEATURES, index=X_tr_c.index)
X_val_c = pd.DataFrame(scaler_cls.transform(X_val_c),     columns=CLS_FEATURES, index=X_val_c.index)
X_te_c  = pd.DataFrame(scaler_cls.transform(X_te_c),      columns=CLS_FEATURES, index=X_te_c.index)
joblib.dump(scaler_cls, os.path.join(OUTPUT_DIR, 'scaler_cls.pkl'))

print("  Regression:")
print(f"    Train : {X_tr_r.shape[0]:>5} rows  ({X_tr_r.shape[0]/len(df)*100:.1f}%)")
print(f"    Val   : {X_val_r.shape[0]:>5} rows  ({X_val_r.shape[0]/len(df)*100:.1f}%)")
print(f"    Test  : {X_te_r.shape[0]:>5} rows  ({X_te_r.shape[0]/len(df)*100:.1f}%)")

print("  Classification Target:", target_cls_col)
print(f"    Train : {X_tr_c.shape[0]:>5} rows  ({X_tr_c.shape[0]/len(df)*100:.1f}%)")
print(f"    Val   : {X_val_c.shape[0]:>5} rows  ({X_val_c.shape[0]/len(df)*100:.1f}%)")
print(f"    Test  : {X_te_c.shape[0]:>5} rows  ({X_te_c.shape[0]/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────
# STEP 9: Save Preprocessed Datasets
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"STEP 9: Saving Preprocessed CSVS to {OUTPUT_DIR}/")
print("=" * 60)

# Save Cleaned Full Dataset
df.to_csv(os.path.join(OUTPUT_DIR, 'cleaned_dataset.csv'), index=False)

# Regression CSVS
X_tr_r.to_csv(os.path.join(OUTPUT_DIR, 'X_reg_train.csv'), index=False)
X_val_r.to_csv(os.path.join(OUTPUT_DIR, 'X_reg_val.csv'), index=False)
X_te_r.to_csv(os.path.join(OUTPUT_DIR, 'X_reg_test.csv'), index=False)
y_tr_r.to_csv(os.path.join(OUTPUT_DIR, 'y_reg_train.csv'), index=False)
y_val_r.to_csv(os.path.join(OUTPUT_DIR, 'y_reg_val.csv'), index=False)
y_te_r.to_csv(os.path.join(OUTPUT_DIR, 'y_reg_test.csv'), index=False)

# Classification CSVS
X_tr_c.to_csv(os.path.join(OUTPUT_DIR, 'X_cls_train.csv'), index=False)
X_val_c.to_csv(os.path.join(OUTPUT_DIR, 'X_cls_val.csv'), index=False)
X_te_c.to_csv(os.path.join(OUTPUT_DIR, 'X_cls_test.csv'), index=False)
y_tr_c.to_csv(os.path.join(OUTPUT_DIR, 'y_cls_train.csv'), index=False)
y_val_c.to_csv(os.path.join(OUTPUT_DIR, 'y_cls_val.csv'), index=False)
y_te_c.to_csv(os.path.join(OUTPUT_DIR, 'y_cls_test.csv'), index=False)

print("  Saved regression & classification train/val/test split files successfully!")

# ─────────────────────────────────────────────
# STEP 10: Generate Summary Visualizations
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10: Generating Summary Visualization Chart")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle(
    'Preprocessing & Cleaning Summary — Match22 Data\n'
    'Team: Light Seekers | CSE-4889',
    fontsize=14, fontweight='bold'
)

# 1: Production Distribution (raw vs log)
ax = axes[0, 0]
ax.hist(df['Production'], bins=40, color='#e74c3c', alpha=0.8, edgecolor='white')
ax.set_title('Production Distribution (Raw)', fontweight='bold')
ax.set_xlabel('Production'); ax.set_ylabel('Count')
ax.grid(axis='y', alpha=0.3)

ax = axes[0, 1]
ax.hist(df['Production_log'], bins=40, color='#2ecc71', alpha=0.8, edgecolor='white')
ax.set_title('Production_log Distribution (After log1p)', fontweight='bold')
ax.set_xlabel('log(1 + Production)'); ax.set_ylabel('Count')
ax.grid(axis='y', alpha=0.3)

# 2: Top Crops by Count
ax = axes[0, 2]
crop_counts = df['Crop Name'].value_counts()
crop_counts.plot(kind='barh', ax=ax, color='#3498db', edgecolor='white')
ax.set_title('Crop Distribution in Match22 Dataset', fontweight='bold')
ax.set_xlabel('Count')
ax.grid(axis='x', alpha=0.3)

# 3: Season Distribution
ax = axes[1, 0]
season_counts = df['Season'].value_counts()
ax.pie(season_counts.values, labels=season_counts.index, autopct='%1.1f%%',
       colors=['#2ecc71', '#3498db', '#e67e22'], startangle=90)
ax.set_title('Season Distribution', fontweight='bold')

# 4: Train/Val/Test Split
ax = axes[1, 1]
split_labels = ['Train', 'Val', 'Test']
split_sizes  = [X_tr_r.shape[0], X_val_r.shape[0], X_te_r.shape[0]]
bars = ax.bar(split_labels, split_sizes, color=['#2ecc71', '#f39c12', '#e74c3c'], width=0.5, edgecolor='white')
for b, v in zip(bars, split_sizes):
    ax.text(b.get_x() + b.get_width()/2, v + 5, f'{v}\n({v/len(df)*100:.1f}%)', ha='center', fontweight='bold', fontsize=11)
ax.set_title('Train / Val / Test Split', fontweight='bold')
ax.set_ylabel('Row Count')
ax.grid(axis='y', alpha=0.3)

# 5: Feature Correlations with Production_log
ax = axes[1, 2]
corr_cols = ['Area', 'N', 'P', 'K', 'ph', 'Avg Temp', 'Avg Humidity', 'Rainfall', 'Production_log']
corr = df[corr_cols].corr()[['Production_log']].drop('Production_log')
colors_list = ['#e74c3c' if v < 0 else '#2ecc71' for v in corr['Production_log']]
bars = ax.barh(corr.index, corr['Production_log'], color=colors_list, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Correlation with Production_log', fontweight='bold')
ax.set_xlabel('Pearson Correlation')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, 'preprocess_summary.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"  Saved chart to {chart_path}")

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE ✅")
print("=" * 60)
