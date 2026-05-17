"""
=============================================================
  PREPROCESSING — augmented_dataset.csv
  Team: Light Seekers | Course: CSE-4889

  Steps:
    1. Load Augmentation/Data/augmented_dataset.csv
    2. Drop leaky columns
    3. Remove zero production rows
    4. IQR outlier removal
    5. Label Encoding (Season, District, Crop)
    6. Log transform Production
    7. StandardScaler
    8. Train/Val/Test split (72/8/20)
    9. Save to Augmentation/Data/ folder
   10. Summary report + shape chart

  Run from: Crop-Yield-Prediction/
  Output  : Augmentation/Data/ folder
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

os.makedirs('Augmentation/Data', exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1: Load
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading augmented_dataset.csv")
print("=" * 60)

df = pd.read_csv('Augmentation/Data/augmented_dataset.csv')
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
df.drop(columns=drop_cols, inplace=True)
print(f"  Dropped        : {drop_cols}")
print(f"  Shape after    : {df.shape}")

# ─────────────────────────────────────────────
# STEP 3: Remove Zero Production
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Removing Zero Production Rows")
print("=" * 60)

before = len(df)
df = df[df['Production'] > 0].copy()
df.reset_index(drop=True, inplace=True)
print(f"  Removed        : {before - len(df)} rows (Production = 0)")
print(f"  Shape after    : {df.shape}")

# ─────────────────────────────────────────────
# STEP 4: IQR Outlier Removal
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
# STEP 5: Label Encoding
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Label Encoding")
print("=" * 60)

le_season   = LabelEncoder()
le_district = LabelEncoder()
le_crop     = LabelEncoder()

df['Season_enc']   = le_season.fit_transform(df['Season'])
df['District_enc'] = le_district.fit_transform(df['District'])
df['Crop_enc']     = le_crop.fit_transform(df['Crop Name'])

print(f"  Season   → {len(le_season.classes_)} classes : {list(le_season.classes_)}")
print(f"  District → {len(le_district.classes_)} classes")
print(f"  Crop     → {len(le_crop.classes_)} classes")

# Save encoding maps
pd.DataFrame({
    'Season_label': le_season.classes_,
    'Season_enc'  : le_season.transform(le_season.classes_)
}).to_csv('Augmentation/Data/season_encoding.csv', index=False)

pd.DataFrame({
    'Crop_label': le_crop.classes_,
    'Crop_enc'  : le_crop.transform(le_crop.classes_)
}).to_csv('Augmentation/Data/crop_encoding.csv', index=False)

pd.DataFrame({
    'District_label': le_district.classes_,
    'District_enc'  : le_district.transform(le_district.classes_)
}).to_csv('Augmentation/Data/district_encoding.csv', index=False)

print(f"  ✅ Encoding maps saved to Augmentation/Data/")

# ─────────────────────────────────────────────
# STEP 6: Log Transform Target
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Log Transform Production → Production_log")
print("=" * 60)

df['Production_log'] = np.log1p(df['Production'])
print(f"  Production     min={df['Production'].min():.0f}  max={df['Production'].max():.0f}")
print(f"  Production_log min={df['Production_log'].min():.4f}  max={df['Production_log'].max():.4f}")

# ─────────────────────────────────────────────
# STEP 7: Feature Sets
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

print(f"  Regression features ({len(REG_FEATURES)})     : {REG_FEATURES}")
print(f"  Classification features ({len(CLS_FEATURES)}) : {CLS_FEATURES}")

# ─────────────────────────────────────────────
# STEP 8: Scale + Split (72/8/20)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: StandardScaler + Train/Val/Test Split (72/8/20)")
print("=" * 60)

# ── Regression ──
X_reg = df[REG_FEATURES]
y_reg = df['Production_log']

scaler_reg = StandardScaler()
X_reg_scaled = pd.DataFrame(scaler_reg.fit_transform(X_reg), columns=REG_FEATURES)

X_temp, X_te_r, y_temp, y_te_r = train_test_split(
    X_reg_scaled, y_reg, test_size=0.20, random_state=42)
X_tr_r, X_val_r, y_tr_r, y_val_r = train_test_split(
    X_temp, y_temp, test_size=0.10, random_state=42)

print(f"  Regression:")
print(f"    Train : {X_tr_r.shape[0]:>6} rows  ({X_tr_r.shape[0]/len(df)*100:.1f}%)")
print(f"    Val   : {X_val_r.shape[0]:>6} rows  ({X_val_r.shape[0]/len(df)*100:.1f}%)")
print(f"    Test  : {X_te_r.shape[0]:>6} rows  ({X_te_r.shape[0]/len(df)*100:.1f}%)")

# ── Classification ──
X_cls = df[CLS_FEATURES]
y_cls = df['Crop_enc']

scaler_cls = StandardScaler()
X_cls_scaled = pd.DataFrame(scaler_cls.fit_transform(X_cls), columns=CLS_FEATURES)

X_tempc, X_te_c, y_tempc, y_te_c = train_test_split(
    X_cls_scaled, y_cls, test_size=0.20, random_state=42)
X_tr_c, X_val_c, y_tr_c, y_val_c = train_test_split(
    X_tempc, y_tempc, test_size=0.10, random_state=42)

print(f"  Classification:")
print(f"    Train : {X_tr_c.shape[0]:>6} rows  ({X_tr_c.shape[0]/len(df)*100:.1f}%)")
print(f"    Val   : {X_val_c.shape[0]:>6} rows  ({X_val_c.shape[0]/len(df)*100:.1f}%)")
print(f"    Test  : {X_te_c.shape[0]:>6} rows  ({X_te_c.shape[0]/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────
# STEP 9: Save
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Saving to Augmentation/Data/")
print("=" * 60)

# Regression
X_tr_r.to_csv('Augmentation/Data/X_reg_train.csv',  index=False)
X_val_r.to_csv('Augmentation/Data/X_reg_val.csv',   index=False)
X_te_r.to_csv('Augmentation/Data/X_reg_test.csv',   index=False)
y_tr_r.to_csv('Augmentation/Data/y_reg_train.csv',  index=False)
y_val_r.to_csv('Augmentation/Data/y_reg_val.csv',   index=False)
y_te_r.to_csv('Augmentation/Data/y_reg_test.csv',   index=False)

# Classification
X_tr_c.to_csv('Augmentation/Data/X_cls_train.csv',  index=False)
X_val_c.to_csv('Augmentation/Data/X_cls_val.csv',   index=False)
X_te_c.to_csv('Augmentation/Data/X_cls_test.csv',   index=False)
y_tr_c.to_csv('Augmentation/Data/y_cls_train.csv',  index=False)
y_val_c.to_csv('Augmentation/Data/y_cls_val.csv',   index=False)
y_te_c.to_csv('Augmentation/Data/y_cls_test.csv',   index=False)

# Scalers (needed for inference — must match training transform)
joblib.dump(scaler_reg, 'Augmentation/Data/scaler_reg.pkl')
joblib.dump(scaler_cls, 'Augmentation/Data/scaler_cls.pkl')

print("  ✅ Saved:")
for f in [
    'X_reg_train.csv','X_reg_val.csv','X_reg_test.csv',
    'y_reg_train.csv','y_reg_val.csv','y_reg_test.csv',
    'X_cls_train.csv','X_cls_val.csv','X_cls_test.csv',
    'y_cls_train.csv','y_cls_val.csv','y_cls_test.csv',
    'season_encoding.csv','crop_encoding.csv','district_encoding.csv',
    'scaler_reg.pkl','scaler_cls.pkl'
]:
    print(f"     ✅ {f}")

# ─────────────────────────────────────────────
# STEP 10: Summary Chart
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10: Generating Summary Chart")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle(
    'Preprocessing Summary — augmented_dataset.csv\n'
    'Team: Light Seekers | CSE-4889',
    fontsize=14, fontweight='bold'
)

# ── 1: Production raw ──
ax = axes[0, 0]
ax.hist(df['Production'], bins=60, color='#e74c3c', alpha=0.8, edgecolor='white')
ax.set_title('Production Distribution (Raw)', fontweight='bold')
ax.set_xlabel('Production'); ax.set_ylabel('Count')
ax.grid(axis='y', alpha=0.3)

# ── 2: Production log ──
ax = axes[0, 1]
ax.hist(df['Production_log'], bins=60, color='#2ecc71', alpha=0.8, edgecolor='white')
ax.set_title('Production_log Distribution (After log1p)', fontweight='bold')
ax.set_xlabel('log(1 + Production)'); ax.set_ylabel('Count')
ax.grid(axis='y', alpha=0.3)

# ── 3: Top 15 Crops ──
ax = axes[0, 2]
top_crops = df['Crop Name'].value_counts().head(15)
top_crops.plot(kind='barh', ax=ax, color='#3498db', edgecolor='white')
ax.set_title('Top 15 Crops by Row Count', fontweight='bold')
ax.set_xlabel('Count')
ax.grid(axis='x', alpha=0.3)

# ── 4: Season distribution ──
ax = axes[1, 0]
season_counts = df['Season'].value_counts()
ax.pie(season_counts.values,
       labels=season_counts.index,
       autopct='%1.1f%%',
       colors=['#2ecc71', '#3498db', '#e67e22'],
       startangle=90)
ax.set_title('Season Distribution', fontweight='bold')

# ── 5: Train/Val/Test split ──
ax = axes[1, 1]
split_labels = ['Train', 'Val', 'Test']
split_sizes  = [X_tr_r.shape[0], X_val_r.shape[0], X_te_r.shape[0]]
split_colors = ['#2ecc71', '#f39c12', '#e74c3c']
bars = ax.bar(split_labels, split_sizes, color=split_colors,
              width=0.5, edgecolor='white', linewidth=1.5)
for b, v in zip(bars, split_sizes):
    ax.text(b.get_x() + b.get_width()/2, v + 50,
            f'{v:,}\n({v/len(df)*100:.1f}%)',
            ha='center', fontweight='bold', fontsize=11)
ax.set_title('Train / Val / Test Split (Regression)', fontweight='bold')
ax.set_ylabel('Row Count')
ax.grid(axis='y', alpha=0.3)

# ── 6: Feature correlation ──
ax = axes[1, 2]
corr_cols = ['Area', 'N', 'P', 'K', 'ph', 'Avg Temp', 'Avg Humidity',
             'Rainfall', 'Production_log']
corr = df[corr_cols].corr()[['Production_log']].drop('Production_log')
colors_list = ['#e74c3c' if v < 0 else '#2ecc71' for v in corr['Production_log']]
bars = ax.barh(corr.index, corr['Production_log'],
               color=colors_list, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Feature Correlation with Production_log', fontweight='bold')
ax.set_xlabel('Pearson Correlation')
ax.grid(axis='x', alpha=0.3)
for b, v in zip(bars, corr['Production_log']):
    ax.text(v + (0.005 if v >= 0 else -0.005),
            b.get_y() + b.get_height()/2,
            f'{v:.3f}', va='center',
            ha='left' if v >= 0 else 'right', fontsize=9)

plt.tight_layout()
plt.savefig('Augmentation/Data/preprocess_summary.png', dpi=150, bbox_inches='tight')
print("  ✅ Saved: Augmentation/Data/preprocess_summary.png")

# ─────────────────────────────────────────────
# Final Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE ✅")
print("=" * 60)
print(f"  Input rows       : 50,000")
print(f"  After cleaning   : {len(df)}")
print(f"  Unique crops     : {df['Crop Name'].nunique()}")
print(f"  Unique districts : {df['District'].nunique()}")
print(f"  Unique seasons   : {df['Season'].nunique()}")
print(f"  Reg features     : {len(REG_FEATURES)}")
print(f"  Cls features     : {len(CLS_FEATURES)}")
print(f"\n  Files saved in Augmentation/Data/:")
for f in [
    'X_reg_train.csv','X_reg_val.csv','X_reg_test.csv',
    'y_reg_train.csv','y_reg_val.csv','y_reg_test.csv',
    'X_cls_train.csv','X_cls_val.csv','X_cls_test.csv',
    'y_cls_train.csv','y_cls_val.csv','y_cls_test.csv',
    'season_encoding.csv','crop_encoding.csv','district_encoding.csv',
    'scaler_reg.pkl','scaler_cls.pkl',
    'preprocess_summary.png'
]:
    print(f"    ✅ {f}")
print("=" * 60)