"""
=============================================================
  RF + GRADIENT BOOSTING TRAINING
  Team: Light Seekers | Course: CSE-4889
  Input : Data/Marge/merged_dataset.csv
  Output: rf_gb_comparison.png
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score
)

# ─────────────────────────────────────────────
# STEP 1: Load & Clean
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading merged_dataset.csv")
print("=" * 60)

df = pd.read_csv('../Data/Marge/merged_dataset.csv')
print(f"Raw shape: {df.shape}")

# Drop leaky / non-feature columns
df.drop(columns=['Transplant', 'Growth', 'Harvest', 'AP Ratio'], inplace=True)

# Remove zero production rows
df = df[df['Production'] > 0].copy()

# IQR outlier removal on Production
Q1 = df['Production'].quantile(0.25)
Q3 = df['Production'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Production'] >= Q1 - 3 * IQR) & (df['Production'] <= Q3 + 3 * IQR)]
df.reset_index(drop=True, inplace=True)

print(f"After cleaning: {df.shape}")

# ─────────────────────────────────────────────
# STEP 2: Encode + Target Transform
# ─────────────────────────────────────────────
print("\nSTEP 2: Encoding & Feature Engineering")

le_season   = LabelEncoder()
le_district = LabelEncoder()
le_crop     = LabelEncoder()

df['Season_enc']     = le_season.fit_transform(df['Season'])
df['District_enc']   = le_district.fit_transform(df['District'])
df['Crop_enc']       = le_crop.fit_transform(df['Crop Name'])
df['Production_log'] = np.log1p(df['Production'])

print(f"  Unique crops    : {df['Crop Name'].nunique()}")
print(f"  Unique districts: {df['District'].nunique()}")
print(f"  Unique seasons  : {df['Season'].nunique()}")

# ─────────────────────────────────────────────
# STEP 3: Feature Sets & Split
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

# Regression
X_reg = df[REG_FEATURES]
y_reg = df['Production_log']
scaler_reg = StandardScaler()
X_reg_scaled = pd.DataFrame(scaler_reg.fit_transform(X_reg), columns=REG_FEATURES)
X_tr, X_te, y_tr, y_te = train_test_split(X_reg_scaled, y_reg, test_size=0.2, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.1, random_state=42)

# Classification
X_cls = df[CLS_FEATURES]
y_cls = df['Crop_enc']
scaler_cls = StandardScaler()
X_cls_scaled = pd.DataFrame(scaler_cls.fit_transform(X_cls), columns=CLS_FEATURES)
X_trc, X_tec, y_trc, y_tec = train_test_split(X_cls_scaled, y_cls, test_size=0.2, random_state=42)
X_trc, X_valc, y_trc, y_valc = train_test_split(X_trc, y_trc, test_size=0.1, random_state=42)

print(f"\n  Regression  — Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_te)}")
print(f"  Classification — Train: {len(X_trc)}, Val: {len(X_valc)}, Test: {len(X_tec)}")

# ─────────────────────────────────────────────
# STEP 4: Train Models
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Training Models")
print("=" * 60)

results_reg = {}
results_cls = {}

# ── RF Regressor ──
print("\n[1/4] Random Forest Regressor...")
rf_reg = RandomForestRegressor(n_estimators=150, max_depth=20,
                                min_samples_leaf=2, n_jobs=-1, random_state=42)
rf_reg.fit(X_tr, y_tr)
pred = rf_reg.predict(X_te)
results_reg['Random Forest'] = dict(
    r2=r2_score(y_te, pred),
    rmse=np.sqrt(mean_squared_error(y_te, pred)),
    mae=mean_absolute_error(y_te, pred),
    feat_imp=rf_reg.feature_importances_
)
print(f"   ✅ R²={results_reg['Random Forest']['r2']:.4f}  "
      f"RMSE={results_reg['Random Forest']['rmse']:.4f}  "
      f"MAE={results_reg['Random Forest']['mae']:.4f}")

# ── GB Regressor ──
print("\n[2/4] Gradient Boosting Regressor...")
gb_reg = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05,
                                        max_depth=6, random_state=42)
gb_reg.fit(X_tr, y_tr)
pred = gb_reg.predict(X_te)
results_reg['Gradient Boosting'] = dict(
    r2=r2_score(y_te, pred),
    rmse=np.sqrt(mean_squared_error(y_te, pred)),
    mae=mean_absolute_error(y_te, pred)
)
print(f"   ✅ R²={results_reg['Gradient Boosting']['r2']:.4f}  "
      f"RMSE={results_reg['Gradient Boosting']['rmse']:.4f}  "
      f"MAE={results_reg['Gradient Boosting']['mae']:.4f}")

# ── RF Classifier ──
print("\n[3/4] Random Forest Classifier...")
rf_cls = RandomForestClassifier(n_estimators=150, max_depth=20,
                                  min_samples_leaf=2, class_weight='balanced',
                                  n_jobs=-1, random_state=42)
rf_cls.fit(X_trc, y_trc)
pred = rf_cls.predict(X_tec)
results_cls['Random Forest'] = dict(
    acc=accuracy_score(y_tec, pred),
    f1=f1_score(y_tec, pred, average='weighted', zero_division=0),
    feat_imp=rf_cls.feature_importances_
)
print(f"   ✅ Accuracy={results_cls['Random Forest']['acc']:.4f}  "
      f"F1={results_cls['Random Forest']['f1']:.4f}")

# ── GB Classifier ──
print("\n[4/4] Gradient Boosting Classifier...")
gb_cls = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05,
                                          max_depth=6, random_state=42)
gb_cls.fit(X_trc, y_trc)
pred = gb_cls.predict(X_tec)
results_cls['Gradient Boosting'] = dict(
    acc=accuracy_score(y_tec, pred),
    f1=f1_score(y_tec, pred, average='weighted', zero_division=0)
)
print(f"   ✅ Accuracy={results_cls['Gradient Boosting']['acc']:.4f}  "
      f"F1={results_cls['Gradient Boosting']['f1']:.4f}")

# ─────────────────────────────────────────────
# STEP 5: Comparison Chart
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Generating Comparison Chart")
print("=" * 60)

models  = ['Random Forest', 'Gradient Boosting']
colors  = ['#2ecc71', '#3498db']
x       = np.arange(len(models))
w       = 0.35

fig, axes = plt.subplots(2, 3, figsize=(17, 11))
fig.suptitle(
    'RF vs Gradient Boosting — Model Comparison\n'
    'Trained on merged_dataset.csv  |  Team: Light Seekers | CSE-4889',
    fontsize=14, fontweight='bold', y=0.98
)

def bar_labels(ax, bars):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.005,
                f'{h:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# ── A1: R² ──
ax = axes[0, 0]
vals = [results_reg[m]['r2'] for m in models]
bars = ax.bar(models, vals, color=colors, width=0.45, edgecolor='white', linewidth=1.5)
ax.set_title('Task A — R² Score (↑ higher better)', fontweight='bold')
ax.set_ylim(0, 1.1); ax.set_ylabel('R²')
bar_labels(ax, bars)
ax.grid(axis='y', alpha=0.3)
best_idx = int(np.argmax(vals))
bars[best_idx].set_edgecolor('gold'); bars[best_idx].set_linewidth(3)

# ── A2: RMSE ──
ax = axes[0, 1]
vals = [results_reg[m]['rmse'] for m in models]
bars = ax.bar(models, vals, color=colors, width=0.45, edgecolor='white', linewidth=1.5)
ax.set_title('Task A — RMSE (↓ lower better)', fontweight='bold')
ax.set_ylabel('RMSE')
bar_labels(ax, bars)
ax.grid(axis='y', alpha=0.3)
best_idx = int(np.argmin(vals))
bars[best_idx].set_edgecolor('gold'); bars[best_idx].set_linewidth(3)

# ── A3: MAE ──
ax = axes[0, 2]
vals = [results_reg[m]['mae'] for m in models]
bars = ax.bar(models, vals, color=colors, width=0.45, edgecolor='white', linewidth=1.5)
ax.set_title('Task A — MAE (↓ lower better)', fontweight='bold')
ax.set_ylabel('MAE')
bar_labels(ax, bars)
ax.grid(axis='y', alpha=0.3)
best_idx = int(np.argmin(vals))
bars[best_idx].set_edgecolor('gold'); bars[best_idx].set_linewidth(3)

# ── B1: Accuracy ──
ax = axes[1, 0]
vals = [results_cls[m]['acc'] for m in models]
bars = ax.bar(models, vals, color=colors, width=0.45, edgecolor='white', linewidth=1.5)
ax.set_title('Task B — Accuracy (↑ higher better)', fontweight='bold')
ax.set_ylim(0, 1.1); ax.set_ylabel('Accuracy')
bar_labels(ax, bars)
ax.grid(axis='y', alpha=0.3)
best_idx = int(np.argmax(vals))
bars[best_idx].set_edgecolor('gold'); bars[best_idx].set_linewidth(3)

# ── B2: F1 Score ──
ax = axes[1, 1]
vals = [results_cls[m]['f1'] for m in models]
bars = ax.bar(models, vals, color=colors, width=0.45, edgecolor='white', linewidth=1.5)
ax.set_title('Task B — F1 Score Weighted (↑ higher better)', fontweight='bold')
ax.set_ylim(0, 1.1); ax.set_ylabel('F1 Score')
bar_labels(ax, bars)
ax.grid(axis='y', alpha=0.3)
best_idx = int(np.argmax(vals))
bars[best_idx].set_edgecolor('gold'); bars[best_idx].set_linewidth(3)

# ── B3: Feature Importance (RF Classifier) ──
ax = axes[1, 2]
fi = pd.Series(rf_cls.feature_importances_, index=CLS_FEATURES).sort_values()
fi.plot(kind='barh', ax=ax, color='#9b59b6', edgecolor='white')
ax.set_title('Task B — Feature Importance\n(Random Forest Classifier)', fontweight='bold')
ax.set_xlabel('Importance Score')
ax.grid(axis='x', alpha=0.3)

# Legend
rf_patch = mpatches.Patch(color='#2ecc71', label='Random Forest')
gb_patch = mpatches.Patch(color='#3498db', label='Gradient Boosting')
gold_patch = mpatches.Patch(edgecolor='gold', facecolor='none',
                             linewidth=2, label='Best Model (gold border)')
fig.legend(handles=[rf_patch, gb_patch, gold_patch],
           loc='lower center', ncol=3, fontsize=11,
           bbox_to_anchor=(0.5, 0.01), framealpha=0.9)

plt.tight_layout(rect=[0, 0.05, 1, 0.97])
plt.savefig('rf_gb_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Saved: rf_gb_comparison.png")

# ─────────────────────────────────────────────
# STEP 6: Final Summary Table
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL RESULT SUMMARY")
print("=" * 60)

best_reg = max(results_reg, key=lambda m: results_reg[m]['r2'])
best_cls = max(results_cls, key=lambda m: results_cls[m]['f1'])

print(f"\n  Task A — Regression")
print(f"  {'Model':<22} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
print("  " + "-" * 50)
for name, res in results_reg.items():
    tag = " ✅ BEST" if name == best_reg else ""
    print(f"  {name:<22} {res['r2']:>8.4f} {res['rmse']:>8.4f} {res['mae']:>8.4f}{tag}")

print(f"\n  Task B — Classification")
print(f"  {'Model':<22} {'Accuracy':>10} {'F1':>8}")
print("  " + "-" * 44)
for name, res in results_cls.items():
    tag = " ✅ BEST" if name == best_cls else ""
    print(f"  {name:<22} {res['acc']:>10.4f} {res['f1']:>8.4f}{tag}")

print("\n" + "=" * 60)
print("Training সম্পন্ন! ✅")
print("=" * 60)
