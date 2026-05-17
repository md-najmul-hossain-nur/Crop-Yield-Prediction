"""
=============================================================
  RF + GRADIENT BOOSTING TRAINING
  Team: Light Seekers | Course: CSE-4889

  Input : Augmentation/Data/ (augmented + preprocessed CSVs)
  Output: Augmentation/aug_rf_gb_comparison.png
         Augmentation/models/rf_reg.pkl
         Augmentation/models/gb_reg.pkl
         Augmentation/models/rf_cls.pkl
         Augmentation/models/gb_cls.pkl

  Models:
    Task A (Regression)      — RF Regressor, GB Regressor
    Task B (Classification)  — RF Classifier, GB Classifier
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import os
import joblib
warnings.filterwarnings('ignore')

from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier
)
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score
)

DATA_DIR  = "Augmentation/Data"
MODEL_DIR = "Augmentation/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1: Load Preprocessed Augmented Data
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Augmented Preprocessed Data")
print("=" * 60)

# Regression
X_tr_r  = pd.read_csv(f"{DATA_DIR}/X_reg_train.csv")
X_val_r = pd.read_csv(f"{DATA_DIR}/X_reg_val.csv")
X_te_r  = pd.read_csv(f"{DATA_DIR}/X_reg_test.csv")
y_tr_r  = pd.read_csv(f"{DATA_DIR}/y_reg_train.csv").squeeze()
y_val_r = pd.read_csv(f"{DATA_DIR}/y_reg_val.csv").squeeze()
y_te_r  = pd.read_csv(f"{DATA_DIR}/y_reg_test.csv").squeeze()

# Classification
X_tr_c  = pd.read_csv(f"{DATA_DIR}/X_cls_train.csv")
X_val_c = pd.read_csv(f"{DATA_DIR}/X_cls_val.csv")
X_te_c  = pd.read_csv(f"{DATA_DIR}/X_cls_test.csv")
y_tr_c  = pd.read_csv(f"{DATA_DIR}/y_cls_train.csv").squeeze()
y_val_c = pd.read_csv(f"{DATA_DIR}/y_cls_val.csv").squeeze()
y_te_c  = pd.read_csv(f"{DATA_DIR}/y_cls_test.csv").squeeze()

REG_FEATURES = list(X_tr_r.columns)
CLS_FEATURES = list(X_tr_c.columns)

print(f"  Regression features  ({len(REG_FEATURES)}) : {REG_FEATURES}")
print(f"  Classification features ({len(CLS_FEATURES)}) : {CLS_FEATURES}")
print(f"\n  Regression     — Train: {len(X_tr_r):,}, Val: {len(X_val_r):,}, Test: {len(X_te_r):,}")
print(f"  Classification — Train: {len(X_tr_c):,}, Val: {len(X_val_c):,}, Test: {len(X_te_c):,}")

# ─────────────────────────────────────────────
# STEP 2: Train Models
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Training Models")
print("=" * 60)

results_reg = {}
results_cls = {}

# ── RF Regressor ──
print("\n[1/4] Random Forest Regressor...")
rf_reg = RandomForestRegressor(
    n_estimators=150, max_depth=20,
    min_samples_leaf=2, n_jobs=-1, random_state=42
)
rf_reg.fit(X_tr_r, y_tr_r)
pred = rf_reg.predict(X_te_r)
results_reg['Random Forest'] = dict(
    r2=r2_score(y_te_r, pred),
    rmse=np.sqrt(mean_squared_error(y_te_r, pred)),
    mae=mean_absolute_error(y_te_r, pred),
    feat_imp=rf_reg.feature_importances_
)
joblib.dump(rf_reg, f"{MODEL_DIR}/rf_reg.pkl")
print(f"   ✅ R²={results_reg['Random Forest']['r2']:.4f}  "
      f"RMSE={results_reg['Random Forest']['rmse']:.4f}  "
      f"MAE={results_reg['Random Forest']['mae']:.4f}")
print(f"   💾 Saved: {MODEL_DIR}/rf_reg.pkl")

# ── GB Regressor ──
print("\n[2/4] Gradient Boosting Regressor...")
gb_reg = HistGradientBoostingRegressor(
    max_iter=150, learning_rate=0.05,
    max_depth=6, random_state=42
)
gb_reg.fit(X_tr_r, y_tr_r)
pred = gb_reg.predict(X_te_r)
results_reg['Gradient Boosting'] = dict(
    r2=r2_score(y_te_r, pred),
    rmse=np.sqrt(mean_squared_error(y_te_r, pred)),
    mae=mean_absolute_error(y_te_r, pred)
)
joblib.dump(gb_reg, f"{MODEL_DIR}/gb_reg.pkl")
print(f"   ✅ R²={results_reg['Gradient Boosting']['r2']:.4f}  "
      f"RMSE={results_reg['Gradient Boosting']['rmse']:.4f}  "
      f"MAE={results_reg['Gradient Boosting']['mae']:.4f}")
print(f"   💾 Saved: {MODEL_DIR}/gb_reg.pkl")

# ── RF Classifier ──
print("\n[3/4] Random Forest Classifier...")
rf_cls = RandomForestClassifier(
    n_estimators=150, max_depth=20,
    min_samples_leaf=2, class_weight='balanced',
    n_jobs=-1, random_state=42
)
rf_cls.fit(X_tr_c, y_tr_c)
pred = rf_cls.predict(X_te_c)
results_cls['Random Forest'] = dict(
    acc=accuracy_score(y_te_c, pred),
    f1=f1_score(y_te_c, pred, average='weighted', zero_division=0),
    feat_imp=rf_cls.feature_importances_
)
joblib.dump(rf_cls, f"{MODEL_DIR}/rf_cls.pkl")
print(f"   ✅ Accuracy={results_cls['Random Forest']['acc']:.4f}  "
      f"F1={results_cls['Random Forest']['f1']:.4f}")
print(f"   💾 Saved: {MODEL_DIR}/rf_cls.pkl")

# ── GB Classifier ──
print("\n[4/4] Gradient Boosting Classifier...")
gb_cls = HistGradientBoostingClassifier(
    max_iter=150, learning_rate=0.05,
    max_depth=6, random_state=42
)
gb_cls.fit(X_tr_c, y_tr_c)
pred = gb_cls.predict(X_te_c)
results_cls['Gradient Boosting'] = dict(
    acc=accuracy_score(y_te_c, pred),
    f1=f1_score(y_te_c, pred, average='weighted', zero_division=0)
)
joblib.dump(gb_cls, f"{MODEL_DIR}/gb_cls.pkl")
print(f"   ✅ Accuracy={results_cls['Gradient Boosting']['acc']:.4f}  "
      f"F1={results_cls['Gradient Boosting']['f1']:.4f}")
print(f"   💾 Saved: {MODEL_DIR}/gb_cls.pkl")

# ─────────────────────────────────────────────
# STEP 3: Save results for DNN comparison
# ─────────────────────────────────────────────
np.save(f"{MODEL_DIR}/rf_gb_results.npy", {
    'reg': {k: {ki: v for ki, v in val.items() if ki != 'feat_imp'}
            for k, val in results_reg.items()},
    'cls': {k: {ki: v for ki, v in val.items() if ki != 'feat_imp'}
            for k, val in results_cls.items()}
})
print(f"\n  💾 Saved: {MODEL_DIR}/rf_gb_results.npy  (for DNN comparison)")

# ─────────────────────────────────────────────
# STEP 4: Comparison Chart
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Generating Comparison Chart")
print("=" * 60)

models = ['Random Forest', 'Gradient Boosting']
colors = ['#2ecc71', '#3498db']

fig, axes = plt.subplots(2, 3, figsize=(17, 11))
fig.suptitle(
    'RF vs Gradient Boosting — Model Comparison\n'
    'Input: Augmented Data  |  Team: Light Seekers | CSE-4889',
    fontsize=14, fontweight='bold', y=0.98
)

def bar_labels(ax, bars):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.005,
                f'{h:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

def highlight_best(bars, vals, mode='max'):
    idx = int(np.argmax(vals)) if mode == 'max' else int(np.argmin(vals))
    bars[idx].set_edgecolor('gold')
    bars[idx].set_linewidth(3)

# ── A1: R² ──
ax = axes[0, 0]
vals = [results_reg[m]['r2'] for m in models]
bars = ax.bar(models, vals, color=colors, width=0.45, edgecolor='white', linewidth=1.5)
ax.set_title('Task A — R² Score (↑ higher better)', fontweight='bold')
ax.set_ylim(0, 1.1); ax.set_ylabel('R²')
bar_labels(ax, bars); highlight_best(bars, vals, 'max')
ax.grid(axis='y', alpha=0.3)

# ── A2: RMSE ──
ax = axes[0, 1]
vals = [results_reg[m]['rmse'] for m in models]
bars = ax.bar(models, vals, color=colors, width=0.45, edgecolor='white', linewidth=1.5)
ax.set_title('Task A — RMSE (↓ lower better)', fontweight='bold')
ax.set_ylabel('RMSE')
bar_labels(ax, bars); highlight_best(bars, vals, 'min')
ax.grid(axis='y', alpha=0.3)

# ── A3: MAE ──
ax = axes[0, 2]
vals = [results_reg[m]['mae'] for m in models]
bars = ax.bar(models, vals, color=colors, width=0.45, edgecolor='white', linewidth=1.5)
ax.set_title('Task A — MAE (↓ lower better)', fontweight='bold')
ax.set_ylabel('MAE')
bar_labels(ax, bars); highlight_best(bars, vals, 'min')
ax.grid(axis='y', alpha=0.3)

# ── B1: Accuracy ──
ax = axes[1, 0]
vals = [results_cls[m]['acc'] for m in models]
bars = ax.bar(models, vals, color=colors, width=0.45, edgecolor='white', linewidth=1.5)
ax.set_title('Task B — Accuracy (↑ higher better)', fontweight='bold')
ax.set_ylim(0, 1.1); ax.set_ylabel('Accuracy')
bar_labels(ax, bars); highlight_best(bars, vals, 'max')
ax.grid(axis='y', alpha=0.3)

# ── B2: F1 Score ──
ax = axes[1, 1]
vals = [results_cls[m]['f1'] for m in models]
bars = ax.bar(models, vals, color=colors, width=0.45, edgecolor='white', linewidth=1.5)
ax.set_title('Task B — F1 Score Weighted (↑ higher better)', fontweight='bold')
ax.set_ylim(0, 1.1); ax.set_ylabel('F1 Score')
bar_labels(ax, bars); highlight_best(bars, vals, 'max')
ax.grid(axis='y', alpha=0.3)

# ── B3: Feature Importance (RF Classifier) ──
ax = axes[1, 2]
fi = pd.Series(rf_cls.feature_importances_, index=CLS_FEATURES).sort_values()
fi.plot(kind='barh', ax=ax, color='#9b59b6', edgecolor='white')
ax.set_title('Task B — Feature Importance\n(Random Forest Classifier)', fontweight='bold')
ax.set_xlabel('Importance Score')
ax.grid(axis='x', alpha=0.3)

# Legend
rf_patch   = mpatches.Patch(color='#2ecc71', label='Random Forest')
gb_patch   = mpatches.Patch(color='#3498db', label='Gradient Boosting')
gold_patch = mpatches.Patch(edgecolor='gold', facecolor='none',
                             linewidth=2, label='Best Model (gold border)')
fig.legend(handles=[rf_patch, gb_patch, gold_patch],
           loc='lower center', ncol=3, fontsize=11,
           bbox_to_anchor=(0.5, 0.01), framealpha=0.9)

plt.tight_layout(rect=[0, 0.05, 1, 0.97])
plt.savefig('Augmentation/aug_rf_gb_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Saved: Augmentation/aug_rf_gb_comparison.png")

# ─────────────────────────────────────────────
# STEP 5: Final Summary Table
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
print(f"  Best Regression    : {best_reg}")
print(f"  Best Classification: {best_cls}")
print("=" * 60)
print("Training সম্পন্ন! ✅")
print("→ Chart  : Augmentation/aug_rf_gb_comparison.png")
print("→ Models : Augmentation/models/ (rf_reg, gb_reg, rf_cls, gb_cls).pkl")
print("→ Results: Augmentation/models/rf_gb_results.npy  ← DNN এ use হবে")
