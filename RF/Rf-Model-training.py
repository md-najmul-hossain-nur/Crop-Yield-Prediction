"""
=============================================================
  MODEL TRAINING — Crop Yield Prediction
  Team: Light Seekers | Course: CSE-4889

  Model 1: Random Forest (Baseline)
  Model 2: Gradient Boosting (HistGradientBoosting — fast version)

  Task A: Yield Prediction     → Regression
  Task B: Crop Recommendation  → Classification
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score
)

# ─────────────────────────────────────────────
# STEP 0: Data Load
# ─────────────────────────────────────────────
print("=" * 65)
print("STEP 0: Data Load")
print("=" * 65)
#  FIXED: preprocess/RF folder থেকে load
X_train_r = pd.read_csv('preprocess/RF/X_reg_train.csv')
X_test_r  = pd.read_csv('preprocess/RF/X_reg_test.csv')
y_train_r = pd.read_csv('preprocess/RF/y_reg_train.csv').squeeze()
y_test_r  = pd.read_csv('preprocess/RF/y_reg_test.csv').squeeze()

X_train_c = pd.read_csv('preprocess/RF/X_cls_train.csv')
X_test_c  = pd.read_csv('preprocess/RF/X_cls_test.csv')
y_train_c = pd.read_csv('preprocess/RF/y_cls_train.csv').squeeze()
y_test_c  = pd.read_csv('preprocess/RF/y_cls_test.csv').squeeze()

# Validation — train থেকে 10% split
X_train_r, X_val_r, y_train_r, y_val_r = train_test_split(
    X_train_r, y_train_r, test_size=0.10, random_state=42)
X_train_c, X_val_c, y_train_c, y_val_c = train_test_split(
    X_train_c, y_train_c, test_size=0.10, random_state=42)

print(f"Train (Reg) : {X_train_r.shape[0]:,} rows")
print(f"Val   (Reg) : {X_val_r.shape[0]:,} rows")
print(f"Test  (Reg) : {X_test_r.shape[0]:,} rows")
print(f"Train (Cls) : {X_train_c.shape[0]:,} rows")
print(f"Val   (Cls) : {X_val_c.shape[0]:,} rows")
print(f"Test  (Cls) : {X_test_c.shape[0]:,} rows")

# Feature names (columns)
REG_FEATURES = list(X_train_r.columns)
CLS_FEATURES = list(X_train_c.columns)

# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────
def reg_report(name, y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    print(f"  {name:<12} R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")
    return r2, rmse, mae

def cls_report(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"  {name:<12} Accuracy={acc:.4f}  F1={f1:.4f}")
    return acc, f1

# ══════════════════════════════════════════════
#   TASK A — REGRESSION
# ══════════════════════════════════════════════
print("\n" + "=" * 65)
print("TASK A: REGRESSION — Yield Prediction (Production_log)")
print("=" * 65)
reg_results = {}

# Model 1: Random Forest Regressor
print("\n[1] Random Forest Regressor")
rf_reg = RandomForestRegressor(
    n_estimators=150, max_depth=20,
    min_samples_leaf=2, n_jobs=-1, random_state=42
)
rf_reg.fit(X_train_r, y_train_r)
v1, _, _         = reg_report("Validation", y_val_r,  rf_reg.predict(X_val_r))
t_r2, t_rmse, t_mae = reg_report("Test",   y_test_r, rf_reg.predict(X_test_r))
reg_results['Random Forest'] = dict(val_r2=v1, test_r2=t_r2, test_rmse=t_rmse, test_mae=t_mae,
                                     feat_imp=rf_reg.feature_importances_)

# Model 2: Gradient Boosting Regressor
print("\n[2] Gradient Boosting Regressor")
gb_reg = HistGradientBoostingRegressor(
    max_iter=150, learning_rate=0.05, max_depth=6, random_state=42
)
gb_reg.fit(X_train_r, y_train_r)
v1, _, _         = reg_report("Validation", y_val_r,  gb_reg.predict(X_val_r))
t_r2, t_rmse, t_mae = reg_report("Test",   y_test_r, gb_reg.predict(X_test_r))
reg_results['Gradient Boosting'] = dict(val_r2=v1, test_r2=t_r2, test_rmse=t_rmse, test_mae=t_mae)

# Summary
print("\n  REGRESSION SUMMARY (Test Set)")
print(f"  {'Model':<22} {'Val R²':>8} {'Test R²':>8} {'RMSE':>8} {'MAE':>8}")
print("  " + "-"*58)
for name, res in reg_results.items():
    print(f"  {name:<22} {res['val_r2']:>8.4f} {res['test_r2']:>8.4f} "
          f"{res['test_rmse']:>8.4f} {res['test_mae']:>8.4f}")
best_reg = max(reg_results, key=lambda m: reg_results[m]['test_r2'])
print(f"\n  Best: {best_reg}  (R²={reg_results[best_reg]['test_r2']:.4f})")
# ══════════════════════════════════════════════
#   TASK B — CLASSIFICATION
# ══════════════════════════════════════════════
print("\n" + "=" * 65)
print("TASK B: CLASSIFICATION — Crop Recommendation (72 classes)")
print("=" * 65)
cls_results = {}

# Model 1: Random Forest Classifier
print("\n[1] Random Forest Classifier")
rf_cls = RandomForestClassifier(
    n_estimators=150, max_depth=20, min_samples_leaf=2,
    class_weight='balanced', n_jobs=-1, random_state=42
)
rf_cls.fit(X_train_c, y_train_c)
v1, _        = cls_report("Validation", y_val_c,  rf_cls.predict(X_val_c))
t_acc, t_f1  = cls_report("Test",       y_test_c, rf_cls.predict(X_test_c))
cls_results['Random Forest'] = dict(val_acc=v1, test_acc=t_acc, test_f1=t_f1,
                                     feat_imp=rf_cls.feature_importances_)

# Model 2: Gradient Boosting Classifier
print("\n[2] Gradient Boosting Classifier")
gb_cls = HistGradientBoostingClassifier(
    max_iter=150, learning_rate=0.05, max_depth=6, random_state=42
)
gb_cls.fit(X_train_c, y_train_c)
v1, _        = cls_report("Validation", y_val_c,  gb_cls.predict(X_val_c))
t_acc, t_f1  = cls_report("Test",       y_test_c, gb_cls.predict(X_test_c))
cls_results['Gradient Boosting'] = dict(val_acc=v1, test_acc=t_acc, test_f1=t_f1)

# Summary
print("\n  CLASSIFICATION SUMMARY (Test Set)")
print(f"  {'Model':<22} {'Val Acc':>9} {'Test Acc':>9} {'F1':>8}")
print("  " + "-"*52)
for name, res in cls_results.items():
    print(f"  {name:<22} {res['val_acc']:>9.4f} {res['test_acc']:>9.4f} {res['test_f1']:>8.4f}")
best_cls = max(cls_results, key=lambda m: cls_results[m]['test_f1'])
print(f"\n  Best: {best_cls}  (F1={cls_results[best_cls]['test_f1']:.4f})")
# ══════════════════════════════════════════════
#   CHARTS
# ══════════════════════════════════════════════
print("\n" + "=" * 65)
print("Charts তৈরি হচ্ছে...")
print("=" * 65)

models = list(reg_results.keys())
colors = ['#2ecc71', '#3498db']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Model Comparison — Crop Yield Prediction\nTeam: Light Seekers | CSE-4889',
             fontsize=14, fontweight='bold')

# A1: R²
ax = axes[0, 0]
vals = [reg_results[m]['test_r2'] for m in models]
bars = ax.bar(models, vals, color=colors, width=0.4, edgecolor='white')
ax.set_title('Task A — R² Score', fontweight='bold'); ax.set_ylim(0, 1.1)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.4f}', ha='center', fontweight='bold')

# A2: RMSE
ax = axes[0, 1]
vals = [reg_results[m]['test_rmse'] for m in models]
bars = ax.bar(models, vals, color=colors, width=0.4, edgecolor='white')
ax.set_title('Task A — RMSE (lower=better)', fontweight='bold')
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.005, f'{v:.4f}', ha='center', fontweight='bold')

# A3: Actual vs Predicted
ax = axes[0, 2]
preds = rf_reg.predict(X_test_r) if best_reg == 'Random Forest' else gb_reg.predict(X_test_r)
ax.scatter(y_test_r, preds, alpha=0.5, color='#3498db', s=20)
mn = min(float(y_test_r.min()), float(preds.min()))
mx = max(float(y_test_r.max()), float(preds.max()))
ax.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Ideal')
ax.set_title(f'Task A — Actual vs Predicted\n({best_reg})', fontweight='bold')
ax.set_xlabel('Actual'); ax.set_ylabel('Predicted'); ax.legend()

# B1: Accuracy
ax = axes[1, 0]
vals = [cls_results[m]['test_acc'] for m in models]
bars = ax.bar(models, vals, color=colors, width=0.4, edgecolor='white')
ax.set_title('Task B — Accuracy', fontweight='bold'); ax.set_ylim(0, 1.1)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.4f}', ha='center', fontweight='bold')

# B2: F1
ax = axes[1, 1]
vals = [cls_results[m]['test_f1'] for m in models]
bars = ax.bar(models, vals, color=colors, width=0.4, edgecolor='white')
ax.set_title('Task B — F1 Score (Weighted)', fontweight='bold'); ax.set_ylim(0, 1.1)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.4f}', ha='center', fontweight='bold')

# B3: Feature Importance
ax = axes[1, 2]
fi = pd.Series(rf_cls.feature_importances_, index=CLS_FEATURES).sort_values()
fi.plot(kind='barh', ax=ax, color='#2ecc71', edgecolor='white')
ax.set_title('Task B — Feature Importance\n(Random Forest)', fontweight='bold')
ax.set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('rf_model_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: rf_model_comparison.png")

# Feature Importance — Regression
fig2, ax2 = plt.subplots(figsize=(10, 6))
fi_r = pd.Series(rf_reg.feature_importances_, index=REG_FEATURES).sort_values()
fi_r.plot(kind='barh', ax=ax2, color='#3498db', edgecolor='white')
ax2.set_title('Task A — Feature Importance (Random Forest Regressor)\n'
              'Team: Light Seekers | CSE-4889', fontweight='bold')
ax2.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('rf_feature_importance_regression.png', dpi=150, bbox_inches='tight')
print("Saved: rf_feature_importance_regression.png")


# ══════════════════════════════════════════════
#   FINAL TABLE
# ══════════════════════════════════════════════
print("\n" + "=" * 65)
print("FINAL RESULT TABLE")
print("=" * 65)
print(f"\n  Task A — Regression")
print(f"  {'Model':<22} {'Val R²':>8} {'Test R²':>8} {'RMSE':>8} {'MAE':>8}")
print("  " + "-"*56)
for name, res in reg_results.items():
    tag = " ← BEST" if name == best_reg else ""
    print(f"  {name:<22} {res['val_r2']:>8.4f} {res['test_r2']:>8.4f} "
          f"{res['test_rmse']:>8.4f} {res['test_mae']:>8.4f}{tag}")

print(f"\n  Task B — Classification")
print(f"  {'Model':<22} {'Val Acc':>8} {'Test Acc':>9} {'F1':>8}")
print("  " + "-"*50)
for name, res in cls_results.items():
    tag = " ← BEST" if name == best_cls else ""
    print(f"  {name:<22} {res['val_acc']:>8.4f} {res['test_acc']:>9.4f} "
          f"{res['test_f1']:>8.4f}{tag}")

print("\n" + "=" * 65)
print(f"  Best Regression    : {best_reg}")
print(f"  Best Classification: {best_cls}")
print("=" * 65)
print("\nTraining সম্পন্ন!")