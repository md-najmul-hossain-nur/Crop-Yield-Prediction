"""
=============================================================
  TRAINING ON AUGMENTED DATASET — Match22 Data
  Models: Random Forest (RF), Gradient Boosting (GB), DNN
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

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score

PROCESSED_DIR = 'Data/Match22-data/Processed_Data_Augmented'
MODEL_DIR = 'Data/Match22-data/Models_Augmented'
OUTPUT_DIR = 'Data/Match22-data/Output'

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1: Load Preprocessed Augmented Datasets
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Preprocessed Augmented Datasets (12,960 Rows)")
print("=" * 60)

X_tr_r = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_reg_train.csv'))
X_te_r = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_reg_test.csv'))
y_tr_r = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_reg_train.csv')).values.ravel()
y_te_r = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_reg_test.csv')).values.ravel()

X_tr_c = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_cls_train.csv'))
X_te_c = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_cls_test.csv'))
y_tr_c = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_cls_train.csv')).values.ravel()
y_te_c = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_cls_test.csv')).values.ravel()

print(f"  Regression Train      : {X_tr_r.shape[0]} rows, Test: {X_te_r.shape[0]} rows")
print(f"  Classification Train : {X_tr_c.shape[0]} rows, Test: {X_te_c.shape[0]} rows")

res_reg = {}
res_cls = {}

# ─────────────────────────────────────────────
# STEP 2: Train & Evaluate Regression Models
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Training & Evaluating Regression Models (Augmented Yield Prediction)")
print("=" * 60)

# 1. RF Regressor
print("  [1/3] Training Random Forest (RF) Regressor...")
rf_reg = RandomForestRegressor(n_estimators=200, max_depth=22, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_reg.fit(X_tr_r, y_tr_r)
pred_rf = rf_reg.predict(X_te_r)
res_reg['RF'] = dict(r2=r2_score(y_te_r, pred_rf), rmse=np.sqrt(mean_squared_error(y_te_r, pred_rf)), mae=mean_absolute_error(y_te_r, pred_rf))
joblib.dump(rf_reg, os.path.join(MODEL_DIR, 'rf_regressor.pkl'))
print(f"        RF Regressor  -> R2: {res_reg['RF']['r2']:.4f} | RMSE: {res_reg['RF']['rmse']:.4f} | MAE: {res_reg['RF']['mae']:.4f}")

# 2. GB Regressor
print("  [2/3] Training Gradient Boosting (GB) Regressor...")
gb_reg = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_depth=8, random_state=42)
gb_reg.fit(X_tr_r, y_tr_r)
pred_gb = gb_reg.predict(X_te_r)
res_reg['GB'] = dict(r2=r2_score(y_te_r, pred_gb), rmse=np.sqrt(mean_squared_error(y_te_r, pred_gb)), mae=mean_absolute_error(y_te_r, pred_gb))
joblib.dump(gb_reg, os.path.join(MODEL_DIR, 'gb_regressor.pkl'))
print(f"        GB Regressor  -> R2: {res_reg['GB']['r2']:.4f} | RMSE: {res_reg['GB']['rmse']:.4f} | MAE: {res_reg['GB']['mae']:.4f}")

# 3. DNN Regressor
print("  [3/3] Training Deep Neural Network (DNN) Regressor on Augmented Data...")
dnn_reg = MLPRegressor(
    hidden_layer_sizes=(512, 256, 128, 64),
    activation='relu', solver='adam', learning_rate_init=0.001,
    max_iter=600, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=25, batch_size=256, random_state=42
)
dnn_reg.fit(X_tr_r, y_tr_r)
pred_dnn = dnn_reg.predict(X_te_r)
res_reg['DNN'] = dict(r2=r2_score(y_te_r, pred_dnn), rmse=np.sqrt(mean_squared_error(y_te_r, pred_dnn)), mae=mean_absolute_error(y_te_r, pred_dnn))
joblib.dump(dnn_reg, os.path.join(MODEL_DIR, 'dnn_regressor.pkl'))
print(f"        DNN Regressor -> R2: {res_reg['DNN']['r2']:.4f} | RMSE: {res_reg['DNN']['rmse']:.4f} | MAE: {res_reg['DNN']['mae']:.4f} (Epochs: {dnn_reg.n_iter_})")

# ─────────────────────────────────────────────
# STEP 3: Train & Evaluate Classification Models
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Training & Evaluating Classification Models (Augmented Crop Recommendation)")
print("=" * 60)

# 1. RF Classifier
print("  [1/3] Training Random Forest (RF) Classifier...")
rf_cls = RandomForestClassifier(n_estimators=200, max_depth=22, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1)
rf_cls.fit(X_tr_c, y_tr_c)
pred_rf_c = rf_cls.predict(X_te_c)
res_cls['RF'] = dict(acc=accuracy_score(y_te_c, pred_rf_c), f1=f1_score(y_te_c, pred_rf_c, average='weighted', zero_division=0))
joblib.dump(rf_cls, os.path.join(MODEL_DIR, 'rf_classifier.pkl'))
print(f"        RF Classifier  -> Accuracy: {res_cls['RF']['acc']:.4f} | F1 Score: {res_cls['RF']['f1']:.4f}")

# 2. GB Classifier
print("  [2/3] Training Gradient Boosting (GB) Classifier...")
gb_cls = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_depth=8, random_state=42)
gb_cls.fit(X_tr_c, y_tr_c)
pred_gb_c = gb_cls.predict(X_te_c)
res_cls['GB'] = dict(acc=accuracy_score(y_te_c, pred_gb_c), f1=f1_score(y_te_c, pred_gb_c, average='weighted', zero_division=0))
joblib.dump(gb_cls, os.path.join(MODEL_DIR, 'gb_classifier.pkl'))
print(f"        GB Classifier  -> Accuracy: {res_cls['GB']['acc']:.4f} | F1 Score: {res_cls['GB']['f1']:.4f}")

# 3. DNN Classifier
print("  [3/3] Training Deep Neural Network (DNN) Classifier on Augmented Data...")
dnn_cls = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation='relu', solver='adam', learning_rate_init=0.001,
    max_iter=600, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=25, batch_size=256, random_state=42
)
dnn_cls.fit(X_tr_c, y_tr_c)
pred_dnn_c = dnn_cls.predict(X_te_c)
res_cls['DNN'] = dict(acc=accuracy_score(y_te_c, pred_dnn_c), f1=f1_score(y_te_c, pred_dnn_c, average='weighted', zero_division=0))
joblib.dump(dnn_cls, os.path.join(MODEL_DIR, 'dnn_classifier.pkl'))
print(f"        DNN Classifier -> Accuracy: {res_cls['DNN']['acc']:.4f} | F1 Score: {res_cls['DNN']['f1']:.4f} (Epochs: {dnn_cls.n_iter_})")

# ─────────────────────────────────────────────
# STEP 4: Generate Comparison Visualization
# ─────────────────────────────────────────────
models = ['RF', 'GB', 'DNN']
colors = {'RF': '#2ecc71', 'GB': '#3498db', 'DNN': '#e74c3c'}
x = np.arange(len(models))

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(
    'Match22 AUGMENTED Dataset Performance (19,200 Rows) — RF vs GB vs DNN\n'
    'Team: Light Seekers | CSE-4889 | UIU Bangladesh',
    fontsize=13, fontweight='bold', y=0.98
)

def bar_metric(ax, vals, title, ylabel, best='max'):
    bar_list = ax.bar(x, vals, color=[colors[m] for m in models], edgecolor='white', linewidth=1.5, width=0.55)
    for b, v in zip(bar_list, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    best_idx = vals.index(max(vals)) if best == 'max' else vals.index(min(vals))
    bar_list[best_idx].set_edgecolor('gold'); bar_list[best_idx].set_linewidth(3)
    ax.set_title(title, fontweight='bold', fontsize=10); ax.set_ylabel(ylabel)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11); ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(vals) * 1.25)

bar_metric(axes[0, 0], [res_reg[m]['r2'] for m in models], 'Regression — R² (higher = better)', 'R²', 'max')
bar_metric(axes[0, 1], [res_reg[m]['rmse'] for m in models], 'Regression — RMSE (lower = better)', 'RMSE', 'min')
bar_metric(axes[0, 2], [res_reg[m]['mae'] for m in models], 'Regression — MAE (lower = better)', 'MAE', 'min')

bar_metric(axes[1, 0], [res_cls[m]['acc'] for m in models], 'Classification — Accuracy (higher)', 'Accuracy', 'max')
bar_metric(axes[1, 1], [res_cls[m]['f1'] for m in models], 'Classification — F1 Score (higher)', 'F1 Score', 'max')

ax_tbl = axes[1, 2]; ax_tbl.axis('off')
tdata = [
    [m, f"{res_reg[m]['r2']:.4f}", f"{res_reg[m]['rmse']:.4f}", f"{res_reg[m]['mae']:.4f}", f"{res_cls[m]['acc']:.4f}", f"{res_cls[m]['f1']:.4f}"]
    for m in models
]
tbl = ax_tbl.table(cellText=tdata, colLabels=['Model', 'R²', 'RMSE', 'MAE', 'Accuracy', 'F1 Score'], loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.2, 1.8)
for j in range(6):
    tbl[0, j].set_facecolor('#2c3e50')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
row_colors = ['#eafaf1', '#eaf4fb', '#fef9e7']
for ri, rc in enumerate(row_colors):
    for j in range(6): tbl[ri + 1, j].set_facecolor(rc)

plt.tight_layout(rect=[0, 0.02, 1, 0.95])
chart_path = os.path.join(OUTPUT_DIR, 'match22_augmented_comparison.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()

# Save text report
report_path = os.path.join(OUTPUT_DIR, 'RESULTS_MATCH22_AUGMENTED.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("================================================================\n")
    f.write("  MATCH22 AUGMENTED DATASET RESULTS & EVALUATION SUMMARY\n")
    f.write("  Dataset Expansion: 1,600 -> 19,200 rows (12.0x Growth)\n")
    f.write("  Models: Random Forest (RF), Gradient Boosting (GB), DNN\n")
    f.write("  Team: Light Seekers | CSE-4889 | UIU Bangladesh\n")
    f.write("================================================================\n\n")
    f.write(f"  Total Cleaned Rows : {X_tr_r.shape[0] + X_te_r.shape[0]}\n")
    f.write(f"  Train Set Rows     : {X_tr_r.shape[0]}\n")
    f.write(f"  Test Set Rows      : {X_te_r.shape[0]}\n\n")
    f.write("----------------------------------------------------------------\n")
    f.write(" 1. REGRESSION (YIELD PREDICTION - Production_log)\n")
    f.write("----------------------------------------------------------------\n")
    for m in models:
        f.write(f"  [{m}] R2 Score: {res_reg[m]['r2']:.4f} | RMSE: {res_reg[m]['rmse']:.4f} | MAE: {res_reg[m]['mae']:.4f}\n")
    f.write("\n----------------------------------------------------------------\n")
    f.write(" 2. CLASSIFICATION (CROP RECOMMENDATION - Crop_22_enc)\n")
    f.write("----------------------------------------------------------------\n")
    for m in models:
        f.write(f"  [{m}] Accuracy: {res_cls[m]['acc']:.4f} | F1 Score: {res_cls[m]['f1']:.4f}\n")
    f.write("\n================================================================\n")

print(f"  Saved comparison chart to {chart_path}")
print(f"  Saved evaluation report to {report_path}")

print("\n" + "=" * 60)
print("AUGMENTED MODEL TRAINING COMPLETE ✅")
print("=" * 60)
