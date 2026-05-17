"""
=============================================================
  DNN (MLP) TRAINING — Crop Yield Prediction
  Team: Light Seekers | Course: CSE-4889

  Input : Augmentation/Data/ (augmented + preprocessed CSVs)
          Augmentation/models/rf_gb_results.npy (RF/GB results)
  Output: Augmentation/aug_dnn_comparison.png
         Augmentation/models/dnn_reg.pkl
         Augmentation/models/dnn_cls.pkl

  Architecture:
    Regression     : Input(15) → 256 → 128 → 64 → 32 → Output(1)
    Classification : Input(9)  → 256 → 128 → 64 → Output(N_classes)

  ⚠️  aug_rf_gb_train.py আগে run করো!
      তাহলে Augmentation/models/rf_gb_results.npy তৈরি হবে।
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

from sklearn.neural_network import MLPRegressor, MLPClassifier
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
X_tr_r  = pd.read_csv(f"{DATA_DIR}/X_reg_train.csv").values
X_val_r = pd.read_csv(f"{DATA_DIR}/X_reg_val.csv").values
X_te_r  = pd.read_csv(f"{DATA_DIR}/X_reg_test.csv").values
y_tr_r  = pd.read_csv(f"{DATA_DIR}/y_reg_train.csv").squeeze().values
y_val_r = pd.read_csv(f"{DATA_DIR}/y_reg_val.csv").squeeze().values
y_te_r  = pd.read_csv(f"{DATA_DIR}/y_reg_test.csv").squeeze().values

# Classification
X_tr_c  = pd.read_csv(f"{DATA_DIR}/X_cls_train.csv").values
X_val_c = pd.read_csv(f"{DATA_DIR}/X_cls_val.csv").values
X_te_c  = pd.read_csv(f"{DATA_DIR}/X_cls_test.csv").values
y_tr_c  = pd.read_csv(f"{DATA_DIR}/y_cls_train.csv").squeeze().values
y_val_c = pd.read_csv(f"{DATA_DIR}/y_cls_val.csv").squeeze().values
y_te_c  = pd.read_csv(f"{DATA_DIR}/y_cls_test.csv").squeeze().values

REG_FEATURES = list(pd.read_csv(f"{DATA_DIR}/X_reg_train.csv").columns)
CLS_FEATURES = list(pd.read_csv(f"{DATA_DIR}/X_cls_train.csv").columns)

print(f"  Regression     — Train: {len(X_tr_r):,}, Val: {len(X_val_r):,}, Test: {len(X_te_r):,}")
print(f"  Classification — Train: {len(X_tr_c):,}, Val: {len(X_val_c):,}, Test: {len(X_te_c):,}")

# ─────────────────────────────────────────────
# STEP 2: Train DNN Models
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Training DNN (MLP) Models")
print("=" * 60)

# ── DNN Regressor ──
print("""
  Architecture (Regression):
  Input(15) → Dense(256,ReLU) → Dense(128,ReLU)
            → Dense(64,ReLU)  → Dense(32,ReLU)
            → Output(1, Linear)
  Optimizer: Adam | LR: 0.001 | Batch: 256 | Early Stop: patience=20
""")

dnn_reg = MLPRegressor(
    hidden_layer_sizes  = (256, 128, 64, 32),
    activation          = 'relu',
    solver              = 'adam',
    learning_rate_init  = 0.001,
    max_iter            = 300,
    early_stopping      = True,
    validation_fraction = 0.1,
    n_iter_no_change    = 20,
    batch_size          = 256,
    random_state        = 42,
    verbose             = False
)

print("Training শুরু হচ্ছে (Regression)...")
dnn_reg.fit(X_tr_r, y_tr_r)
print(f"সম্পন্ন! Epochs: {dnn_reg.n_iter_}")

pred_r = dnn_reg.predict(X_te_r)
dnn_reg_res = dict(
    r2         = r2_score(y_te_r, pred_r),
    rmse       = np.sqrt(mean_squared_error(y_te_r, pred_r)),
    mae        = mean_absolute_error(y_te_r, pred_r),
    loss_curve = dnn_reg.loss_curve_,
    val_scores = dnn_reg.validation_scores_,
    n_iter     = dnn_reg.n_iter_
)
joblib.dump(dnn_reg, f"{MODEL_DIR}/dnn_reg.pkl")
print(f"  ✅ R²={dnn_reg_res['r2']:.4f}  RMSE={dnn_reg_res['rmse']:.4f}  MAE={dnn_reg_res['mae']:.4f}")
print(f"  💾 Saved: {MODEL_DIR}/dnn_reg.pkl")

# ── DNN Classifier ──
print("""
  Architecture (Classification):
  Input(9) → Dense(256,ReLU) → Dense(128,ReLU)
           → Dense(64,ReLU)  → Output(N_classes, Softmax)
  Optimizer: Adam | LR: 0.001 | Batch: 256 | Early Stop: patience=20
""")

dnn_cls = MLPClassifier(
    hidden_layer_sizes  = (256, 128, 64),
    activation          = 'relu',
    solver              = 'adam',
    learning_rate_init  = 0.001,
    max_iter            = 300,
    early_stopping      = True,
    validation_fraction = 0.1,
    n_iter_no_change    = 20,
    batch_size          = 256,
    random_state        = 42,
    verbose             = False
)

print("Training শুরু হচ্ছে (Classification)...")
dnn_cls.fit(X_tr_c, y_tr_c)
print(f"সম্পন্ন! Epochs: {dnn_cls.n_iter_}")

pred_c = dnn_cls.predict(X_te_c)
dnn_cls_res = dict(
    acc        = accuracy_score(y_te_c, pred_c),
    f1         = f1_score(y_te_c, pred_c, average='weighted', zero_division=0),
    loss_curve = dnn_cls.loss_curve_,
    val_scores = dnn_cls.validation_scores_,
    n_iter     = dnn_cls.n_iter_
)
joblib.dump(dnn_cls, f"{MODEL_DIR}/dnn_cls.pkl")
print(f"  ✅ Accuracy={dnn_cls_res['acc']:.4f}  F1={dnn_cls_res['f1']:.4f}")
print(f"  💾 Saved: {MODEL_DIR}/dnn_cls.pkl")

# ─────────────────────────────────────────────
# STEP 3: Load RF/GB Results for Comparison
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Loading RF/GB Results for Comparison")
print("=" * 60)

npy_path = f"{MODEL_DIR}/rf_gb_results.npy"

if os.path.exists(npy_path):
    saved   = np.load(npy_path, allow_pickle=True).item()
    ref_reg = saved['reg']
    ref_cls = saved['cls']
    print(f"  ✅ Loaded from {npy_path}")
    for name, res in ref_reg.items():
        print(f"     {name} Reg  → R²={res['r2']:.4f}, RMSE={res['rmse']:.4f}, MAE={res['mae']:.4f}")
    for name, res in ref_cls.items():
        print(f"     {name} Cls  → Acc={res['acc']:.4f}, F1={res['f1']:.4f}")
else:
    # Fallback: run RF/GB inline
    print("  ⚠️  rf_gb_results.npy not found — running RF/GB inline...")
    rf_reg_f = RandomForestRegressor(n_estimators=150, max_depth=20,
                                     min_samples_leaf=2, n_jobs=-1, random_state=42)
    rf_reg_f.fit(X_tr_r, y_tr_r)
    p = rf_reg_f.predict(X_te_r)
    rf_r2, rf_rmse, rf_mae = (r2_score(y_te_r, p),
                               np.sqrt(mean_squared_error(y_te_r, p)),
                               mean_absolute_error(y_te_r, p))

    gb_reg_f = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05,
                                              max_depth=6, random_state=42)
    gb_reg_f.fit(X_tr_r, y_tr_r)
    p = gb_reg_f.predict(X_te_r)
    gb_r2, gb_rmse, gb_mae = (r2_score(y_te_r, p),
                               np.sqrt(mean_squared_error(y_te_r, p)),
                               mean_absolute_error(y_te_r, p))

    rf_cls_f = RandomForestClassifier(n_estimators=150, max_depth=20,
                                      min_samples_leaf=2, class_weight='balanced',
                                      n_jobs=-1, random_state=42)
    rf_cls_f.fit(X_tr_c, y_tr_c)
    p = rf_cls_f.predict(X_te_c)
    rf_acc = accuracy_score(y_te_c, p)
    rf_f1  = f1_score(y_te_c, p, average='weighted', zero_division=0)

    gb_cls_f = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05,
                                               max_depth=6, random_state=42)
    gb_cls_f.fit(X_tr_c, y_tr_c)
    p = gb_cls_f.predict(X_te_c)
    gb_acc = accuracy_score(y_te_c, p)
    gb_f1  = f1_score(y_te_c, p, average='weighted', zero_division=0)

    ref_reg = {
        'Random Forest':   dict(r2=rf_r2,   rmse=rf_rmse, mae=rf_mae),
        'Grad. Boosting':  dict(r2=gb_r2,   rmse=gb_rmse, mae=gb_mae),
    }
    ref_cls = {
        'Random Forest':   dict(acc=rf_acc, f1=rf_f1),
        'Grad. Boosting':  dict(acc=gb_acc, f1=gb_f1),
    }
    print(f"  ✅ RF  Reg  → R²={rf_r2:.4f}, RMSE={rf_rmse:.4f}, MAE={rf_mae:.4f}")
    print(f"  ✅ GB  Reg  → R²={gb_r2:.4f}, RMSE={gb_rmse:.4f}, MAE={gb_mae:.4f}")
    print(f"  ✅ RF  Cls  → Acc={rf_acc:.4f}, F1={rf_f1:.4f}")
    print(f"  ✅ GB  Cls  → Acc={gb_acc:.4f}, F1={gb_f1:.4f}")

# ─────────────────────────────────────────────
# STEP 4: Comparison Chart (RF vs GB vs DNN)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Generating Full Comparison Chart (RF vs GB vs DNN)")
print("=" * 60)

all_models_reg = {
    **ref_reg,
    'DNN (MLP)': dict(r2=dnn_reg_res['r2'],
                      rmse=dnn_reg_res['rmse'],
                      mae=dnn_reg_res['mae'])
}
all_models_cls = {
    **ref_cls,
    'DNN (MLP)': dict(acc=dnn_cls_res['acc'],
                      f1=dnn_cls_res['f1'])
}

m_names    = list(all_models_reg.keys())
bar_colors = ['#2ecc71', '#3498db', '#e74c3c']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(
    'RF vs Gradient Boosting vs DNN — Full Model Comparison\n'
    'Input: Augmented Data  |  Team: Light Seekers | CSE-4889',
    fontsize=14, fontweight='bold', y=0.98
)

def bar_labels(ax, bars):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.005,
                f'{h:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

def highlight_best(bars, vals, mode='max'):
    idx = int(np.argmax(vals)) if mode == 'max' else int(np.argmin(vals))
    bars[idx].set_edgecolor('gold')
    bars[idx].set_linewidth(3)

# ── A1: R² ──
ax = axes[0, 0]
vals = [all_models_reg[m]['r2'] for m in m_names]
bars = ax.bar(m_names, vals, color=bar_colors, width=0.5, edgecolor='white', linewidth=1.5)
ax.set_title('Task A — R² Score (↑ higher better)', fontweight='bold')
ax.set_ylim(0, 1.15); ax.set_ylabel('R²')
bar_labels(ax, bars); highlight_best(bars, vals, 'max')
ax.grid(axis='y', alpha=0.3)

# ── A2: RMSE ──
ax = axes[0, 1]
vals = [all_models_reg[m]['rmse'] for m in m_names]
bars = ax.bar(m_names, vals, color=bar_colors, width=0.5, edgecolor='white', linewidth=1.5)
ax.set_title('Task A — RMSE (↓ lower better)', fontweight='bold')
ax.set_ylabel('RMSE')
bar_labels(ax, bars); highlight_best(bars, vals, 'min')
ax.grid(axis='y', alpha=0.3)

# ── A3: DNN Regression Loss Curve ──
ax = axes[0, 2]
ax.plot(range(len(dnn_reg_res['loss_curve'])),
        dnn_reg_res['loss_curve'],
        color='#e74c3c', linewidth=2, label='Train Loss (MSE)')
val_x = np.linspace(0, len(dnn_reg_res['loss_curve']),
                    len(dnn_reg_res['val_scores']))
ax.plot(val_x, [1 - s for s in dnn_reg_res['val_scores']],
        color='#3498db', linewidth=2, linestyle='--', label='Val Loss (1−R²)')
ax.set_title(f"DNN Regression — Loss Curve\n"
             f"(Epochs: {dnn_reg_res['n_iter']}, Test R²: {dnn_reg_res['r2']:.4f})",
             fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend(); ax.grid(alpha=0.3)

# ── B1: Accuracy ──
ax = axes[1, 0]
vals = [all_models_cls[m]['acc'] for m in m_names]
bars = ax.bar(m_names, vals, color=bar_colors, width=0.5, edgecolor='white', linewidth=1.5)
ax.set_title('Task B — Accuracy (↑ higher better)', fontweight='bold')
ax.set_ylim(0, 1.15); ax.set_ylabel('Accuracy')
bar_labels(ax, bars); highlight_best(bars, vals, 'max')
ax.grid(axis='y', alpha=0.3)

# ── B2: F1 ──
ax = axes[1, 1]
vals = [all_models_cls[m]['f1'] for m in m_names]
bars = ax.bar(m_names, vals, color=bar_colors, width=0.5, edgecolor='white', linewidth=1.5)
ax.set_title('Task B — F1 Score Weighted (↑ higher better)', fontweight='bold')
ax.set_ylim(0, 1.15); ax.set_ylabel('F1 Score')
bar_labels(ax, bars); highlight_best(bars, vals, 'max')
ax.grid(axis='y', alpha=0.3)

# ── B3: DNN Classification Loss & Val Accuracy ──
ax = axes[1, 2]
ax.plot(range(len(dnn_cls_res['loss_curve'])),
        dnn_cls_res['loss_curve'],
        color='#e74c3c', linewidth=2, label='Train Loss')
val_x = np.linspace(0, len(dnn_cls_res['loss_curve']),
                    len(dnn_cls_res['val_scores']))
ax.plot(val_x, dnn_cls_res['val_scores'],
        color='#2ecc71', linewidth=2, linestyle='--', label='Val Accuracy')
ax.set_title(f"DNN Classification — Loss & Val Accuracy\n"
             f"(Epochs: {dnn_cls_res['n_iter']}, Test Acc: {dnn_cls_res['acc']:.4f})",
             fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss / Accuracy')
ax.legend(); ax.grid(alpha=0.3)

# Legend
rf_p   = mpatches.Patch(color='#2ecc71', label='Random Forest')
gb_p   = mpatches.Patch(color='#3498db', label='Gradient Boosting')
dnn_p  = mpatches.Patch(color='#e74c3c', label='DNN (MLP)')
gold_p = mpatches.Patch(edgecolor='gold', facecolor='none',
                         linewidth=2, label='Best (gold border)')
fig.legend(handles=[rf_p, gb_p, dnn_p, gold_p],
           loc='lower center', ncol=4, fontsize=11,
           bbox_to_anchor=(0.5, 0.01), framealpha=0.9)

plt.tight_layout(rect=[0, 0.05, 1, 0.97])
plt.savefig('Augmentation/aug_dnn_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Saved: Augmentation/aug_dnn_comparison.png")

# ─────────────────────────────────────────────
# STEP 5: Final Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL RESULT SUMMARY — ALL 3 MODELS")
print("=" * 60)

best_reg_m = max(all_models_reg, key=lambda m: all_models_reg[m]['r2'])
best_cls_m = max(all_models_cls, key=lambda m: all_models_cls[m]['f1'])

print(f"\n  Task A — Regression")
print(f"  {'Model':<20} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
print("  " + "-" * 48)
for name, res in all_models_reg.items():
    tag = " ✅ BEST" if name == best_reg_m else ""
    print(f"  {name:<20} {res['r2']:>8.4f} {res['rmse']:>8.4f} {res['mae']:>8.4f}{tag}")

print(f"\n  Task B — Classification")
print(f"  {'Model':<20} {'Accuracy':>10} {'F1':>8}")
print("  " + "-" * 42)
for name, res in all_models_cls.items():
    tag = " ✅ BEST" if name == best_cls_m else ""
    print(f"  {name:<20} {res['acc']:>10.4f} {res['f1']:>8.4f}{tag}")

print("\n" + "=" * 60)
print(f"  Best Regression    : {best_reg_m}")
print(f"  Best Classification: {best_cls_m}")
print("=" * 60)
print("DNN Training সম্পন্ন! ✅")
print("→ Chart  : Augmentation/aug_dnn_comparison.png")
print("→ Models : Augmentation/models/ (dnn_reg, dnn_cls).pkl")
