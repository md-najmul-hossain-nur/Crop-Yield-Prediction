"""
=============================================================
  DNN MODEL TRAINING — Crop Yield Prediction
  Team: Light Seekers | Course: CSE-4889

  Model: MLP (Multi-Layer Perceptron) — Deep Neural Network
         sklearn implementation

  Architecture:
    Task A (Regression)     : Input → 256 → 128 → 64 → 32 → Output(1)
    Task B (Classification) : Input → 256 → 128 → 64 → Output(72)

  Task A: Yield Prediction     → Regression
  Task B: Crop Recommendation  → Classification
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
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

# ✅ FIXED: val_augmented নেই, train থেকেই বানাচ্ছি
df_full = pd.read_csv('train_augmented.csv')
test    = pd.read_csv('test_original.csv')
train, val = train_test_split(df_full, test_size=0.10, random_state=42)

print(f"Train : {train.shape[0]:,} rows")
print(f"Val   : {val.shape[0]:,} rows")
print(f"Test  : {test.shape[0]:,} rows  (original only)")

# ─────────────────────────────────────────────
# Features
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

# ─────────────────────────────────────────────
# STEP 1: Scaling
# DNN-এর জন্য scaling অবশ্যই দরকার
# StandardScaler: mean=0, std=1
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 1: Feature Scaling (StandardScaler)")
print("=" * 65)

# Regression scaling
scaler_r = StandardScaler()
X_train_r = scaler_r.fit_transform(train[REG_FEATURES])
X_val_r   = scaler_r.transform(val[REG_FEATURES])
X_test_r  = scaler_r.transform(test[REG_FEATURES])

y_train_r = train['Production_log'].values
y_val_r   = val['Production_log'].values
y_test_r  = test['Production_log'].values

# Classification scaling
scaler_c = StandardScaler()
X_train_c = scaler_c.fit_transform(train[CLS_FEATURES])
X_val_c   = scaler_c.transform(val[CLS_FEATURES])
X_test_c  = scaler_c.transform(test[CLS_FEATURES])

y_train_c = train['Crop_enc'].values
y_val_c   = val['Crop_enc'].values
y_test_c  = test['Crop_enc'].values

print(f"Regression  — X_train: {X_train_r.shape}  X_test: {X_test_r.shape}")
print(f"Classification — X_train: {X_train_c.shape}  X_test: {X_test_c.shape}")
print("Scaling সম্পন্ন ✅")


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────
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


# ══════════════════════════════════════════════
#   TASK A — DNN REGRESSION
# ══════════════════════════════════════════════
print("\n" + "=" * 65)
print("TASK A: DNN REGRESSION — Yield Prediction")
print("=" * 65)

print("""
  Architecture:
  Input(15) → Dense(256,ReLU) → Dense(128,ReLU)
            → Dense(64,ReLU)  → Dense(32,ReLU)
            → Output(1, Linear)

  Optimizer : Adam  (lr=0.001)
  Loss      : MSE
  Batch size: 256
  Early stop: patience=20
""")

dnn_reg = MLPRegressor(
    hidden_layer_sizes = (256, 128, 64, 32),  # 4টা hidden layer
    activation         = 'relu',              # ReLU activation
    solver             = 'adam',              # Adam optimizer
    learning_rate_init = 0.001,               # learning rate
    max_iter           = 300,                 # সর্বোচ্চ epoch
    early_stopping     = True,                # overfitting রোধ
    validation_fraction= 0.1,                 # 10% internal validation
    n_iter_no_change   = 20,                  # 20 epoch improvement না হলে stop
    batch_size         = 256,                 # mini-batch size
    random_state       = 42,
    verbose            = False
)

print("Training শুরু হচ্ছে...")
dnn_reg.fit(X_train_r, y_train_r)
print(f"Training সম্পন্ন! Epochs ran: {dnn_reg.n_iter_}")

print("\nResults:")
v_r2, v_rmse, v_mae = reg_report("Validation", y_val_r, dnn_reg.predict(X_val_r))
t_r2, t_rmse, t_mae = reg_report("Test",       y_test_r, dnn_reg.predict(X_test_r))

dnn_reg_results = dict(
    val_r2=v_r2, test_r2=t_r2,
    test_rmse=t_rmse, test_mae=t_mae,
    loss_curve=dnn_reg.loss_curve_,
    val_loss_curve=dnn_reg.validation_scores_,
    n_iter=dnn_reg.n_iter_
)


# ══════════════════════════════════════════════
#   TASK B — DNN CLASSIFICATION
# ══════════════════════════════════════════════
print("\n" + "=" * 65)
print("TASK B: DNN CLASSIFICATION — Crop Recommendation")
print("=" * 65)

print("""
  Architecture:
  Input(9) → Dense(256,ReLU) → Dense(128,ReLU)
           → Dense(64,ReLU)  → Output(72, Softmax)

  Optimizer : Adam  (lr=0.001)
  Loss      : Cross-Entropy
  Batch size: 256
  Early stop: patience=20
  Classes   : 72 crops
""")

dnn_cls = MLPClassifier(
    hidden_layer_sizes = (256, 128, 64),  # 3টা hidden layer
    activation         = 'relu',
    solver             = 'adam',
    learning_rate_init = 0.001,
    max_iter           = 300,
    early_stopping     = True,
    validation_fraction= 0.1,
    n_iter_no_change   = 20,
    batch_size         = 256,
    random_state       = 42,
    verbose            = False
)

print("Training শুরু হচ্ছে...")
dnn_cls.fit(X_train_c, y_train_c)
print(f"Training সম্পন্ন! Epochs ran: {dnn_cls.n_iter_}")

print("\nResults:")
v_acc, v_f1 = cls_report("Validation", y_val_c, dnn_cls.predict(X_val_c))
t_acc, t_f1 = cls_report("Test",       y_test_c, dnn_cls.predict(X_test_c))

dnn_cls_results = dict(
    val_acc=v_acc, test_acc=t_acc, test_f1=t_f1,
    loss_curve=dnn_cls.loss_curve_,
    val_loss_curve=dnn_cls.validation_scores_,
    n_iter=dnn_cls.n_iter_
)


# ══════════════════════════════════════════════
#   CHARTS
# ══════════════════════════════════════════════
print("\n" + "=" * 65)
print("Charts তৈরি হচ্ছে...")
print("=" * 65)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('DNN Model Training — Crop Yield Prediction\n'
             'Team: Light Seekers | CSE-4889',
             fontsize=14, fontweight='bold')

# ── Chart 1: Regression Loss Curve ──
ax = axes[0, 0]
ax.plot(dnn_reg_results['loss_curve'], color='#e74c3c', linewidth=2, label='Train Loss')
ax.plot(
    np.linspace(0, len(dnn_reg_results['loss_curve']),
                len(dnn_reg_results['val_loss_curve'])),
    [1 - s for s in dnn_reg_results['val_loss_curve']],
    color='#3498db', linewidth=2, linestyle='--', label='Val Loss (1-R²)'
)
ax.set_title('Task A — DNN Regression\nTraining Loss Curve', fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (MSE)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.text(0.95, 0.95,
        f'Final Test R²: {dnn_reg_results["test_r2"]:.4f}\nEpochs: {dnn_reg_results["n_iter"]}',
        transform=ax.transAxes, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ── Chart 2: Actual vs Predicted (Regression) ──
ax = axes[0, 1]
preds_r = dnn_reg.predict(X_test_r)
ax.scatter(y_test_r, preds_r, alpha=0.5, color='#9b59b6', s=20)
mn = min(float(y_test_r.min()), float(preds_r.min()))
mx = max(float(y_test_r.max()), float(preds_r.max()))
ax.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Ideal line')
ax.set_title('Task A — DNN\nActual vs Predicted', fontweight='bold')
ax.set_xlabel('Actual (log scale)')
ax.set_ylabel('Predicted (log scale)')
ax.legend()
ax.grid(True, alpha=0.3)

# ── Chart 3: Classification Loss Curve ──
ax = axes[1, 0]
ax.plot(dnn_cls_results['loss_curve'], color='#e74c3c', linewidth=2, label='Train Loss')
ax.plot(
    np.linspace(0, len(dnn_cls_results['loss_curve']),
                len(dnn_cls_results['val_loss_curve'])),
    dnn_cls_results['val_loss_curve'],
    color='#2ecc71', linewidth=2, linestyle='--', label='Val Accuracy'
)
ax.set_title('Task B — DNN Classification\nTraining Loss & Val Accuracy', fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss / Accuracy')
ax.legend()
ax.grid(True, alpha=0.3)
ax.text(0.95, 0.95,
        f'Test Acc: {dnn_cls_results["test_acc"]:.4f}\nEpochs: {dnn_cls_results["n_iter"]}',
        transform=ax.transAxes, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# ── Chart 4: Final Comparison Bar Chart ──
ax = axes[1, 1]
metrics = ['R² (Reg)', 'Accuracy\n(Cls)', 'F1\n(Cls)']
values  = [
    dnn_reg_results['test_r2'],
    dnn_cls_results['test_acc'],
    dnn_cls_results['test_f1']
]
colors  = ['#3498db', '#2ecc71', '#e67e22']
bars = ax.bar(metrics, values, color=colors, width=0.4, edgecolor='white')
ax.set_title('DNN — Final Test Metrics', fontweight='bold')
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score')
for b, v in zip(bars, values):
    ax.text(b.get_x()+b.get_width()/2, v+0.01,
            f'{v:.4f}', ha='center', fontweight='bold', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('dnn_results.png', dpi=150, bbox_inches='tight')
print("Saved: dnn_results.png ✅")


# ══════════════════════════════════════════════
#   FINAL SUMMARY TABLE (All 3 Models)
# ══════════════════════════════════════════════
print("\n" + "=" * 65)
print("DNN FINAL RESULT SUMMARY")
print("=" * 65)

print(f"\n  Task A — Regression (Test Set)")
print(f"  {'Model':<22} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
print("  " + "-"*50)

prev_reg = {
    'Random Forest':  dict(r2=0.8952, rmse=0.6085, mae=0.4247),
    'Grad. Boosting': dict(r2=0.9110, rmse=0.5607, mae=0.4013),
}
for name, res in prev_reg.items():
    print(f"  {name:<22} {res['r2']:>8.4f} {res['rmse']:>8.4f} {res['mae']:>8.4f}")

tag = " ← BEST" if dnn_reg_results['test_r2'] > 0.9110 else ""
print(f"  {'DNN (MLP)':<22} {dnn_reg_results['test_r2']:>8.4f} "
      f"{dnn_reg_results['test_rmse']:>8.4f} {dnn_reg_results['test_mae']:>8.4f}{tag}")

print(f"\n  Task B — Classification (Test Set)")
print(f"  {'Model':<22} {'Accuracy':>10} {'F1':>8}")
print("  " + "-"*44)

prev_cls = {
    'Random Forest':  dict(acc=0.8579, f1=0.8572),
    'Grad. Boosting': dict(acc=0.8525, f1=0.8538),
}
for name, res in prev_cls.items():
    print(f"  {name:<22} {res['acc']:>10.4f} {res['f1']:>8.4f}")

tag = " ← BEST" if dnn_cls_results['test_f1'] > 0.8572 else ""
print(f"  {'DNN (MLP)':<22} {dnn_cls_results['test_acc']:>10.4f} "
      f"{dnn_cls_results['test_f1']:>8.4f}{tag}")

print("\n" + "=" * 65)
print("DNN Training সম্পন্ন! ✅")
print("Output: dnn_results.png")
print("=" * 65)