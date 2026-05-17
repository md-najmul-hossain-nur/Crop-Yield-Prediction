"""
=============================================================
  DNN (MLP) TRAINING — Crop Yield Prediction
  Team: Light Seekers | Course: CSE-4889
  Input : Data/Marge/merged_dataset.csv
  Output: dnn_comparison.png
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from sklearn.neural_network import MLPRegressor, MLPClassifier
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

# Drop leaky columns
df.drop(columns=['Transplant', 'Growth', 'Harvest', 'AP Ratio'], inplace=True)

# Remove zero production
df = df[df['Production'] > 0].copy()

# IQR outlier removal
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

# Regression split
X_reg = df[REG_FEATURES]
y_reg = df['Production_log']
scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)
X_tr, X_te, y_tr, y_te = train_test_split(X_reg_scaled, y_reg.values, test_size=0.2, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.1, random_state=42)

# Classification split
X_cls = df[CLS_FEATURES]
y_cls = df['Crop_enc']
scaler_cls = StandardScaler()
X_cls_scaled = scaler_cls.fit_transform(X_cls)
X_trc, X_tec, y_trc, y_tec = train_test_split(X_cls_scaled, y_cls.values, test_size=0.2, random_state=42)
X_trc, X_valc, y_trc, y_valc = train_test_split(X_trc, y_trc, test_size=0.1, random_state=42)

print(f"\n  Regression     — Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_te)}")
print(f"  Classification — Train: {len(X_trc)}, Val: {len(X_valc)}, Test: {len(X_tec)}")

# ─────────────────────────────────────────────
# STEP 4: Train DNN Models
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Training DNN (MLP) Models")
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
dnn_reg.fit(X_tr, y_tr)
print(f"সম্পন্ন! Epochs: {dnn_reg.n_iter_}")

pred_r = dnn_reg.predict(X_te)
dnn_reg_res = dict(
    r2   = r2_score(y_te, pred_r),
    rmse = np.sqrt(mean_squared_error(y_te, pred_r)),
    mae  = mean_absolute_error(y_te, pred_r),
    loss_curve = dnn_reg.loss_curve_,
    val_scores = dnn_reg.validation_scores_,
    n_iter = dnn_reg.n_iter_
)
print(f"  ✅ R²={dnn_reg_res['r2']:.4f}  RMSE={dnn_reg_res['rmse']:.4f}  MAE={dnn_reg_res['mae']:.4f}")

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
dnn_cls.fit(X_trc, y_trc)
print(f"সম্পন্ন! Epochs: {dnn_cls.n_iter_}")

pred_c = dnn_cls.predict(X_tec)
dnn_cls_res = dict(
    acc  = accuracy_score(y_tec, pred_c),
    f1   = f1_score(y_tec, pred_c, average='weighted', zero_division=0),
    loss_curve = dnn_cls.loss_curve_,
    val_scores = dnn_cls.validation_scores_,
    n_iter = dnn_cls.n_iter_
)
print(f"  ✅ Accuracy={dnn_cls_res['acc']:.4f}  F1={dnn_cls_res['f1']:.4f}")

# ─────────────────────────────────────────────
# STEP 5: Comparison Chart
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Generating DNN Comparison Chart")
print("=" * 60)

# Reference results from RF/GB (from rf_gb_train.py — update if you re-run)
ref_reg = {
    'Random Forest':   dict(r2=0.8952, rmse=0.6085, mae=0.4247),
    'Grad. Boosting':  dict(r2=0.9110, rmse=0.5607, mae=0.4013),
}
ref_cls = {
    'Random Forest':   dict(acc=0.8579, f1=0.8572),
    'Grad. Boosting':  dict(acc=0.8525, f1=0.8538),
}

all_models_reg = {**ref_reg, 'DNN (MLP)': dict(r2=dnn_reg_res['r2'],
                                                 rmse=dnn_reg_res['rmse'],
                                                 mae=dnn_reg_res['mae'])}
all_models_cls = {**ref_cls, 'DNN (MLP)': dict(acc=dnn_cls_res['acc'],
                                                 f1=dnn_cls_res['f1'])}

m_names  = list(all_models_reg.keys())
bar_colors = ['#2ecc71', '#3498db', '#e74c3c']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(
    'RF vs Gradient Boosting vs DNN — Full Model Comparison\n'
    'Trained on merged_dataset.csv  |  Team: Light Seekers | CSE-4889',
    fontsize=14, fontweight='bold', y=0.98
)

def bar_labels(ax, bars):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.005,
                f'{h:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# ── A1: R² ──
ax = axes[0, 0]
vals = [all_models_reg[m]['r2'] for m in m_names]
bars = ax.bar(m_names, vals, color=bar_colors, width=0.5, edgecolor='white', linewidth=1.5)
ax.set_title('Task A — R² Score (↑ higher better)', fontweight='bold')
ax.set_ylim(0, 1.15); ax.set_ylabel('R²')
bar_labels(ax, bars)
ax.grid(axis='y', alpha=0.3)
bars[int(np.argmax(vals))].set_edgecolor('gold'); bars[int(np.argmax(vals))].set_linewidth(3)

# ── A2: RMSE ──
ax = axes[0, 1]
vals = [all_models_reg[m]['rmse'] for m in m_names]
bars = ax.bar(m_names, vals, color=bar_colors, width=0.5, edgecolor='white', linewidth=1.5)
ax.set_title('Task A — RMSE (↓ lower better)', fontweight='bold')
ax.set_ylabel('RMSE')
bar_labels(ax, bars)
ax.grid(axis='y', alpha=0.3)
bars[int(np.argmin(vals))].set_edgecolor('gold'); bars[int(np.argmin(vals))].set_linewidth(3)

# ── A3: DNN Regression Loss Curve ──
ax = axes[0, 2]
epochs_train = range(len(dnn_reg_res['loss_curve']))
ax.plot(epochs_train, dnn_reg_res['loss_curve'],
        color='#e74c3c', linewidth=2, label='Train Loss (MSE)')
val_x = np.linspace(0, len(dnn_reg_res['loss_curve']), len(dnn_reg_res['val_scores']))
ax.plot(val_x, [1 - s for s in dnn_reg_res['val_scores']],
        color='#3498db', linewidth=2, linestyle='--', label='Val Loss (1−R²)')
ax.set_title(f'DNN Regression — Loss Curve\n(Epochs: {dnn_reg_res["n_iter"]}, Test R²: {dnn_reg_res["r2"]:.4f})',
             fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend(); ax.grid(alpha=0.3)

# ── B1: Accuracy ──
ax = axes[1, 0]
vals = [all_models_cls[m]['acc'] for m in m_names]
bars = ax.bar(m_names, vals, color=bar_colors, width=0.5, edgecolor='white', linewidth=1.5)
ax.set_title('Task B — Accuracy (↑ higher better)', fontweight='bold')
ax.set_ylim(0, 1.15); ax.set_ylabel('Accuracy')
bar_labels(ax, bars)
ax.grid(axis='y', alpha=0.3)
bars[int(np.argmax(vals))].set_edgecolor('gold'); bars[int(np.argmax(vals))].set_linewidth(3)

# ── B2: F1 ──
ax = axes[1, 1]
vals = [all_models_cls[m]['f1'] for m in m_names]
bars = ax.bar(m_names, vals, color=bar_colors, width=0.5, edgecolor='white', linewidth=1.5)
ax.set_title('Task B — F1 Score Weighted (↑ higher better)', fontweight='bold')
ax.set_ylim(0, 1.15); ax.set_ylabel('F1 Score')
bar_labels(ax, bars)
ax.grid(axis='y', alpha=0.3)
bars[int(np.argmax(vals))].set_edgecolor('gold'); bars[int(np.argmax(vals))].set_linewidth(3)

# ── B3: DNN Classification Loss & Val Accuracy Curve ──
ax = axes[1, 2]
epochs_train = range(len(dnn_cls_res['loss_curve']))
ax.plot(epochs_train, dnn_cls_res['loss_curve'],
        color='#e74c3c', linewidth=2, label='Train Loss')
val_x = np.linspace(0, len(dnn_cls_res['loss_curve']), len(dnn_cls_res['val_scores']))
ax.plot(val_x, dnn_cls_res['val_scores'],
        color='#2ecc71', linewidth=2, linestyle='--', label='Val Accuracy')
ax.set_title(f'DNN Classification — Loss & Val Accuracy\n(Epochs: {dnn_cls_res["n_iter"]}, Test Acc: {dnn_cls_res["acc"]:.4f})',
             fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss / Accuracy')
ax.legend(); ax.grid(alpha=0.3)

# Legend
rf_p  = mpatches.Patch(color='#2ecc71', label='Random Forest')
gb_p  = mpatches.Patch(color='#3498db', label='Gradient Boosting')
dnn_p = mpatches.Patch(color='#e74c3c', label='DNN (MLP)')
gold_p = mpatches.Patch(edgecolor='gold', facecolor='none',
                         linewidth=2, label='Best (gold border)')
fig.legend(handles=[rf_p, gb_p, dnn_p, gold_p],
           loc='lower center', ncol=4, fontsize=11,
           bbox_to_anchor=(0.5, 0.01), framealpha=0.9)

plt.tight_layout(rect=[0, 0.05, 1, 0.97])
plt.savefig('dnn_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Saved: dnn_comparison.png")

# ─────────────────────────────────────────────
# STEP 6: Final Summary
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
print("DNN Training সম্পন্ন! ✅  →  dnn_comparison.png")
