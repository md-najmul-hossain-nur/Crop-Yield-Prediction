"""
=============================================================
  MASTER COMPARISON — সব Pipeline একসাথে
  Team: Light Seekers | Course: CSE-4889

  Pipeline 1: Merged Raw Data       (Marge_Data_train/)
  Pipeline 2: Preprocessed Data     (Preprocessing/Data/)
  Pipeline 3: Augmented Data        (Augmentation/Data/)

  প্রতিটা Pipeline-এ: RF, GB, DNN train হবে
  শেষে একটা বড় comparison chart

  Run from: Crop-Yield-Prediction/
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score
)

os.makedirs("Output", exist_ok=True)

# ═══════════════════════════════════════════════════════════
# HELPER: Train all 3 models on a given dataset
# ═══════════════════════════════════════════════════════════
def train_all_models(X_tr_r, y_tr_r, X_te_r, y_te_r,
                     X_tr_c, y_tr_c, X_te_c, y_te_c,
                     label=""):
    res_reg = {}
    res_cls = {}

    # ── RF Regressor ──
    print(f"  [{label}] RF Regressor...")
    rf_reg = RandomForestRegressor(n_estimators=150, max_depth=20,
                                   min_samples_leaf=2, n_jobs=-1, random_state=42)
    rf_reg.fit(X_tr_r, y_tr_r)
    p = rf_reg.predict(X_te_r)
    res_reg['RF'] = dict(r2=r2_score(y_te_r, p),
                         rmse=np.sqrt(mean_squared_error(y_te_r, p)),
                         mae=mean_absolute_error(y_te_r, p))
    print(f"         R²={res_reg['RF']['r2']:.4f}  RMSE={res_reg['RF']['rmse']:.4f}")

    # ── GB Regressor ──
    print(f"  [{label}] GB Regressor...")
    gb_reg = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05,
                                            max_depth=6, random_state=42)
    gb_reg.fit(X_tr_r, y_tr_r)
    p = gb_reg.predict(X_te_r)
    res_reg['GB'] = dict(r2=r2_score(y_te_r, p),
                         rmse=np.sqrt(mean_squared_error(y_te_r, p)),
                         mae=mean_absolute_error(y_te_r, p))
    print(f"         R²={res_reg['GB']['r2']:.4f}  RMSE={res_reg['GB']['rmse']:.4f}")

    # ── DNN Regressor ──
    print(f"  [{label}] DNN Regressor...")
    dnn_reg = MLPRegressor(hidden_layer_sizes=(256, 128, 64, 32), activation='relu',
                           solver='adam', learning_rate_init=0.001, max_iter=300,
                           early_stopping=True, validation_fraction=0.1,
                           n_iter_no_change=20, batch_size=256,
                           random_state=42, verbose=False)
    dnn_reg.fit(X_tr_r, y_tr_r)
    p = dnn_reg.predict(X_te_r)
    res_reg['DNN'] = dict(r2=r2_score(y_te_r, p),
                          rmse=np.sqrt(mean_squared_error(y_te_r, p)),
                          mae=mean_absolute_error(y_te_r, p))
    print(f"         R²={res_reg['DNN']['r2']:.4f}  RMSE={res_reg['DNN']['rmse']:.4f}  Epochs={dnn_reg.n_iter_}")

    # ── RF Classifier ──
    print(f"  [{label}] RF Classifier...")
    rf_cls = RandomForestClassifier(n_estimators=150, max_depth=20,
                                    min_samples_leaf=2, class_weight='balanced',
                                    n_jobs=-1, random_state=42)
    rf_cls.fit(X_tr_c, y_tr_c)
    p = rf_cls.predict(X_te_c)
    res_cls['RF'] = dict(acc=accuracy_score(y_te_c, p),
                         f1=f1_score(y_te_c, p, average='weighted', zero_division=0))
    print(f"         Acc={res_cls['RF']['acc']:.4f}  F1={res_cls['RF']['f1']:.4f}")

    # ── GB Classifier ──
    print(f"  [{label}] GB Classifier...")
    gb_cls = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05,
                                             max_depth=6, random_state=42)
    gb_cls.fit(X_tr_c, y_tr_c)
    p = gb_cls.predict(X_te_c)
    res_cls['GB'] = dict(acc=accuracy_score(y_te_c, p),
                         f1=f1_score(y_te_c, p, average='weighted', zero_division=0))
    print(f"         Acc={res_cls['GB']['acc']:.4f}  F1={res_cls['GB']['f1']:.4f}")

    # ── DNN Classifier ──
    print(f"  [{label}] DNN Classifier...")
    dnn_cls = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu',
                            solver='adam', learning_rate_init=0.001, max_iter=300,
                            early_stopping=True, validation_fraction=0.1,
                            n_iter_no_change=20, batch_size=256,
                            random_state=42, verbose=False)
    dnn_cls.fit(X_tr_c, y_tr_c)
    p = dnn_cls.predict(X_te_c)
    res_cls['DNN'] = dict(acc=accuracy_score(y_te_c, p),
                          f1=f1_score(y_te_c, p, average='weighted', zero_division=0))
    print(f"         Acc={res_cls['DNN']['acc']:.4f}  F1={res_cls['DNN']['f1']:.4f}  Epochs={dnn_cls.n_iter_}")

    return res_reg, res_cls


# ═══════════════════════════════════════════════════════════
# PIPELINE 1: Merged Raw Data
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PIPELINE 1: Merged Raw Data")
print("=" * 60)

df_marge = pd.read_csv("Data/Marge/merged_dataset.csv")
print(f"  Shape: {df_marge.shape}")

# Drop leaky cols if present
drop_cols = [c for c in ['Transplant', 'Growth', 'Harvest', 'AP Ratio'] if c in df_marge.columns]
df_marge.drop(columns=drop_cols, inplace=True)

# Remove zero production
df_marge = df_marge[df_marge['Production'] > 0].copy()

# Encode categoricals
for col in ['Season', 'District', 'Crop Name']:
    if col in df_marge.columns:
        df_marge[col + '_enc'] = LabelEncoder().fit_transform(df_marge[col])

# Log transform target
df_marge['Production_log'] = np.log1p(df_marge['Production'])

REG_FEAT_M = ['Area', 'N', 'P', 'K', 'ph',
               'Avg Temp', 'Min Temp', 'Max Temp',
               'Avg Humidity', 'Min Relative Humidity', 'Max Relative Humidity',
               'Rainfall', 'Season_enc', 'District_enc', 'Crop Name_enc']
CLS_FEAT_M = ['N', 'P', 'K', 'ph', 'Avg Temp', 'Avg Humidity', 'Rainfall',
               'Season_enc', 'District_enc']

REG_FEAT_M = [f for f in REG_FEAT_M if f in df_marge.columns]
CLS_FEAT_M = [f for f in CLS_FEAT_M if f in df_marge.columns]

X_r = df_marge[REG_FEAT_M]
y_r = df_marge['Production_log']
X_c = df_marge[CLS_FEAT_M]
y_c = df_marge['Crop Name_enc']

sc_r = StandardScaler()
sc_c = StandardScaler()
X_r_sc = pd.DataFrame(sc_r.fit_transform(X_r), columns=REG_FEAT_M)
X_c_sc = pd.DataFrame(sc_c.fit_transform(X_c), columns=CLS_FEAT_M)

X_tmp, X_te_r, y_tmp, y_te_r = train_test_split(X_r_sc, y_r, test_size=0.20, random_state=42)
X_tr_r, X_vl_r, y_tr_r, y_vl_r = train_test_split(X_tmp, y_tmp, test_size=0.10, random_state=42)

X_tmp, X_te_c, y_tmp, y_te_c = train_test_split(X_c_sc, y_c, test_size=0.20, random_state=42)
X_tr_c, X_vl_c, y_tr_c, y_vl_c = train_test_split(X_tmp, y_tmp, test_size=0.10, random_state=42)

print(f"  Train: {len(X_tr_r):,}  Val: {len(X_vl_r):,}  Test: {len(X_te_r):,}")
p1_reg, p1_cls = train_all_models(X_tr_r, y_tr_r, X_te_r, y_te_r,
                                   X_tr_c, y_tr_c, X_te_c, y_te_c,
                                   label="Merged")


# ═══════════════════════════════════════════════════════════
# PIPELINE 2: Preprocessed Data
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PIPELINE 2: Preprocessed Data")
print("=" * 60)

DATA2 = "Preprocessing/Data"
X_tr_r2 = pd.read_csv(f"{DATA2}/X_reg_train.csv")
X_te_r2 = pd.read_csv(f"{DATA2}/X_reg_test.csv")
y_tr_r2 = pd.read_csv(f"{DATA2}/y_reg_train.csv").squeeze()
y_te_r2 = pd.read_csv(f"{DATA2}/y_reg_test.csv").squeeze()

X_tr_c2 = pd.read_csv(f"{DATA2}/X_cls_train.csv")
X_te_c2 = pd.read_csv(f"{DATA2}/X_cls_test.csv")
y_tr_c2 = pd.read_csv(f"{DATA2}/y_cls_train.csv").squeeze()
y_te_c2 = pd.read_csv(f"{DATA2}/y_cls_test.csv").squeeze()

print(f"  Train: {len(X_tr_r2):,}  Test: {len(X_te_r2):,}")
p2_reg, p2_cls = train_all_models(X_tr_r2, y_tr_r2, X_te_r2, y_te_r2,
                                   X_tr_c2, y_tr_c2, X_te_c2, y_te_c2,
                                   label="Preprocess")


# ═══════════════════════════════════════════════════════════
# PIPELINE 3: Augmented Data
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PIPELINE 3: Augmented Data")
print("=" * 60)

DATA3 = "Augmentation/Data"
X_tr_r3 = pd.read_csv(f"{DATA3}/X_reg_train.csv")
X_te_r3 = pd.read_csv(f"{DATA3}/X_reg_test.csv")
y_tr_r3 = pd.read_csv(f"{DATA3}/y_reg_train.csv").squeeze()
y_te_r3 = pd.read_csv(f"{DATA3}/y_reg_test.csv").squeeze()

X_tr_c3 = pd.read_csv(f"{DATA3}/X_cls_train.csv")
X_te_c3 = pd.read_csv(f"{DATA3}/X_cls_test.csv")
y_tr_c3 = pd.read_csv(f"{DATA3}/y_cls_train.csv").squeeze()
y_te_c3 = pd.read_csv(f"{DATA3}/y_cls_test.csv").squeeze()

print(f"  Train: {len(X_tr_r3):,}  Test: {len(X_te_r3):,}")
p3_reg, p3_cls = train_all_models(X_tr_r3, y_tr_r3, X_te_r3, y_te_r3,
                                   X_tr_c3, y_tr_c3, X_te_c3, y_te_c3,
                                   label="Augmented")


# ═══════════════════════════════════════════════════════════
# MASTER COMPARISON CHART
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MASTER COMPARISON CHART তৈরি হচ্ছে...")
print("=" * 60)

pipelines  = ['Merged', 'Preprocessed', 'Augmented']
models     = ['RF', 'GB', 'DNN']
all_reg    = [p1_reg, p2_reg, p3_reg]
all_cls    = [p1_cls, p2_cls, p3_cls]

# Color per model
M_COLORS = {'RF': '#2ecc71', 'GB': '#3498db', 'DNN': '#e74c3c'}

# Bar positions
x     = np.arange(len(pipelines))
width = 0.25

fig, axes = plt.subplots(2, 3, figsize=(20, 13))
fig.suptitle(
    'Master Model Comparison — Merged vs Preprocessed vs Augmented Data\n'
    'Models: Random Forest | Gradient Boosting | DNN (MLP)  |  Team: Light Seekers | CSE-4889',
    fontsize=14, fontweight='bold', y=0.98
)

def grouped_bars(ax, metric, results_list, title, ylabel, mode='max'):
    best_val = -np.inf if mode == 'max' else np.inf
    best_pos = (0, 0)

    for mi, model in enumerate(models):
        vals = [r[model][metric] for r in results_list]
        offset = (mi - 1) * width
        bars = ax.bar(x + offset, vals, width,
                      label=model, color=M_COLORS[model],
                      edgecolor='white', linewidth=1.2)
        for bi, (b, v) in enumerate(zip(bars, vals)):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.003,
                    f'{v:.3f}', ha='center', va='bottom',
                    fontsize=7.5, fontweight='bold', rotation=0)
            if (mode == 'max' and v > best_val) or (mode == 'min' and v < best_val):
                best_val = v
                best_pos = (b, v)

    # Gold border on best bar
    best_pos[0].set_edgecolor('gold')
    best_pos[0].set_linewidth(3)

    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(pipelines, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(
        [r[m][metric] for r in results_list for m in models]
    ) * 1.18)

# ── Row 1: Regression ──
grouped_bars(axes[0, 0], 'r2',   all_reg, 'Task A — R² Score (↑ higher)', 'R²',   'max')
grouped_bars(axes[0, 1], 'rmse', all_reg, 'Task A — RMSE (↓ lower)',      'RMSE', 'min')
grouped_bars(axes[0, 2], 'mae',  all_reg, 'Task A — MAE (↓ lower)',       'MAE',  'min')

# ── Row 2: Classification ──
grouped_bars(axes[1, 0], 'acc', all_cls, 'Task B — Accuracy (↑ higher)', 'Accuracy', 'max')
grouped_bars(axes[1, 1], 'f1',  all_cls, 'Task B — F1 Score (↑ higher)', 'F1 Score', 'max')

# ── Summary Table (last cell) ──
ax = axes[1, 2]
ax.axis('off')

table_data = []
headers = ['Pipeline', 'Model', 'R²', 'RMSE', 'Acc', 'F1']

for pi, (pipe, reg, cls) in enumerate(zip(pipelines, all_reg, all_cls)):
    for model in models:
        table_data.append([
            pipe if model == 'RF' else '',
            model,
            f"{reg[model]['r2']:.4f}",
            f"{reg[model]['rmse']:.4f}",
            f"{cls[model]['acc']:.4f}",
            f"{cls[model]['f1']:.4f}",
        ])

tbl = ax.table(cellText=table_data, colLabels=headers,
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.2, 1.5)

# Header styling
for j in range(len(headers)):
    tbl[0, j].set_facecolor('#2c3e50')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

# Pipeline row highlights
pipe_colors = ['#eafaf1', '#eaf4fb', '#fef9e7']
row = 1
for pi in range(len(pipelines)):
    for mi in range(len(models)):
        for j in range(len(headers)):
            tbl[row, j].set_facecolor(pipe_colors[pi])
        row += 1

ax.set_title('Full Results Table', fontweight='bold', fontsize=11, pad=10)

# Legend
patches = [mpatches.Patch(color=M_COLORS[m], label=m) for m in models]
gold_p  = mpatches.Patch(edgecolor='gold', facecolor='none',
                          linewidth=2, label='Best overall (gold border)')
fig.legend(handles=patches + [gold_p],
           loc='lower center', ncol=4, fontsize=11,
           bbox_to_anchor=(0.5, 0.01), framealpha=0.9)

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.savefig('Output/master_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Saved: Output/master_comparison.png")


# ═══════════════════════════════════════════════════════════
# FINAL SUMMARY PRINT
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FINAL SUMMARY — সব Pipeline ও Model")
print("=" * 60)

print(f"\n  {'Pipeline':<14} {'Model':<6} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'Acc':>8} {'F1':>8}")
print("  " + "-" * 62)

best_r2   = -np.inf
best_acc  = -np.inf

for pipe, reg, cls in zip(pipelines, all_reg, all_cls):
    for model in models:
        r2   = reg[model]['r2']
        rmse = reg[model]['rmse']
        mae  = reg[model]['mae']
        acc  = cls[model]['acc']
        f1   = cls[model]['f1']
        star = ""
        if r2 > best_r2:
            best_r2 = r2
        if acc > best_acc:
            best_acc = acc
        print(f"  {pipe:<14} {model:<6} {r2:>8.4f} {rmse:>8.4f} {mae:>8.4f} {acc:>8.4f} {f1:>8.4f}")
    print()

# Overall best
all_r2  = [(p, m, all_reg[i][m]['r2'])  for i, p in enumerate(pipelines) for m in models]
all_acc = [(p, m, all_cls[i][m]['acc']) for i, p in enumerate(pipelines) for m in models]

best_r2_entry  = max(all_r2,  key=lambda x: x[2])
best_acc_entry = max(all_acc, key=lambda x: x[2])

print("=" * 60)
print(f"  🏆 Best Regression (R²) : {best_r2_entry[0]} — {best_r2_entry[1]}  →  R²={best_r2_entry[2]:.4f}")
print(f"  🏆 Best Classification  : {best_acc_entry[0]} — {best_acc_entry[1]}  →  Acc={best_acc_entry[2]:.4f}")
print("=" * 60)
print("✅ Master Comparison সম্পন্ন!")
print("→ Chart: Output/master_comparison.png")
