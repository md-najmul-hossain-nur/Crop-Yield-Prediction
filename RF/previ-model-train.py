"""
=============================================================
  MODEL TRAINING — Crop Yield Prediction
  Team: Light Seekers | Course: CSE-4889

  Source : Data/Marge/merged_dataset.csv
  Flow   : Load -> preprocess -> split -> train -> evaluate

  Task A: Yield Prediction     -> Regression
  Task B: Crop Recommendation  -> Classification
=============================================================
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")


print("=" * 65)
print("STEP 0: Load Merged Dataset")
print("=" * 65)
df = pd.read_csv("Data/Marge/merged_dataset.csv")
print(f"Loaded shape: {df.shape}")

RAW_COMPARE_MODE = True
print(f"Raw compare mode: {RAW_COMPARE_MODE}")

print("\n" + "=" * 65)
print("STEP 1: Prepare Dataset")
print("=" * 65)
required_cols = [
    "District",
    "Season",
    "Crop Name",
    "Area",
    "N",
    "P",
    "K",
    "ph",
    "Avg Temp",
    "Min Temp",
    "Max Temp",
    "Avg Humidity",
    "Min Relative Humidity",
    "Max Relative Humidity",
    "Rainfall",
    "Production",
]

df = df.dropna(subset=required_cols).copy()

if RAW_COMPARE_MODE:
    # Keep the merged dataset as raw as possible for before-preprocessing comparison.
    # Only remove impossible target values that would break training.
    df = df[df["Production"] > 0].copy()
    print(f"Rows kept for raw comparison: {len(df):,}")
else:
    drop_cols = ["Transplant", "Growth", "Harvest", "AP Ratio"]
    df = df.drop(columns=drop_cols, errors="ignore")
    df = df[df["Production"] > 0].copy()

    q1 = df["Production"].quantile(0.25)
    q3 = df["Production"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 3 * iqr
    upper = q3 + 3 * iqr
    df = df[(df["Production"] >= lower) & (df["Production"] <= upper)].copy()
    print(f"Rows after cleaning: {len(df):,}")


print("\n" + "=" * 65)
print("STEP 2: Encode Categorical Columns")
print("=" * 65)
le_season = LabelEncoder()
le_district = LabelEncoder()
le_crop = LabelEncoder()

df["Season_enc"] = le_season.fit_transform(df["Season"])
df["District_enc"] = le_district.fit_transform(df["District"])
df["Crop_enc"] = le_crop.fit_transform(df["Crop Name"])

if RAW_COMPARE_MODE:
    df["Production_target"] = df["Production"]
else:
    df["Production_target"] = np.log1p(df["Production"])

print(f"Season classes   : {len(le_season.classes_)}")
print(f"District classes : {len(le_district.classes_)}")
print(f"Crop classes     : {len(le_crop.classes_)}")


REG_FEATURES = [
    "Area",
    "N",
    "P",
    "K",
    "ph",
    "Avg Temp",
    "Min Temp",
    "Max Temp",
    "Avg Humidity",
    "Min Relative Humidity",
    "Max Relative Humidity",
    "Rainfall",
    "Season_enc",
    "District_enc",
    "Crop_enc",
]
CLS_FEATURES = [
    "N",
    "P",
    "K",
    "ph",
    "Avg Temp",
    "Avg Humidity",
    "Rainfall",
    "Season_enc",
    "District_enc",
]


print("\n" + "=" * 65)
print("STEP 3: Split and Scale")
print("=" * 65)
X_reg = df[REG_FEATURES].copy()
y_reg = df["Production_target"].copy()

X_cls = df[CLS_FEATURES].copy()
y_cls = df["Crop_enc"].copy()

X_train_r_raw, X_test_r_raw, y_train_r_full, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
X_train_r_raw, X_val_r_raw, y_train_r, y_val_r = train_test_split(
    X_train_r_raw, y_train_r_full, test_size=0.1, random_state=42
)

X_train_c_raw, X_test_c_raw, y_train_c_full, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)
X_train_c_raw, X_val_c_raw, y_train_c, y_val_c = train_test_split(
    X_train_c_raw, y_train_c_full, test_size=0.1, random_state=42
)

if RAW_COMPARE_MODE:
    X_train_r = X_train_r_raw.reset_index(drop=True)
    X_val_r = X_val_r_raw.reset_index(drop=True)
    X_test_r = X_test_r_raw.reset_index(drop=True)

    X_train_c = X_train_c_raw.reset_index(drop=True)
    X_val_c = X_val_c_raw.reset_index(drop=True)
    X_test_c = X_test_c_raw.reset_index(drop=True)
    print("Scaling skipped for raw comparison.")
else:
    scaler_reg = StandardScaler()
    X_train_r = pd.DataFrame(
        scaler_reg.fit_transform(X_train_r_raw), columns=REG_FEATURES
    )
    X_val_r = pd.DataFrame(scaler_reg.transform(X_val_r_raw), columns=REG_FEATURES)
    X_test_r = pd.DataFrame(scaler_reg.transform(X_test_r_raw), columns=REG_FEATURES)

    scaler_cls = StandardScaler()
    X_train_c = pd.DataFrame(
        scaler_cls.fit_transform(X_train_c_raw), columns=CLS_FEATURES
    )
    X_val_c = pd.DataFrame(scaler_cls.transform(X_val_c_raw), columns=CLS_FEATURES)
    X_test_c = pd.DataFrame(scaler_cls.transform(X_test_c_raw), columns=CLS_FEATURES)

print(f"Train (Reg): {X_train_r.shape[0]:,} rows")
print(f"Val   (Reg): {X_val_r.shape[0]:,} rows")
print(f"Test  (Reg): {X_test_r.shape[0]:,} rows")
print(f"Train (Cls): {X_train_c.shape[0]:,} rows")
print(f"Val   (Cls): {X_val_c.shape[0]:,} rows")
print(f"Test  (Cls): {X_test_c.shape[0]:,} rows")


def reg_report(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"  {name:<12} R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")
    return r2, rmse, mae


def cls_report(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"  {name:<12} Accuracy={acc:.4f}  F1={f1:.4f}")
    return acc, f1


print("\n" + "=" * 65)
print("TASK A: REGRESSION — Yield Prediction")
print("=" * 65)
reg_results = {}

print("\n[1] Random Forest Regressor")
rf_reg = RandomForestRegressor(
    n_estimators=150,
    max_depth=20,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
)
rf_reg.fit(X_train_r, y_train_r)
v_r2, _, _ = reg_report("Validation", y_val_r, rf_reg.predict(X_val_r))
t_r2, t_rmse, t_mae = reg_report("Test", y_test_r, rf_reg.predict(X_test_r))
reg_results["Random Forest"] = {
    "val_r2": v_r2,
    "test_r2": t_r2,
    "test_rmse": t_rmse,
    "test_mae": t_mae,
}

print("\n[2] Gradient Boosting Regressor")
gb_reg = HistGradientBoostingRegressor(
    max_iter=150, learning_rate=0.05, max_depth=6, random_state=42
)
gb_reg.fit(X_train_r, y_train_r)
v_r2, _, _ = reg_report("Validation", y_val_r, gb_reg.predict(X_val_r))
t_r2, t_rmse, t_mae = reg_report("Test", y_test_r, gb_reg.predict(X_test_r))
reg_results["Gradient Boosting"] = {
    "val_r2": v_r2,
    "test_r2": t_r2,
    "test_rmse": t_rmse,
    "test_mae": t_mae,
}

best_reg = max(reg_results, key=lambda name: reg_results[name]["test_r2"])


print("\n" + "=" * 65)
print("TASK B: CLASSIFICATION — Crop Recommendation")
print("=" * 65)
cls_results = {}

print("\n[1] Random Forest Classifier")
rf_cls = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
rf_cls.fit(X_train_c, y_train_c)
v_acc, _ = cls_report("Validation", y_val_c, rf_cls.predict(X_val_c))
t_acc, t_f1 = cls_report("Test", y_test_c, rf_cls.predict(X_test_c))
cls_results["Random Forest"] = {
    "val_acc": v_acc,
    "test_acc": t_acc,
    "test_f1": t_f1,
}

print("\n[2] Gradient Boosting Classifier")
gb_cls = HistGradientBoostingClassifier(
    max_iter=150, learning_rate=0.05, max_depth=6, random_state=42
)
gb_cls.fit(X_train_c, y_train_c)
v_acc, _ = cls_report("Validation", y_val_c, gb_cls.predict(X_val_c))
t_acc, t_f1 = cls_report("Test", y_test_c, gb_cls.predict(X_test_c))
cls_results["Gradient Boosting"] = {
    "val_acc": v_acc,
    "test_acc": t_acc,
    "test_f1": t_f1,
}

best_cls = max(cls_results, key=lambda name: cls_results[name]["test_f1"])


print("\n" + "=" * 65)
print("Generating charts...")
print("=" * 65)
models = list(reg_results.keys())
colors = ["#2ecc71", "#3498db"]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    "Model Comparison — Crop Yield Prediction\nTeam: Light Seekers | CSE-4889",
    fontsize=14,
    fontweight="bold",
)

ax = axes[0, 0]
vals = [reg_results[m]["test_r2"] for m in models]
bars = ax.bar(models, vals, color=colors, width=0.4, edgecolor="white")
ax.set_title("Task A — R2 Score", fontweight="bold")
ax.set_ylim(0, 1.1)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.4f}", ha="center")

ax = axes[0, 1]
vals = [reg_results[m]["test_rmse"] for m in models]
bars = ax.bar(models, vals, color=colors, width=0.4, edgecolor="white")
ax.set_title("Task A — RMSE (lower=better)", fontweight="bold")
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f"{val:.4f}", ha="center")

ax = axes[0, 2]
best_reg_preds = rf_reg.predict(X_test_r) if best_reg == "Random Forest" else gb_reg.predict(X_test_r)
mn = min(float(y_test_r.min()), float(best_reg_preds.min()))
mx = max(float(y_test_r.max()), float(best_reg_preds.max()))
ax.scatter(y_test_r, best_reg_preds, alpha=0.5, color="#3498db", s=20)
ax.plot([mn, mx], [mn, mx], "r--", lw=2, label="Ideal")
ax.set_title(f"Task A — Actual vs Predicted\n({best_reg})", fontweight="bold")
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.legend()

ax = axes[1, 0]
vals = [cls_results[m]["test_acc"] for m in models]
bars = ax.bar(models, vals, color=colors, width=0.4, edgecolor="white")
ax.set_title("Task B — Accuracy", fontweight="bold")
ax.set_ylim(0, 1.1)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.4f}", ha="center")

ax = axes[1, 1]
vals = [cls_results[m]["test_f1"] for m in models]
bars = ax.bar(models, vals, color=colors, width=0.4, edgecolor="white")
ax.set_title("Task B — F1 Score (Weighted)", fontweight="bold")
ax.set_ylim(0, 1.1)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.4f}", ha="center")

ax = axes[1, 2]
pd.Series(rf_cls.feature_importances_, index=CLS_FEATURES).sort_values().plot(
    kind="barh", ax=ax, color="#2ecc71", edgecolor="white"
)
ax.set_title("Task B — Feature Importance\n(Random Forest)", fontweight="bold")
ax.set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
print("Saved: model_comparison.png")

fig2, ax2 = plt.subplots(figsize=(10, 6))
pd.Series(rf_reg.feature_importances_, index=REG_FEATURES).sort_values().plot(
    kind="barh", ax=ax2, color="#3498db", edgecolor="white"
)
ax2.set_title(
    "Task A — Feature Importance (Random Forest Regressor)\n"
    "Team: Light Seekers | CSE-4889",
    fontweight="bold",
)
ax2.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance_regression.png", dpi=150, bbox_inches="tight")
print("Saved: feature_importance_regression.png")


print("\n" + "=" * 65)
print("FINAL RESULT TABLE")
print("=" * 65)
print("\nTask A — Regression")
print(f"{'Model':<22} {'Val R2':>8} {'Test R2':>9} {'RMSE':>8} {'MAE':>8}")
print("-" * 60)
for name, res in reg_results.items():
    tag = " <- BEST" if name == best_reg else ""
    print(
        f"{name:<22} {res['val_r2']:>8.4f} {res['test_r2']:>9.4f} "
        f"{res['test_rmse']:>8.4f} {res['test_mae']:>8.4f}{tag}"
    )

print("\nTask B — Classification")
print(f"{'Model':<22} {'Val Acc':>8} {'Test Acc':>9} {'F1':>8}")
print("-" * 55)
for name, res in cls_results.items():
    tag = " <- BEST" if name == best_cls else ""
    print(
        f"{name:<22} {res['val_acc']:>8.4f} {res['test_acc']:>9.4f} "
        f"{res['test_f1']:>8.4f}{tag}"
    )

print("\n" + "=" * 65)
print(f"Best Regression    : {best_reg}")
print(f"Best Classification: {best_cls}")
print("=" * 65)
print("\nTraining completed!")
