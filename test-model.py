"""
=============================================================
  MODEL TEST SCRIPT — Crop Yield Prediction
  Team: Light Seekers | Course: CSE-4889

  এই script দিয়ে নিজের input দিয়ে prediction নিতে পারবে
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  CROP YIELD PREDICTION — MODEL TEST")
print("=" * 60)

# ─────────────────────────────────────────────
# STEP 1: Data ও Encoding Map Load
# ─────────────────────────────────────────────
crop_map     = pd.read_csv('preprocess/RF/encoding_crop.csv')
district_map = pd.read_csv('preprocess/RF/encoding_district.csv')
season_map   = pd.read_csv('preprocess/RF/encoding_season.csv')

crop_to_code     = dict(zip(crop_map['Crop'],     crop_map['Code']))
district_to_code = dict(zip(district_map['District'], district_map['Code']))
season_to_code   = dict(zip(season_map['Season'], season_map['Code']))

code_to_crop = dict(zip(crop_map['Code'], crop_map['Crop']))

# ─────────────────────────────────────────────
# STEP 2: Model Train (preprocess data থেকে)
# ─────────────────────────────────────────────
print("\n⏳ Model training হচ্ছে, একটু অপেক্ষা করো...")

X_train_r = pd.read_csv('preprocess/RF/X_reg_train.csv')
X_test_r  = pd.read_csv('preprocess/RF/X_reg_test.csv')
y_train_r = pd.read_csv('preprocess/RF/y_reg_train.csv').squeeze()

X_train_c = pd.read_csv('preprocess/RF/X_cls_train.csv')
X_test_c  = pd.read_csv('preprocess/RF/X_cls_test.csv')
y_train_c = pd.read_csv('preprocess/RF/y_cls_train.csv').squeeze()

# Scaler fit করো (regression)
scaler_r = StandardScaler()
scaler_r.fit(X_train_r)

# Scaler fit করো (classification)
scaler_c = StandardScaler()
scaler_c.fit(X_train_c)

# Regression model — Gradient Boosting (best)
from sklearn.ensemble import HistGradientBoostingRegressor
gb_reg = HistGradientBoostingRegressor(
    max_iter=150, learning_rate=0.05, max_depth=6, random_state=42
)
gb_reg.fit(X_train_r, y_train_r)

# Classification model — Random Forest
rf_cls = RandomForestClassifier(
    n_estimators=150, max_depth=20, min_samples_leaf=2,
    class_weight='balanced', n_jobs=-1, random_state=42
)
rf_cls.fit(X_train_c, y_train_c)

print("✅ Model ready!\n")

# ─────────────────────────────────────────────
# STEP 3: Available Options দেখাও
# ─────────────────────────────────────────────
print("=" * 60)
print("📋 AVAILABLE OPTIONS")
print("=" * 60)

print("\n🌾 CROPS (Task A - Regression এর জন্য):")
crops = crop_map['Crop'].tolist()
for i, c in enumerate(crops):
    print(f"  {c}", end="\t")
    if (i+1) % 5 == 0:
        print()
print()

print("\n🗺️  DISTRICTS:")
districts = district_map['District'].tolist()
for i, d in enumerate(districts):
    print(f"  {d}", end="\t")
    if (i+1) % 5 == 0:
        print()
print()

print("\n📅 SEASONS:")
for s in season_map['Season'].tolist():
    print(f"  {s}")

# ─────────────────────────────────────────────
# STEP 4: User Input নাও
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("📝 তোমার DATA দাও")
print("=" * 60)

def get_input(prompt, valid_list=None, input_type=float):
    while True:
        val = input(prompt).strip()
        if valid_list is not None:
            if val in valid_list:
                return val
            else:
                print(f"  ❌ এটা valid না! এগুলো থেকে বেছে নাও: {valid_list}")
        else:
            try:
                return input_type(val)
            except:
                print(f"  ❌ সংখ্যা দাও!")

# Common inputs
print("\n--- সাধারণ তথ্য ---")
district = get_input("District (উপরের list থেকে): ", valid_list=districts)
season   = get_input("Season (Kharif 1 / Kharif 2 / Rabi): ", valid_list=['Kharif 1', 'Kharif 2', 'Rabi'])

print("\n--- মাটির তথ্য ---")
N  = get_input("N (Nitrogen, mg/kg) [সাধারণত 0-140]: ")
P  = get_input("P (Phosphorus, mg/kg) [সাধারণত 5-145]: ")
K  = get_input("K (Potassium, mg/kg) [সাধারণত 5-205]: ")
ph = get_input("pH [সাধারণত 3.5-9.5]: ")

print("\n--- আবহাওয়ার তথ্য ---")
avg_temp     = get_input("Average Temperature (°C) [সাধারণত 15-40]: ")
min_temp     = get_input("Min Temperature (°C): ")
max_temp     = get_input("Max Temperature (°C): ")
avg_humidity = get_input("Average Humidity (%) [সাধারণত 20-100]: ")
min_humidity = get_input("Min Humidity (%): ")
max_humidity = get_input("Max Humidity (%): ")
rainfall     = get_input("Rainfall (mm) [সাধারণত 20-3000]: ")

# ─────────────────────────────────────────────
# STEP 5: Task B — Crop Recommendation
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("🌱 TASK B: কোন ফসল ভালো হবে? (Crop Recommendation)")
print("=" * 60)

district_enc = district_to_code[district]
season_enc   = season_to_code[season]

cls_input = pd.DataFrame([[
    N, P, K, ph,
    avg_temp, avg_humidity, rainfall,
    season_enc, district_enc
]], columns=['N', 'P', 'K', 'ph',
             'Avg Temp', 'Avg Humidity', 'Rainfall',
             'Season_enc', 'District_enc'])

# Top 3 crop prediction
proba = rf_cls.predict_proba(cls_input)[0]
top3_idx = np.argsort(proba)[::-1][:3]

print(f"\n  📍 District : {district}")
print(f"  📅 Season   : {season}")
print(f"\n  🏆 Recommended Crops (Top 3):")
for rank, idx in enumerate(top3_idx):
    crop_name = code_to_crop[rf_cls.classes_[idx]]
    confidence = proba[idx] * 100
    print(f"    {rank+1}. {crop_name:<20} — confidence: {confidence:.1f}%")

# ─────────────────────────────────────────────
# STEP 6: Task A — Yield Prediction
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("📊 TASK A: কত উৎপাদন হবে? (Yield Prediction)")
print("=" * 60)

crop_name_input = get_input(
    "\nকোন ফসলের yield দেখতে চাও? (উপরের list থেকে): ",
    valid_list=crops
)
area = get_input("জমির পরিমাণ (হেক্টর): ")

crop_enc = crop_to_code[crop_name_input]

reg_input = pd.DataFrame([[
    area, N, P, K, ph,
    avg_temp, min_temp, max_temp,
    avg_humidity, min_humidity, max_humidity,
    rainfall, season_enc, district_enc, crop_enc
]], columns=[
    'Area', 'N', 'P', 'K', 'ph',
    'Avg Temp', 'Min Temp', 'Max Temp',
    'Avg Humidity', 'Min Relative Humidity', 'Max Relative Humidity',
    'Rainfall', 'Season_enc', 'District_enc', 'Crop_enc'
])

pred_log = gb_reg.predict(reg_input)[0]
pred_production = np.expm1(pred_log)  # log transform reverse

print(f"\n  🌾 Crop      : {crop_name_input}")
print(f"  📍 District  : {district}")
print(f"  📅 Season    : {season}")
print(f"  🏞️  Area      : {area} হেক্টর")
print(f"\n  ✅ Predicted Production : {pred_production:.2f} metric tons")
print(f"  ✅ Per Hectare          : {pred_production/area:.2f} metric tons/হেক্টর")

print("\n" + "=" * 60)
print("✅ PREDICTION COMPLETE!")
print("=" * 60)