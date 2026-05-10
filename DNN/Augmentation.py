"""
=============================================================
  DATA AUGMENTATION PIPELINE — Crop Yield Prediction
  Team: Light Seekers | Course: CSE-4889
  
  Original: 3,729 rows  →  Augmented: ~22,000+ rows
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────────
# STEP 0: Load Cleaned Dataset
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 0: Loading Dataset")
print("=" * 60)

df = pd.read_csv('preprocess/RF/dataset_cleaned.csv')
print(f"✅ Original shape: {df.shape}")

# Numeric columns যেগুলোতে augmentation করব
# Season_enc, District_enc, Crop_enc — এগুলো categorical, noise দেব না
NUMERIC_COLS = [
    'Area', 'N', 'P', 'K', 'ph',
    'Avg Temp', 'Min Temp', 'Max Temp',
    'Avg Humidity', 'Min Relative Humidity', 'Max Relative Humidity',
    'Rainfall', 'Production'
]

# Categorical columns — augmentation-এ same রাখব
CATEGORICAL_COLS = ['District', 'Season', 'Crop Name', 
                    'Season_enc', 'District_enc', 'Crop_enc']


# ─────────────────────────────────────────────
# STEP 1: Train/Test Split আগে করো
# (Test set-এ কোনো augmented data থাকবে না)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: Train/Test Split (আগে করতে হবে)")
print("=" * 60)

# Rare class (only 1 row) — stratify করা সম্ভব না, train-এ রাখো
rare_mask = df['Crop_enc'].map(df['Crop_enc'].value_counts()) == 1
df_rare   = df[rare_mask]
df_common = df[~rare_mask]

df_train_common, df_test = train_test_split(
    df_common, test_size=0.10, random_state=42, stratify=df_common['Crop_enc']
)
df_train_orig = pd.concat([df_train_common, df_rare], ignore_index=True)

print(f"✅ Original Train set : {len(df_train_orig)} rows")
print(f"✅ Test set (original): {len(df_test)} rows  ← এটা পরিবর্তন হবে না")


# ─────────────────────────────────────────────
# TECHNIQUE 1: Gaussian Noise Augmentation
# প্রতিটা numeric value-এ ছোট random noise যোগ
# noise = 3% of column std — বাস্তবসম্মত variation
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TECHNIQUE 1: Gaussian Noise Augmentation")
print("=" * 60)

def gaussian_noise_augment(df, n_copies=3, noise_pct=0.03):
    """
    প্রতিটা row-এর numeric value-এ ছোট Gaussian noise যোগ করে
    n_copies নতুন row তৈরি করে।
    noise_pct = column std-এর কত % noise দেব
    """
    augmented_list = []
    
    for copy_num in range(n_copies):
        df_copy = df.copy()
        
        for col in NUMERIC_COLS:
            col_std  = df[col].std()
            noise    = np.random.normal(
                loc=0,
                scale=col_std * noise_pct,
                size=len(df)
            )
            df_copy[col] = df[col] + noise
            # Negative value যেন না হয়
            df_copy[col] = df_copy[col].clip(lower=0)
        
        # Production_log আবার recalculate করো
        df_copy['Production_log'] = np.log1p(df_copy['Production'])
        
        augmented_list.append(df_copy)
    
    return pd.concat(augmented_list, ignore_index=True)


noise_data = gaussian_noise_augment(df_train_orig, n_copies=3, noise_pct=0.03)
print(f"✅ Gaussian Noise augmented rows: {len(noise_data)}")
print(f"   (3 copies × {len(df_train_orig)} original train rows)")


# ─────────────────────────────────────────────
# TECHNIQUE 2: Class Interpolation (SMOTE-style)
# একই crop-এর দুটো row মিশিয়ে নতুন row বানাও
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TECHNIQUE 2: Class Interpolation (SMOTE-style)")
print("=" * 60)

def class_interpolation_augment(df, n_synthetic_per_row=2):
    """
    একই Crop Name-এর দুটো random row নিয়ে
    linear interpolation করে synthetic row তৈরি করে।
    alpha = mixing ratio (0.3 থেকে 0.7 এর মধ্যে)
    """
    synthetic_rows = []
    
    for crop in df['Crop Name'].unique():
        crop_df = df[df['Crop Name'] == crop].reset_index(drop=True)
        
        if len(crop_df) < 2:
            continue
        
        for _ in range(n_synthetic_per_row * len(crop_df)):
            # দুটো random row বেছে নাও
            idx_i, idx_j = np.random.choice(len(crop_df), 2, replace=False)
            row_i = crop_df.iloc[idx_i]
            row_j = crop_df.iloc[idx_j]
            
            # Random mixing ratio
            alpha = np.random.uniform(0.3, 0.7)
            
            # নতুন synthetic row
            new_row = row_i.copy()
            for col in NUMERIC_COLS:
                new_row[col] = alpha * row_i[col] + (1 - alpha) * row_j[col]
            
            new_row['Production_log'] = np.log1p(new_row['Production'])
            synthetic_rows.append(new_row)
    
    return pd.DataFrame(synthetic_rows).reset_index(drop=True)


interp_data = class_interpolation_augment(df_train_orig, n_synthetic_per_row=2)
print(f"✅ Interpolation augmented rows: {len(interp_data)}")
print(f"   (2 synthetic rows per original row, same-class mixing)")


# ─────────────────────────────────────────────
# TECHNIQUE 3: Seasonal Variation Augmentation
# Season অনুযায়ী realistic temp/humidity shift
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TECHNIQUE 3: Seasonal Variation Augmentation")
print("=" * 60)

# Bangladesh-এর বাস্তব seasonal variation
SEASON_SHIFTS = {
    'Kharif 1': {'Avg Temp': +1.5,  'Avg Humidity': +3.0,  'Rainfall': +15.0},
    'Kharif 2': {'Avg Temp': +1.0,  'Avg Humidity': +2.0,  'Rainfall': +10.0},
    'Rabi':     {'Avg Temp': -1.5,  'Avg Humidity': -3.0,  'Rainfall': -8.0},
}

def seasonal_variation_augment(df):
    """
    প্রতিটা row-এর Season অনুযায়ী realistic
    temperature, humidity, rainfall shift করে
    নতুন row তৈরি করে।
    """
    augmented_rows = []
    
    for _, row in df.iterrows():
        season  = row['Season']
        shifts  = SEASON_SHIFTS.get(season, {})
        
        new_row = row.copy()
        
        for col, shift in shifts.items():
            # Shift + small random noise
            variation  = np.random.uniform(-0.3, 0.3) * abs(shift)
            new_row[col] = max(0, row[col] + shift + variation)
        
        # Temp columns consistent রাখো
        if 'Avg Temp' in shifts:
            temp_delta = new_row['Avg Temp'] - row['Avg Temp']
            new_row['Min Temp'] = max(0, row['Min Temp'] + temp_delta * 0.8)
            new_row['Max Temp'] = row['Max Temp'] + temp_delta * 1.2
        
        if 'Avg Humidity' in shifts:
            hum_delta = new_row['Avg Humidity'] - row['Avg Humidity']
            new_row['Min Relative Humidity'] = max(0, row['Min Relative Humidity'] + hum_delta)
            new_row['Max Relative Humidity'] = min(100, row['Max Relative Humidity'] + hum_delta)
        
        # Production slight variation
        prod_noise = np.random.uniform(-0.05, 0.05) * row['Production']
        new_row['Production'] = max(1, row['Production'] + prod_noise)
        new_row['Production_log'] = np.log1p(new_row['Production'])
        
        augmented_rows.append(new_row)
    
    return pd.DataFrame(augmented_rows).reset_index(drop=True)


seasonal_data = seasonal_variation_augment(df_train_orig)
print(f"✅ Seasonal variation augmented rows: {len(seasonal_data)}")
print(f"   (1 shifted copy per original train row)")


# ─────────────────────────────────────────────
# STEP 2: সব Data একত্রিত করো
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: সব Data Combine করা")
print("=" * 60)

df_train_augmented = pd.concat([
    df_train_orig,    # Original train data
    noise_data,       # Gaussian noise copies
    interp_data,      # Interpolated synthetic rows
    seasonal_data,    # Seasonal shifted rows
], ignore_index=True)

# Shuffle করো
df_train_augmented = df_train_augmented.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Original train  : {len(df_train_orig):>6,} rows")
print(f"Gaussian noise  : {len(noise_data):>6,} rows")
print(f"Interpolation   : {len(interp_data):>6,} rows")
print(f"Seasonal shift  : {len(seasonal_data):>6,} rows")
print(f"─────────────────────────────────────")
print(f"Total train     : {len(df_train_augmented):>6,} rows  ✅")
print(f"Test set        : {len(df_test):>6,} rows  (original, unchanged)")
print(f"Grand total     : {len(df_train_augmented) + len(df_test):>6,} rows")


# ─────────────────────────────────────────────
# STEP 3: Validation Split (train থেকে 10%)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Validation Split")
print("=" * 60)

df_train_final, df_val = train_test_split(
    df_train_augmented, test_size=0.10, random_state=42
)

print(f"✅ Final Train      : {len(df_train_final):,} rows")
print(f"✅ Validation       : {len(df_val):,} rows")
print(f"✅ Test (original)  : {len(df_test):,} rows")


# ─────────────────────────────────────────────
# STEP 4: Feature/Target Split করো
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Feature & Target Split")
print("=" * 60)

# Regression features
REG_FEATURES = [
    'Area', 'N', 'P', 'K', 'ph',
    'Avg Temp', 'Min Temp', 'Max Temp',
    'Avg Humidity', 'Min Relative Humidity', 'Max Relative Humidity',
    'Rainfall', 'Season_enc', 'District_enc', 'Crop_enc'
]
REG_TARGET = 'Production_log'

# Classification features
CLS_FEATURES = [
    'N', 'P', 'K', 'ph',
    'Avg Temp', 'Avg Humidity', 'Rainfall',
    'Season_enc', 'District_enc'
]
CLS_TARGET = 'Crop_enc'

# --- Regression ---
X_train_reg = df_train_final[REG_FEATURES]
y_train_reg = df_train_final[REG_TARGET]

X_val_reg   = df_val[REG_FEATURES]
y_val_reg   = df_val[REG_TARGET]

X_test_reg  = df_test[REG_FEATURES]
y_test_reg  = df_test[REG_TARGET]

# --- Classification ---
X_train_cls = df_train_final[CLS_FEATURES]
y_train_cls = df_train_final[CLS_TARGET]

X_val_cls   = df_val[CLS_FEATURES]
y_val_cls   = df_val[CLS_TARGET]

X_test_cls  = df_test[CLS_FEATURES]
y_test_cls  = df_test[CLS_TARGET]

print(f"✅ Regression  — X_train: {X_train_reg.shape} | X_val: {X_val_reg.shape} | X_test: {X_test_reg.shape}")
print(f"✅ Classification — X_train: {X_train_cls.shape} | X_val: {X_val_cls.shape} | X_test: {X_test_cls.shape}")


# ─────────────────────────────────────────────
# STEP 5: Save সব Files
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Files Save করা")
print("=" * 60)

import os
os.makedirs('Data/agumnetation', exist_ok=True)
os.makedirs('preprocess/dnn', exist_ok=True)

# Full augmented datasets
df_train_final.to_csv('Data/agumnetation/train_augmented.csv', index=False)
df_val.to_csv('Data/agumnetation/val_augmented.csv', index=False)
df_test.to_csv('Data/agumnetation/test_original.csv', index=False)

# Regression splits
X_train_reg.to_csv('X_train_reg.csv', index=False)
y_train_reg.to_csv('y_train_reg.csv', index=False)
X_val_reg.to_csv('X_val_reg.csv',     index=False)
y_val_reg.to_csv('y_val_reg.csv',     index=False)
X_test_reg.to_csv('X_test_reg.csv',   index=False)
y_test_reg.to_csv('y_test_reg.csv',   index=False)

# Classification splits
X_train_cls.to_csv('X_train_cls.csv', index=False)
y_train_cls.to_csv('y_train_cls.csv', index=False)
X_val_cls.to_csv('X_val_cls.csv',     index=False)
y_val_cls.to_csv('y_val_cls.csv',     index=False)
X_test_cls.to_csv('X_test_cls.csv',   index=False)
y_test_cls.to_csv('y_test_cls.csv',   index=False)

print("✅ Saved files:")
print("   train_augmented.csv, val_augmented.csv, test_original.csv")
print("   X_train_reg.csv, y_train_reg.csv, X_val_reg.csv, y_val_reg.csv")
print("   X_test_reg.csv, y_test_reg.csv")
print("   X_train_cls.csv, y_train_cls.csv, X_val_cls.csv, y_val_cls.csv")
print("   X_test_cls.csv, y_test_cls.csv")


# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ AUGMENTATION COMPLETE — FINAL SUMMARY")
print("=" * 60)
print(f"  Original dataset       : 3,729 rows")
print(f"  Final Train set        : {len(df_train_final):,} rows  (augmented)")
print(f"  Validation set         : {len(df_val):,} rows  (augmented)")
print(f"  Test set               : {len(df_test):,} rows  (ORIGINAL ONLY)")
print(f"  Total                  : {len(df_train_final)+len(df_val)+len(df_test):,} rows")
print(f"  Increase               : {(len(df_train_final)+len(df_val)+len(df_test))/3729:.1f}x")
print(f"")
print(f"  Regression features    : {len(REG_FEATURES)}")
print(f"  Classification features: {len(CLS_FEATURES)}")
print(f"  Regression target      : Production_log")
print(f"  Classification target  : Crop_enc (72 classes)")
print("=" * 60)
print("\n  এখন DNN model train করতে পারবে! 🎉")

