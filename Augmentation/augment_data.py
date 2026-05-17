"""
=============================================================
  DATA AUGMENTATION — merged_dataset.csv
  Team: Light Seekers | Course: CSE-4889

  Strategy:
    1. Gaussian Noise Injection    (numeric features)
    2. Crop-wise Interpolation     (between districts)
    3. SMOTE-style KNN Synthesis   (neighborhood sampling)
    4. Per-crop scaling variation  (realistic yield variance)

  Target: 4607 → ~45,000 rows
  Output: Data/Marge/augmented_dataset.csv
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import NearestNeighbors

np.random.seed(42)

# ─────────────────────────────────────────────
# STEP 1: Load
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading merged_dataset.csv")
print("=" * 60)

df = pd.read_csv('Data/Marge/merged_dataset.csv')
print(f"  Original shape : {df.shape}")
print(f"  Columns        : {list(df.columns)}")

ORIG_LEN = len(df)

# Separate column types
CAT_COLS  = ['District', 'Season', 'Crop Name', 'Transplant', 'Growth', 'Harvest']
NUM_COLS  = ['Area', 'N', 'P', 'K', 'ph',
             'Avg Temp', 'Min Temp', 'Max Temp',
             'Avg Humidity', 'Min Relative Humidity', 'Max Relative Humidity',
             'Rainfall', 'AP Ratio', 'Production']

# Per-column realistic noise scale (% of std)
# Tighter for soil/climate, looser for area/production
NOISE_SCALE = {
    'Area'                   : 0.12,
    'N'                      : 0.10,
    'P'                      : 0.10,
    'K'                      : 0.10,
    'ph'                     : 0.04,   # pH doesn't vary much
    'Avg Temp'               : 0.06,
    'Min Temp'               : 0.06,
    'Max Temp'               : 0.06,
    'Avg Humidity'           : 0.06,
    'Min Relative Humidity'  : 0.06,
    'Max Relative Humidity'  : 0.06,
    'Rainfall'               : 0.10,
    'AP Ratio'               : 0.08,
    'Production'             : 0.15,
}

# Global per-column stats (used for clipping)
col_stats = {}
for col in NUM_COLS:
    col_stats[col] = {
        'min' : df[col].min(),
        'max' : df[col].max(),
        'std' : df[col].std(),
        'mean': df[col].mean(),
    }

print(f"\n  Numeric columns  : {NUM_COLS}")
print(f"  Categorical cols : {CAT_COLS}")

all_augmented = [df.copy()]   # always keep original

# ─────────────────────────────────────────────
# STEP 2: Method A — Gaussian Noise Injection
#   Per-group noise so inter-crop variance is preserved
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Method A — Gaussian Noise Injection (×5 copies)")
print("=" * 60)

def add_noise(source_df, n_copies=5):
    """Add Gaussian noise to each row, n_copies times."""
    noise_rows = []
    for _ in range(n_copies):
        copy = source_df.copy()
        for col in NUM_COLS:
            sigma = NOISE_SCALE[col] * col_stats[col]['std']
            noise = np.random.normal(0, sigma, size=len(copy))
            copy[col] = copy[col] + noise

        # Clip to realistic domain
        copy['ph']   = copy['ph'].clip(5.0, 8.5)
        copy['Area'] = copy['Area'].clip(0, col_stats['Area']['max'] * 1.1)
        copy['Production'] = copy['Production'].clip(0, col_stats['Production']['max'] * 1.15)
        copy['Avg Humidity']          = copy['Avg Humidity'].clip(20, 100)
        copy['Min Relative Humidity'] = copy['Min Relative Humidity'].clip(20, 100)
        copy['Max Relative Humidity'] = copy['Max Relative Humidity'].clip(20, 100)
        copy['Rainfall']  = copy['Rainfall'].clip(50, 600)
        copy['N']  = copy['N'].clip(0, 200)
        copy['P']  = copy['P'].clip(0, 200)
        copy['K']  = copy['K'].clip(0, 250)
        copy['Min Temp'] = copy['Min Temp'].clip(-5, 40)
        copy['Max Temp'] = copy['Max Temp'].clip(15, 55)
        copy['Avg Temp'] = copy['Avg Temp'].clip(5, 45)

        # Ensure Min < Avg < Max for temps & humidity
        copy['Min Temp'] = np.minimum(copy['Min Temp'], copy['Avg Temp'] - 0.5)
        copy['Max Temp'] = np.maximum(copy['Max Temp'], copy['Avg Temp'] + 0.5)
        copy['Min Relative Humidity'] = np.minimum(
            copy['Min Relative Humidity'], copy['Avg Humidity'] - 1)
        copy['Max Relative Humidity'] = np.maximum(
            copy['Max Relative Humidity'], copy['Avg Humidity'] + 1)

        # Integer columns
        for icol in ['Area', 'Max Temp', 'Min Relative Humidity',
                     'Max Relative Humidity', 'Production']:
            copy[icol] = copy[icol].round().astype(int)

        noise_rows.append(copy)

    return pd.concat(noise_rows, ignore_index=True)

method_a = add_noise(df, n_copies=5)
all_augmented.append(method_a)
print(f"  Method A rows : {len(method_a):,}  (5 × {ORIG_LEN:,})")

# ─────────────────────────────────────────────
# STEP 3: Method B — KNN Interpolation
#   For each crop, find K nearest neighbors (by numeric features)
#   and synthesize new rows by weighted interpolation (like SMOTE)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Method B — KNN Interpolation (SMOTE-style)")
print("=" * 60)

FEAT_COLS = ['N', 'P', 'K', 'ph', 'Avg Temp', 'Avg Humidity', 'Rainfall']
K_NEIGHBORS = 5
SYNTH_PER_ROW = 3   # synthetic samples per original row

synth_rows = []

for crop, grp in df.groupby('Crop Name'):
    grp = grp.reset_index(drop=True)
    n = len(grp)

    if n < 3:
        # Just noise for tiny groups
        for _ in range(SYNTH_PER_ROW * n):
            row = grp.sample(1).copy()
            for col in NUM_COLS:
                sigma = NOISE_SCALE[col] * col_stats[col]['std'] * 0.5
                row[col] += np.random.normal(0, sigma)
            synth_rows.append(row)
        continue

    k = min(K_NEIGHBORS, n - 1)
    X = grp[FEAT_COLS].values

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, indices = nbrs.kneighbors(X)

    for i in range(n):
        neighbors = indices[i][1:]          # exclude self
        for _ in range(SYNTH_PER_ROW):
            j = np.random.choice(neighbors)
            alpha = np.random.uniform(0.2, 0.8)

            new_row = grp.iloc[i].copy()
            for col in NUM_COLS:
                new_row[col] = (alpha * grp.iloc[i][col] +
                                (1 - alpha) * grp.iloc[j][col])
            synth_rows.append(new_row.to_frame().T)

method_b = pd.concat(synth_rows, ignore_index=True)

# Fix dtypes
for icol in ['Area', 'Max Temp', 'Min Relative Humidity',
             'Max Relative Humidity', 'Production']:
    method_b[icol] = pd.to_numeric(method_b[icol], errors='coerce').fillna(0).round().astype(int)
for col in ['N','P','K','ph','Avg Temp','Min Temp','Avg Humidity','Rainfall','AP Ratio']:
    method_b[col] = pd.to_numeric(method_b[col], errors='coerce').astype(float)

# Clip same as method A
method_b['ph'] = method_b['ph'].clip(5.0, 8.5)
method_b['Rainfall'] = method_b['Rainfall'].clip(50, 600)
method_b['Production'] = method_b['Production'].clip(0, None)

all_augmented.append(method_b)
print(f"  Method B rows : {len(method_b):,}  (~{SYNTH_PER_ROW}× per crop group)")

# ─────────────────────────────────────────────
# STEP 4: Method C — Season-Aware Scaling
#   Each crop×district row gets seasonal perturbation:
#   Production scaled by a crop-realistic factor (±15%)
#   Climate features shifted slightly (realistic seasonal drift)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Method C — Season-Aware Production Scaling")
print("=" * 60)

SCALE_COPIES = 3
scaled_rows = []

# Crop-level production range (for realistic scale factor bounds)
crop_prod_std = df.groupby('Crop Name')['Production'].std().fillna(1)

for _, row in df.iterrows():
    crop = row['Crop Name']
    for _ in range(SCALE_COPIES):
        new_row = row.copy()

        # Scale production within ±20% of original
        scale = np.random.uniform(0.80, 1.20)
        new_row['Production'] = max(1, round(row['Production'] * scale))

        # Area proportionally
        area_scale = np.random.uniform(0.85, 1.15)
        new_row['Area'] = max(0, round(row['Area'] * area_scale))

        # Climate small drift
        new_row['Avg Temp']   += np.random.uniform(-1.5, 1.5)
        new_row['Rainfall']   += np.random.uniform(-20, 20)
        new_row['Avg Humidity'] += np.random.uniform(-3, 3)

        # Soil slight variation
        new_row['N']  += np.random.uniform(-5, 5)
        new_row['P']  += np.random.uniform(-5, 5)
        new_row['K']  += np.random.uniform(-5, 5)
        new_row['ph'] += np.random.uniform(-0.1, 0.1)

        # Clip
        new_row['ph']  = np.clip(new_row['ph'], 5.0, 8.5)
        new_row['N']   = np.clip(new_row['N'], 0, 200)
        new_row['Rainfall'] = np.clip(new_row['Rainfall'], 50, 600)

        scaled_rows.append(new_row)

method_c = pd.DataFrame(scaled_rows)
method_c.reset_index(drop=True, inplace=True)
all_augmented.append(method_c)
print(f"  Method C rows : {len(method_c):,}  ({SCALE_COPIES}× per row)")

# ─────────────────────────────────────────────
# STEP 5: Combine & Deduplicate
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Combining All Methods")
print("=" * 60)

combined = pd.concat(all_augmented, ignore_index=True)
print(f"  Before dedup : {len(combined):,} rows")

# Drop exact duplicates (shouldn't be many but just in case)
combined.drop_duplicates(inplace=True)
combined.reset_index(drop=True, inplace=True)
print(f"  After dedup  : {len(combined):,} rows")

# If we overshot 50k, sample down; if we undershot 40k, add more noise
TARGET_MIN = 40_000
TARGET_MAX = 50_000

if len(combined) > TARGET_MAX:
    # Keep original + random sample of augmented
    orig   = combined.iloc[:ORIG_LEN]
    extra  = combined.iloc[ORIG_LEN:].sample(n=TARGET_MAX - ORIG_LEN, random_state=42)
    combined = pd.concat([orig, extra], ignore_index=True)
    print(f"  Trimmed to   : {len(combined):,} rows")
elif len(combined) < TARGET_MIN:
    # Add more gaussian noise to fill up
    needed = TARGET_MIN - len(combined)
    extra  = add_noise(df.sample(n=min(needed, ORIG_LEN), replace=True,
                                  random_state=99), n_copies=1)
    extra  = extra.head(needed)
    combined = pd.concat([combined, extra], ignore_index=True)
    print(f"  Topped up to : {len(combined):,} rows")

# Final type enforcement
for icol in ['Area', 'Max Temp', 'Min Relative Humidity',
             'Max Relative Humidity', 'Production']:
    combined[icol] = pd.to_numeric(combined[icol], errors='coerce').fillna(0).round().astype(int)

# Ensure column order matches original
combined = combined[df.columns]

print(f"\n  ✅ Final shape : {combined.shape}")

# ─────────────────────────────────────────────
# STEP 6: Save
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Saving augmented_dataset.csv")
print("=" * 60)

import os
os.makedirs('Augmentation/Data', exist_ok=True)
combined.to_csv('Augmentation/Data/augmented_dataset.csv', index=False)
print(f"  ✅ Saved: Augmentation/Data/augmented_dataset.csv  ({len(combined):,} rows)")

# ─────────────────────────────────────────────
# STEP 7: Quality Check
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Quality Verification")
print("=" * 60)

orig_stats = df[NUM_COLS].describe()
aug_stats  = combined[NUM_COLS].describe()

print("\n  Mean comparison (Original vs Augmented):")
print(f"  {'Column':<30} {'Orig Mean':>12} {'Aug Mean':>12} {'Drift%':>8}")
print("  " + "-" * 68)
for col in NUM_COLS:
    om = orig_stats.loc['mean', col]
    am = aug_stats.loc['mean', col]
    drift = abs(am - om) / (abs(om) + 1e-9) * 100
    flag = " ⚠️" if drift > 15 else ""
    print(f"  {col:<30} {om:>12.2f} {am:>12.2f} {drift:>7.1f}%{flag}")

print(f"\n  Crop distribution preserved: {combined['Crop Name'].nunique()} crops")
print(f"  District coverage           : {combined['District'].nunique()} districts")
print(f"  Season coverage             : {combined['Season'].nunique()} seasons")

# ─────────────────────────────────────────────
# STEP 8: Visualization
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Generating Augmentation Report Chart")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle(
    f'Data Augmentation Report\n'
    f'Original: {ORIG_LEN:,} rows  →  Augmented: {len(combined):,} rows  |  '
    f'Team: Light Seekers | CSE-4889',
    fontsize=13, fontweight='bold'
)

# ── 1: Row count per method ──
ax = axes[0, 0]
methods = ['Original', 'Noise ×5', 'KNN Interp ×3', 'Scale ×3', 'Final']
counts  = [ORIG_LEN,
           len(method_a),
           len(method_b),
           len(method_c),
           len(combined)]
colors_m = ['#95a5a6', '#2ecc71', '#3498db', '#e67e22', '#e74c3c']
bars = ax.bar(methods, counts, color=colors_m, edgecolor='white', linewidth=1.5)
for b, v in zip(bars, counts):
    ax.text(b.get_x() + b.get_width()/2, v + 200,
            f'{v:,}', ha='center', fontsize=9, fontweight='bold')
ax.set_title('Row Count by Method', fontweight='bold')
ax.set_ylabel('Rows')
ax.grid(axis='y', alpha=0.3)
ax.tick_params(axis='x', rotation=15)

# ── 2: Production distribution ──
ax = axes[0, 1]
# Remove extreme outliers for visibility
orig_prod = df['Production'].clip(0, df['Production'].quantile(0.99))
aug_prod  = combined['Production'].clip(0, combined['Production'].quantile(0.99))
ax.hist(orig_prod, bins=50, alpha=0.6, color='#3498db', label=f'Original ({ORIG_LEN:,})', density=True)
ax.hist(aug_prod,  bins=50, alpha=0.6, color='#e74c3c', label=f'Augmented ({len(combined):,})', density=True)
ax.set_title('Production Distribution\n(Original vs Augmented)', fontweight='bold')
ax.set_xlabel('Production'); ax.set_ylabel('Density')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# ── 3: Rainfall distribution ──
ax = axes[0, 2]
ax.hist(df['Rainfall'],       bins=40, alpha=0.6, color='#3498db', label='Original', density=True)
ax.hist(combined['Rainfall'], bins=40, alpha=0.6, color='#e74c3c', label='Augmented', density=True)
ax.set_title('Rainfall Distribution', fontweight='bold')
ax.set_xlabel('Rainfall (mm)'); ax.set_ylabel('Density')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# ── 4: Crop balance ──
ax = axes[1, 0]
orig_crop = df['Crop Name'].value_counts().sort_index()
aug_crop  = combined['Crop Name'].value_counts().sort_index()
x = np.arange(len(orig_crop))
w = 0.4
ax.bar(x - w/2, orig_crop.values, width=w, color='#3498db', label='Original', alpha=0.8)
ax.bar(x + w/2, aug_crop.values,  width=w, color='#e74c3c', label='Augmented', alpha=0.8)
ax.set_title('Crop Distribution (all crops)', fontweight='bold')
ax.set_ylabel('Row Count')
ax.set_xticks([]); ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# ── 5: Soil features scatter (N vs P) ──
ax = axes[1, 1]
sample_orig = df.sample(min(500, ORIG_LEN), random_state=42)
sample_aug  = combined.sample(min(1500, len(combined)), random_state=42)
ax.scatter(sample_orig['N'], sample_orig['P'], alpha=0.5,
           color='#3498db', s=15, label='Original', zorder=3)
ax.scatter(sample_aug['N'],  sample_aug['P'],  alpha=0.3,
           color='#e74c3c', s=8,  label='Augmented')
ax.set_title('Soil Features: N vs P\n(sample)', fontweight='bold')
ax.set_xlabel('N (Nitrogen)'); ax.set_ylabel('P (Phosphorus)')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# ── 6: Mean drift bar chart ──
ax = axes[1, 2]
check_cols = ['N', 'P', 'K', 'ph', 'Avg Temp', 'Avg Humidity', 'Rainfall']
drifts = []
for col in check_cols:
    om = df[col].mean()
    am = combined[col].mean()
    drifts.append(abs(am - om) / (abs(om) + 1e-9) * 100)
bar_c = ['#2ecc71' if d < 5 else '#f39c12' if d < 10 else '#e74c3c' for d in drifts]
bars = ax.bar(check_cols, drifts, color=bar_c, edgecolor='white')
ax.axhline(5, color='orange', linestyle='--', linewidth=1, label='5% threshold')
ax.axhline(10, color='red',   linestyle='--', linewidth=1, label='10% threshold')
ax.set_title('Mean Drift % After Augmentation\n(lower = better quality)', fontweight='bold')
ax.set_ylabel('Drift %'); ax.tick_params(axis='x', rotation=30)
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
for b, v in zip(bars, drifts):
    ax.text(b.get_x() + b.get_width()/2, v + 0.1,
            f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('Augmentation/Data/augmentation_report.png', dpi=150, bbox_inches='tight')
print("  ✅ Saved: Augmentation/Data/augmentation_report.png")

# ─────────────────────────────────────────────
# Final Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("AUGMENTATION COMPLETE ✅")
print("=" * 60)
print(f"  Original rows   : {ORIG_LEN:,}")
print(f"  Augmented rows  : {len(combined):,}")
print(f"  Growth factor   : {len(combined)/ORIG_LEN:.1f}×")
print(f"\n  Method breakdown:")
print(f"    A — Gaussian Noise (×5) : {len(method_a):,} rows")
print(f"    B — KNN Interpolation   : {len(method_b):,} rows")
print(f"    C — Scale Variation (×3): {len(method_c):,} rows")
print(f"\n  Saved to: Data/Marge/augmented_dataset.csv")
print(f"  Chart  : Data/Marge/augmentation_report.png")
print("\n  এখন Preprocesse-data.py তে")
print("  merged_dataset.csv → augmented_dataset.csv করে দাও!")
print("=" * 60)
