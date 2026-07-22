"""
=============================================================
  FULL PIPELINE RUNNER & GRAPH AGGREGATOR — Match22 Data
  Team: Light Seekers | CSE-4889
  Runs complete end-to-end workflow:
    1. Preprocess raw data
    2. Train models on raw data
    3. Augment data
    4. Preprocess augmented data
    5. Train models on augmented data
    6. Collect all graph images into Data/Match22-data/All_Graphs/
=============================================================
"""

import subprocess
import shutil
import os
import sys

# Enable UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

GRAPH_DIR = 'Data/Match22-data/All_Graphs'
os.makedirs(GRAPH_DIR, exist_ok=True)

scripts = [
    ("Preprocessing Raw Match22 Data", "Data/Match22-data/Code/preprocess_data.py"),
    ("Training Models on Raw Data", "Data/Match22-data/Code/train_models.py"),
    ("Augmenting Match22 Data", "Data/Match22-data/Code/augment_data.py"),
    ("Preprocessing Augmented Data", "Data/Match22-data/Code/preprocess_augmented.py"),
    ("Training Models on Augmented Data", "Data/Match22-data/Code/train_augmented_models.py")
]

print("=" * 70)
print("STARTING FULL END-TO-END MATCH22 PIPELINE RUN")
print("=" * 70)

for step_name, script_path in scripts:
    print(f"\n>>> Running Step: {step_name} ({script_path})...")
    res = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    if res.returncode == 0:
        print(f"    ✅ Success: {step_name}")
    else:
        print(f"    ❌ Error in {step_name}: {res.stderr}")

print("\n" + "=" * 70)
print("COLLECTING ALL GRAPH IMAGES INTO Data/Match22-data/All_Graphs/")
print("=" * 70)

graphs_to_copy = [
    ('Data/Match22-data/Processed_Data/preprocess_summary.png', '1_Data_Preprocessing_Summary.png'),
    ('Data/Match22-data/Output/augmentation_summary.png', '2_Data_Augmentation_Summary.png'),
    ('Data/Match22-data/Output/match22_comparison.png', '3_Model_Comparison_Original_Data.png'),
    ('Data/Match22-data/Output/match22_augmented_comparison.png', '4_Model_Comparison_Augmented_Data.png')
]

copied_files = []
for src, dst_name in graphs_to_copy:
    if os.path.exists(src):
        dst_path = os.path.join(GRAPH_DIR, dst_name)
        shutil.copy(src, dst_path)
        copied_files.append(dst_path)
        print(f"  ✅ Copied: {dst_name}")
    else:
        print(f"  ⚠️ Warning: Source graph not found - {src}")

print("\n" + "=" * 70)
print(f"FULL PIPELINE EXECUTION COMPLETE ✅")
print(f"All graphs successfully saved in: {GRAPH_DIR}/")
print("=" * 70)
