"""
=============================================================
  DATA MERGE PIPELINE — Crop Yield Prediction
  Team: Light Seekers | Course: CSE-4889
  
  Files needed (same folder এ রাখো):
    1. SPAS-Dataset-BD.csv
    2. 65_Years_of_Weather_Data_Bangladesh__1948_-_2013_.csv
    3. Crop_recommendation.csv
=============================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STEP 0: Load করো তিনটা Dataset
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 0: Loading Datasets")
print("=" * 60)

spas    = pd.read_csv('SPAS-Dataset-BD.csv')
weather = pd.read_csv('65_Years_of_Weather_Data_Bangladesh__1948_-_2013_.csv')
crop    = pd.read_csv('Crop_recommendation.csv')

print(f"✅ SPAS Dataset         : {spas.shape}")
print(f"✅ Weather Dataset      : {weather.shape}")
print(f"✅ Crop Recommend. Data : {crop.shape}")


# ─────────────────────────────────────────────
# STEP 1: SPAS Dataset Clean করো
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: Cleaning SPAS Dataset")
print("=" * 60)

# dirty row বাদ দাও
spas = spas[spas['Crop Name'] != '#REF!'].copy()
print(f"✅ '#REF!' row removed")

# Season এর null fix করো
spas['Season'] = spas['Season'].fillna('Kharif 2')
print(f"✅ Season null → 'Kharif 2'")

# AP Ratio কে numeric বানাও, null হলে median দিয়ে fill করো
spas['AP Ratio'] = pd.to_numeric(spas['AP Ratio'], errors='coerce')
spas['AP Ratio'] = spas['AP Ratio'].fillna(spas['AP Ratio'].median())
print(f"✅ AP Ratio fixed (median fill)")

print(f"   SPAS shape after cleaning: {spas.shape}")


# ─────────────────────────────────────────────
# STEP 2: Weather Dataset থেকে Rainfall নাও
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Extracting Rainfall from Weather Dataset")
print("=" * 60)

# প্রতিটা District কে কাছের Weather Station-এ map করা হয়েছে
district_to_station = {
    # Direct match
    'Barishal'       : 'Barisal',
    'Bhola'          : 'Bhola',
    'Bogura'         : 'Bogra',
    'Chandpur'       : 'Chandpur',
    'Chattogram'     : 'Chittagong (City-Ambagan)',
    'CoxsBazar'      : "Cox's Bazar",
    'Cumilla'        : 'Comilla',
    'Chuadanga'      : 'Chuadanga',
    'Dhaka'          : 'Dhaka',
    'Dinajpur'       : 'Dinajpur',
    'Faridpur'       : 'Faridpur',
    'Feni'           : 'Feni',
    'Khulna'         : 'Khulna',
    'Madaripur'      : 'Madaripur',
    'Mymensingh'     : 'Mymensingh',
    'Patuakhali'     : 'Patuakhali',
    'Rajshahi'       : 'Rajshahi',
    'Rangamati'      : 'Rangamati',
    'Rangpur'        : 'Rangpur',
    'Satkhira'       : 'Satkhira',
    'Sylhet'         : 'Sylhet',
    'Tangail'        : 'Tangail',
    # Nearest station mapping
    'Bagerhat'       : 'Khulna',
    'Barguna'        : 'Patuakhali',
    'Bandarban'      : 'Chittagong (City-Ambagan)',
    'Brahmanbaria'   : 'Comilla',
    'Chapai Nawabganj': 'Rajshahi',
    'Gaibandha'      : 'Rangpur',
    'Gazipur'        : 'Dhaka',
    'Gopalganj'      : 'Faridpur',
    'Habiganj'       : 'Sylhet',
    'Jamalpur'       : 'Mymensingh',
    'Jashore'        : 'Jessore',
    'Jhallokati'     : 'Barisal',
    'Jhenaidah'      : 'Jessore',
    'Joypurhat'      : 'Bogra',
    'Khagrachari'    : 'Rangamati',
    'Kishoreganj'    : 'Mymensingh',
    'Kurigram'       : 'Rangpur',
    'Kushtia'        : 'Chuadanga',
    'Lakshmipur'     : 'Comilla',
    'Lalmonirhat'    : 'Rangpur',
    'Magura'         : 'Jessore',
    'Manikganj'      : 'Dhaka',
    'Meherpur'       : 'Chuadanga',
    'Moulvibazar'    : 'Sylhet',
    'Munshiganj'     : 'Dhaka',
    'Naogaon'        : 'Rajshahi',
    'Narail'         : 'Jessore',
    'Narayanganj'    : 'Dhaka',
    'Narsingdi'      : 'Dhaka',
    'Natore'         : 'Rajshahi',
    'Netrokona'      : 'Mymensingh',
    'Nilphamari'     : 'Rangpur',
    'Noakhali'       : 'Comilla',
    'Pabna'          : 'Rajshahi',
    'Panchagar'      : 'Dinajpur',
    'Pirojpur'       : 'Barisal',
    'Rajbari'        : 'Faridpur',
    'Shariatpur'     : 'Faridpur',
    'Sherpur'        : 'Mymensingh',
    'Sirajganj'      : 'Bogra',
    'Sunamganj'      : 'Sylhet',
    'Thakurgaon'     : 'Dinajpur',
}

# প্রতিটা station-এর গড় বার্ষিক rainfall হিসাব করো
weather_avg = weather.groupby('Station Names')['Rainfall'].mean().reset_index()
weather_avg.columns = ['Station', 'Rainfall']
print(f"✅ Rainfall avg calculated for {len(weather_avg)} stations")

# District → Station → Rainfall
spas['Station'] = spas['District'].map(district_to_station)
spas = spas.merge(weather_avg, on='Station', how='left')
spas = spas.drop(columns=['Station'])

print(f"✅ Rainfall merged into SPAS")
print(f"   Null rainfall: {spas['Rainfall'].isnull().sum()}")


# ─────────────────────────────────────────────
# STEP 3: Crop Recommendation থেকে N, P, K, pH নাও
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Merging N, P, K, pH from Crop Recommendation")
print("=" * 60)

# SPAS crop name → Crop Recommendation label mapping
crop_label_map = {
    # Rice varieties
    'Aman'           : 'rice',
    'Aus'            : 'rice',
    'Boro'           : 'rice',
    'Cheena'         : 'rice',
    # Maize
    'Maize 1'        : 'maize',
    'Maize 2'        : 'maize',
    # Fruits
    'Banana'         : 'banana',
    'Mango'          : 'mango',
    'Green Papaya'   : 'papaya',
    'Ripe Papaya'    : 'papaya',
    'Green Coconut'  : 'coconut',
    'Palmyra Palm'   : 'coconut',
    'Green Palmyra'  : 'coconut',
    'Betelnut'       : 'coconut',
    'Pineapple'      : 'papaya',
    'Guava'          : 'papaya',
    'Jambura'        : 'orange',
    'Malta'          : 'orange',
    'Lemon'          : 'orange',
    'Dalim'          : 'pomegranate',
    'Amra'           : 'mango',
    'Boroi'          : 'mango',
    'Jamrul'         : 'mango',
    'Safeda'         : 'mango',
    'Black Berry'    : 'grapes',
    'Taramind'       : 'mango',
    'Wood Apple'     : 'mango',
    'Date Palm'      : 'coconut',
    # Pulses / Legumes
    'Gram'           : 'chickpea',
    'Arhar'          : 'pigeonpeas',
    'Mashkalai'      : 'blackgram',
    'Motor'          : 'mothbeans',
    'Mug'            : 'mungbean',
    'Lentil'         : 'lentil',
    'Beans'          : 'kidneybeans',
    'Barbati'        : 'kidneybeans',
    # Vegetables
    'Onion'          : 'maize',
    'Garlic'         : 'maize',
    'Ginger'         : 'maize',
    'Chili'          : 'maize',
    'Cabbage'        : 'maize',
    'Cauliflower'    : 'maize',
    'Carrot'         : 'maize',
    'Radish'         : 'maize',
    'Cucumber'       : 'watermelon',
    'Kakrol'         : 'watermelon',
    'Karala'         : 'watermelon',
    'Patal'          : 'watermelon',
    'Jhinga'         : 'watermelon',
    'Gourd'          : 'watermelon',
    'Pumpkin'        : 'watermelon',
    'Chalkumra'      : 'watermelon',
    'Lady\'s Finger' : 'maize',
    'Sweet Potato'   : 'maize',
    'Mukhi Kachu'    : 'maize',
    'Oal Kachu'      : 'maize',
    # Greens
    'Lal Shak'       : 'maize',
    'Palong Shak'    : 'maize',
    'Puishak'        : 'maize',
    'Kolmi Shak'     : 'maize',
    'Danta Shak'     : 'maize',
    'Danta'          : 'maize',
    'Laushak'        : 'maize',
    'Shalgom'        : 'maize',
    # Cash crops
    'Jute'           : 'jute',
    'Sugarcane'      : 'maize',
    'Tobacco'        : 'cotton',
    'Sesame'         : 'cotton',
    'Rape & Mustard' : 'cotton',
    'Groundnut'      : 'mothbeans',
    'Wheat'          : 'rice',
    'Jack Fruit'     : 'mango',
}

# প্রতিটা label-এর জন্য NPK এবং pH-এর গড় বের করো
npk_avg = crop.groupby('label')[['N', 'P', 'K', 'ph']].mean().round(2)
global_npk = npk_avg.mean()  # যেসব crop match হবে না তাদের জন্য global average

def get_npk(crop_name):
    label = crop_label_map.get(crop_name, None)
    if label and label in npk_avg.index:
        return npk_avg.loc[label]
    return global_npk

npk_data = spas['Crop Name'].apply(get_npk)
spas = pd.concat([spas.reset_index(drop=True), npk_data.reset_index(drop=True)], axis=1)

print(f"✅ N, P, K, pH merged successfully")
print(f"   Mapped crops : {sum(1 for c in spas['Crop Name'].unique() if c in crop_label_map)}/{spas['Crop Name'].nunique()}")


# ─────────────────────────────────────────────
# STEP 4: Final Column Order ঠিক করো
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Organizing Final Columns")
print("=" * 60)

final_columns = [
    'District', 'Season', 'Crop Name',
    'Area',
    'N', 'P', 'K', 'ph',
    'Avg Temp', 'Min Temp', 'Max Temp',
    'Avg Humidity', 'Min Relative Humidity', 'Max Relative Humidity',
    'Rainfall',
    'AP Ratio',
    'Transplant', 'Growth', 'Harvest',
    'Production'
]

df_merged = spas[final_columns].copy()

print(f"✅ Final columns: {len(final_columns)}")
print(f"   {final_columns}")


# ─────────────────────────────────────────────
# STEP 5: Quality Check করো
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Quality Check")
print("=" * 60)

print(f"Total rows       : {len(df_merged)}")
print(f"Total columns    : {len(df_merged.columns)}")
print(f"\nNull values per column:")
nulls = df_merged.isnull().sum()
for col, n in nulls.items():
    status = "✅" if n == 0 else "⚠️"
    print(f"  {status} {col}: {n}")

print(f"\nSample data (first 3 rows):")
print(df_merged.head(3).to_string())

print(f"\nData types:")
print(df_merged.dtypes)


# ─────────────────────────────────────────────
# STEP 6: Save করো
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Saving Merged Dataset")
print("=" * 60)

df_merged.to_csv('merged_dataset.csv', index=False)
print(f"✅ Saved: merged_dataset.csv")
print(f"   Shape: {df_merged.shape}")

print("\n" + "=" * 60)
print("✅ DATA MERGE COMPLETE!")
print("=" * 60)
print("পরবর্তী কাজ: preprocessing.py চালাও")
print("=" * 60)
