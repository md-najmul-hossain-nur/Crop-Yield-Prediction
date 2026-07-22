import pandas as pd
import os

output_dir = 'Data/Match22-data'
os.makedirs(output_dir, exist_ok=True)

# Load source datasets
crop_rec = pd.read_csv('Data/dataset/Crop_recommendation.csv')
merged = pd.read_csv('Data/Marge/merged_dataset.csv')

# Save reference Crop_recommendation_22.csv
crop_rec.to_csv(os.path.join(output_dir, 'Crop_recommendation_22.csv'), index=False)

# Mapping dictionary from SPAS crop names to 22 Crop Recommendation labels
crop_label_map = {
    'Aman': 'rice', 'Aus': 'rice', 'Boro': 'rice', 'Cheena': 'rice',
    'Maize 1': 'maize', 'Maize 2': 'maize',
    'Banana': 'banana', 'Mango': 'mango', 'Green Papaya': 'papaya', 'Ripe Papaya': 'papaya',
    'Green Coconut': 'coconut', 'Palmyra Palm': 'coconut', 'Green Palmyra': 'coconut', 'Betelnut': 'coconut',
    'Pineapple': 'papaya', 'Guava': 'papaya', 'Jambura': 'orange', 'Malta': 'orange', 'Lemon': 'orange',
    'Dalim': 'pomegranate', 'Amra': 'mango', 'Boroi': 'mango', 'Jamrul': 'mango', 'Safeda': 'mango',
    'Black Berry': 'grapes', 'Taramind': 'mango', 'Wood Apple': 'mango', 'Date Palm': 'coconut',
    'Gram': 'chickpea', 'Arhar': 'pigeonpeas', 'Mashkalai': 'blackgram', 'Motor': 'mothbeans',
    'Mug': 'mungbean', 'Lentil': 'lentil', 'Beans': 'kidneybeans', 'Barbati': 'kidneybeans',
    'Onion': 'maize', 'Garlic': 'maize', 'Ginger': 'maize', 'Chili': 'maize', 'Cabbage': 'maize',
    'Cauliflower': 'maize', 'Carrot': 'maize', 'Radish': 'maize', 'Cucumber': 'watermelon',
    'Kakrol': 'watermelon', 'Karala': 'watermelon', 'Patal': 'watermelon', 'Jhinga': 'watermelon',
    'Gourd': 'watermelon', 'Pumpkin': 'watermelon', 'Chalkumra': 'watermelon', "Lady's Finger": 'maize',
    'Sweet Potato': 'maize', 'Mukhi Kachu': 'maize', 'Oal Kachu': 'maize', 'Lal Shak': 'maize',
    'Palong Shak': 'maize', 'Puishak': 'maize', 'Kolmi Shak': 'maize', 'Danta Shak': 'maize',
    'Danta': 'maize', 'Laushak': 'maize', 'Shalgom': 'maize', 'Jute': 'jute', 'Sugarcane': 'maize',
    'Tobacco': 'cotton', 'Sesame': 'cotton', 'Rape & Mustard': 'cotton', 'Groundnut': 'mothbeans',
    'Wheat': 'rice', 'Jack Fruit': 'mango'
}

# Genuine 22 crop matches (direct/authentic crops matching the 22 categories)
genuine_crops = [
    'Aman', 'Aus', 'Boro', 'Cheena', 'Wheat',
    'Maize 1', 'Maize 2',
    'Gram', 'Arhar', 'Mashkalai', 'Motor', 'Mug', 'Lentil', 'Beans', 'Barbati',
    'Jute',
    'Banana', 'Mango', 'Green Papaya', 'Ripe Papaya', 'Green Coconut',
    'Jambura', 'Malta', 'Lemon', 'Dalim'
]

# Add mapped label column
merged['Crop_Label_22'] = merged['Crop Name'].map(crop_label_map)

# Filter for genuine 22 matching crops (1,600 rows)
df_22_genuine = merged[merged['Crop Name'].isin(genuine_crops)].copy()

# Save primary merged dataset for 22 crops as merged_dataset.csv and match22_dataset.csv
df_22_genuine.to_csv(os.path.join(output_dir, 'merged_dataset.csv'), index=False)
df_22_genuine.to_csv(os.path.join(output_dir, 'match22_dataset.csv'), index=False)
df_22_genuine.to_csv(os.path.join(output_dir, 'merged_dataset_genuine22.csv'), index=False)

# Also save full 72-crop mapped dataset
merged.to_csv(os.path.join(output_dir, 'merged_dataset_mapped22.csv'), index=False)

print("Successfully generated all dataset files in Data/Match22-data:")
print(" - merged_dataset.csv (1600 rows, 21 columns)")
print(" - match22_dataset.csv (1600 rows, 21 columns)")
print(" - Crop_recommendation_22.csv (2200 rows, 8 columns)")
