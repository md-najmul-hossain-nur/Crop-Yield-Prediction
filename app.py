"""
=============================================================
  FLASK WEB APPLICATION — Crop Yield Prediction
  Team: Light Seekers | Course: CSE-4889
  United International University, Bangladesh

  Routes:
    /           → Home / Farmer Dashboard
    /predict    → POST: Crop Recommendation (Task B)
    /yield      → POST: Yield Prediction (Task B)
    /compare    → Model Comparison Dashboard
    /api/predict → JSON API endpoint
=============================================================
"""

from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Lazy model loading — train on first request
# ─────────────────────────────────────────────
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score

app = Flask(__name__)

# ─── Global State ────────────────────────────
_models_ready = False
_rf_reg = None
_gb_reg = None
_rf_cls = None
_gb_cls = None
_scaler_reg = None
_scaler_cls = None
_le_season   = None
_le_district = None
_le_crop     = None
_reg_results = {}
_cls_results = {}
_df_cleaned  = None

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

ALL_DISTRICTS = [
    'Bagerhat','Bandarban','Barguna','Barishal','Bhola','Bogura','Brahmanbaria',
    'Chandpur','Chapai Nawabganj','Chattogram','Chuadanga','CoxsBazar','Cumilla',
    'Dhaka','Dinajpur','Faridpur','Feni','Gaibandha','Gazipur','Gopalganj',
    'Habiganj','Jamalpur','Jashore','Jhallokati','Jhenaidah','Joypurhat',
    'Khagrachari','Khulna','Kishoreganj','Kurigram','Kushtia','Lakshmipur',
    'Lalmonirhat','Madaripur','Magura','Manikganj','Meherpur','Moulvibazar',
    'Munshiganj','Mymensingh','Naogaon','Narail','Narayanganj','Narsingdi',
    'Natore','Netrokona','Nilphamari','Noakhali','Pabna','Panchagar',
    'Patuakhali','Pirojpur','Rajbari','Rajshahi','Rangamati','Rangpur',
    'Satkhira','Shariatpur','Sherpur','Sirajganj','Sunamganj','Sylhet',
    'Tangail','Thakurgaon'
]

ALL_CROPS = [
    'Aman','Amra','Arhar','Aus','Banana','Barbati','Beans','Betelnut',
    'Black Berry','Boro','Boroi','Cabbage','Carrot','Cauliflower','Chalkumra',
    'Cheena','Chili','Cucumber','Dalim','Danta','Danta Shak','Date Palm',
    'Garlic','Ginger','Gourd','Gram','Green Coconut','Green Palmyra',
    'Green Papaya','Groundnut','Guava','Jack Fruit','Jambura','Jamrul',
    'Jhinga','Jute','Kakrol','Karala','Kolmi Shak',"Lady's Finger",
    'Lal Shak','Laushak','Lemon','Lentil','Maize 1','Maize 2','Malta',
    'Mango','Mashkalai','Motor','Mug','Mukhi Kachu','Oal Kachu','Onion',
    'Palmyra Palm','Palong Shak','Patal','Pineapple','Puishak','Pumpkin',
    'Radish','Rape & Mustard','Ripe Papaya','Safeda','Sesame','Shalgom',
    'Sugarcane','Sweet Potato','Taramind','Tobacco','Wheat','Wood Apple'
]

ALL_SEASONS = ['Kharif 1', 'Kharif 2', 'Rabi']


# ─────────────────────────────────────────────
# Model Training Function
# ─────────────────────────────────────────────
def train_models():
    global _models_ready, _rf_reg, _gb_reg, _rf_cls, _gb_cls
    global _scaler_reg, _scaler_cls, _le_season, _le_district, _le_crop
    global _reg_results, _cls_results, _df_cleaned

    print("=" * 60)
    print("মডেল ট্রেনিং শুরু হচ্ছে... অপেক্ষা করুন")
    print("=" * 60)

    # ── Load & Clean Data ──
    data_path = 'Data/Marge/merged_dataset.csv'
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), data_path)

    df = pd.read_csv(data_path)
    df.drop(columns=['Transplant', 'Growth', 'Harvest', 'AP Ratio'], inplace=True)
    df = df[df['Production'] > 0]

    Q1 = df['Production'].quantile(0.25)
    Q3 = df['Production'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['Production'] >= Q1 - 3*IQR) & (df['Production'] <= Q3 + 3*IQR)]

    # ── Encode ──
    _le_season   = LabelEncoder()
    _le_district = LabelEncoder()
    _le_crop     = LabelEncoder()

    df['Season_enc']   = _le_season.fit_transform(df['Season'])
    df['District_enc'] = _le_district.fit_transform(df['District'])
    df['Crop_enc']     = _le_crop.fit_transform(df['Crop Name'])
    df['Production_log'] = np.log1p(df['Production'])

    _df_cleaned = df.copy()

    # ── Regression ──
    X_reg = df[REG_FEATURES]
    y_reg = df['Production_log']
    _scaler_reg = StandardScaler()
    X_reg_scaled = pd.DataFrame(_scaler_reg.fit_transform(X_reg), columns=REG_FEATURES)
    X_tr, X_te, y_tr, y_te = train_test_split(X_reg_scaled, y_reg, test_size=0.2, random_state=42)

    # ── Classification ──
    X_cls = df[CLS_FEATURES]
    y_cls = df['Crop_enc']
    _scaler_cls = StandardScaler()
    X_cls_scaled = pd.DataFrame(_scaler_cls.fit_transform(X_cls), columns=CLS_FEATURES)
    X_trc, X_tec, y_trc, y_tec = train_test_split(X_cls_scaled, y_cls, test_size=0.2, random_state=42)

    # ── Train RF Regressor ──
    print("[1/4] Random Forest Regressor ট্রেন হচ্ছে...")
    _rf_reg = RandomForestRegressor(n_estimators=150, max_depth=20, min_samples_leaf=2, n_jobs=-1, random_state=42)
    _rf_reg.fit(X_tr, y_tr)
    pred = _rf_reg.predict(X_te)
    r2 = r2_score(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    mae  = mean_absolute_error(y_te, pred)
    _reg_results['Random Forest'] = dict(r2=r2, rmse=rmse, mae=mae)
    print(f"   ✅ R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    # ── Train GB Regressor ──
    print("[2/4] Gradient Boosting Regressor ট্রেন হচ্ছে...")
    _gb_reg = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=6, random_state=42)
    _gb_reg.fit(X_tr, y_tr)
    pred = _gb_reg.predict(X_te)
    r2 = r2_score(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    mae  = mean_absolute_error(y_te, pred)
    _reg_results['Gradient Boosting'] = dict(r2=r2, rmse=rmse, mae=mae)
    print(f"   ✅ R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    # ── Train RF Classifier ──
    print("[3/4] Random Forest Classifier ট্রেন হচ্ছে...")
    _rf_cls = RandomForestClassifier(n_estimators=150, max_depth=20, min_samples_leaf=2,
                                      class_weight='balanced', n_jobs=-1, random_state=42)
    _rf_cls.fit(X_trc, y_trc)
    pred = _rf_cls.predict(X_tec)
    acc = accuracy_score(y_tec, pred)
    f1  = f1_score(y_tec, pred, average='weighted', zero_division=0)
    _cls_results['Random Forest'] = dict(acc=acc, f1=f1)
    print(f"   ✅ Accuracy={acc:.4f}  F1={f1:.4f}")

    # ── Train GB Classifier ──
    print("[4/4] Gradient Boosting Classifier ট্রেন হচ্ছে...")
    _gb_cls = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=6, random_state=42)
    _gb_cls.fit(X_trc, y_trc)
    pred = _gb_cls.predict(X_tec)
    acc = accuracy_score(y_tec, pred)
    f1  = f1_score(y_tec, pred, average='weighted', zero_division=0)
    _cls_results['Gradient Boosting'] = dict(acc=acc, f1=f1)
    print(f"   ✅ Accuracy={acc:.4f}  F1={f1:.4f}")

    _models_ready = True
    print("\n✅ সব মডেল ট্রেনিং সম্পন্ন! ওয়েব অ্যাপ রেডি।")
    print("=" * 60)


# ─────────────────────────────────────────────
# Prediction Helper
# ─────────────────────────────────────────────
def predict_crop(N, P, K, ph, avg_temp, avg_humidity, rainfall, season, district):
    season_enc   = int(_le_season.transform([season])[0])
    district_enc = int(_le_district.transform([district])[0])

    features = pd.DataFrame([[N, P, K, ph, avg_temp, avg_humidity, rainfall, season_enc, district_enc]],
                             columns=CLS_FEATURES)
    scaled = _scaler_cls.transform(features)

    # Use GB classifier (best)
    crop_enc = _gb_cls.predict(scaled)[0]
    proba    = _gb_cls.predict_proba(scaled)[0]
    top3_idx = np.argsort(proba)[::-1][:3]

    crop_name = _le_crop.inverse_transform([crop_enc])[0]
    top3 = [(str(_le_crop.inverse_transform([i])[0]), round(float(proba[i])*100, 1)) for i in top3_idx]
    return crop_name, top3


def predict_yield(area, N, P, K, ph, avg_temp, min_temp, max_temp,
                  avg_hum, min_hum, max_hum, rainfall, season, district, crop):
    season_enc   = int(_le_season.transform([season])[0])
    district_enc = int(_le_district.transform([district])[0])
    crop_enc     = int(_le_crop.transform([crop])[0])

    features = pd.DataFrame([[area, N, P, K, ph, avg_temp, min_temp, max_temp,
                               avg_hum, min_hum, max_hum, rainfall,
                               season_enc, district_enc, crop_enc]],
                             columns=REG_FEATURES)
    scaled = _scaler_reg.transform(features)

    # Use GB regressor (best)
    pred_log = _gb_reg.predict(scaled)[0]
    pred_actual = np.expm1(pred_log)
    return round(float(pred_actual), 0)


# ─────────────────────────────────────────────
# HTML Templates
# ─────────────────────────────────────────────
BASE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Crop Yield Prediction — Light Seekers</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: 'Segoe UI', sans-serif; background: #f0f4f8; color: #2d3748; }

  .navbar {
    background: linear-gradient(135deg, #1a6b3c, #2d8a55);
    padding: 14px 30px;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
  }
  .navbar .logo { color: #fff; font-size: 1.3rem; font-weight: 700; }
  .navbar .logo span { color: #a8e6c1; }
  .navbar nav a {
    color: #cce8d8; text-decoration: none; margin-left: 20px;
    font-size: 0.95rem; transition: color 0.2s;
  }
  .navbar nav a:hover, .navbar nav a.active { color: #fff; font-weight: 600; }

  .hero {
    background: linear-gradient(135deg, #1a6b3c 0%, #2d8a55 50%, #3da368 100%);
    color: white; padding: 50px 30px; text-align: center;
  }
  .hero h1 { font-size: 2.2rem; margin-bottom: 10px; }
  .hero p  { font-size: 1.1rem; opacity: 0.9; }

  .container { max-width: 1100px; margin: 0 auto; padding: 30px 20px; }

  .card {
    background: white; border-radius: 12px; padding: 28px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08); margin-bottom: 24px;
  }
  .card h2 { font-size: 1.3rem; color: #1a6b3c; margin-bottom: 20px;
             padding-bottom: 10px; border-bottom: 2px solid #e8f5ee; }

  .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }
  .form-group { display: flex; flex-direction: column; }
  .form-group label { font-size: 0.82rem; font-weight: 600; color: #555; margin-bottom: 5px; }
  .form-group input, .form-group select {
    padding: 9px 12px; border: 1.5px solid #d0e8da; border-radius: 7px;
    font-size: 0.95rem; transition: border 0.2s;
    background: #fafffe;
  }
  .form-group input:focus, .form-group select:focus {
    outline: none; border-color: #2d8a55; background: #fff;
  }

  .btn {
    padding: 11px 28px; border: none; border-radius: 8px; font-size: 1rem;
    font-weight: 600; cursor: pointer; transition: all 0.2s; margin-top: 10px;
  }
  .btn-primary { background: #1a6b3c; color: white; }
  .btn-primary:hover { background: #145530; transform: translateY(-1px); }
  .btn-secondary { background: #3498db; color: white; }
  .btn-secondary:hover { background: #2980b9; transform: translateY(-1px); }

  .result-box {
    background: linear-gradient(135deg, #e8f5ee, #d4eddd);
    border: 2px solid #2d8a55; border-radius: 10px;
    padding: 20px; margin-top: 20px; display: none;
  }
  .result-box.error { background: #fef2f2; border-color: #e74c3c; }
  .result-box h3 { color: #1a6b3c; font-size: 1.2rem; margin-bottom: 8px; }
  .result-box .main-result { font-size: 1.6rem; font-weight: 700; color: #145530; }
  .result-box .sub { font-size: 0.9rem; color: #555; margin-top: 6px; }
  .top3 { margin-top: 12px; }
  .top3 .item {
    display: flex; align-items: center; gap: 10px; margin-bottom: 6px;
  }
  .top3 .bar-wrap { flex: 1; background: #c8e6d4; border-radius: 4px; height: 18px; }
  .top3 .bar { background: #2d8a55; height: 18px; border-radius: 4px; transition: width 0.5s; }
  .top3 .label { width: 130px; font-size: 0.85rem; font-weight: 600; }
  .top3 .pct { width: 45px; text-align:right; font-size: 0.85rem; color: #1a6b3c; font-weight:700; }

  /* Compare page */
  .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .metric-card {
    background: linear-gradient(135deg, #f8fffb, #eef7f2);
    border: 1.5px solid #b8ddc8; border-radius: 10px; padding: 18px; text-align: center;
  }
  .metric-card .title { font-size: 0.8rem; color: #666; font-weight: 600; text-transform: uppercase; }
  .metric-card .value { font-size: 2rem; font-weight: 700; color: #1a6b3c; margin: 6px 0; }
  .metric-card .model-name { font-size: 0.8rem; color: #888; }

  .table-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  th { background: #1a6b3c; color: white; padding: 10px 14px; text-align: left; }
  td { padding: 9px 14px; border-bottom: 1px solid #e8f0eb; }
  tr:hover td { background: #f4fdf7; }
  .badge { padding: 3px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 700; }
  .badge-best { background: #d4eddd; color: #1a6b3c; }

  .loading {
    display: none; text-align: center; padding: 20px;
    font-size: 1rem; color: #2d8a55; font-weight: 600;
  }
  .spinner {
    display: inline-block; width: 22px; height: 22px;
    border: 3px solid #c8e6d4; border-top-color: #2d8a55;
    border-radius: 50%; animation: spin 0.8s linear infinite;
    vertical-align: middle; margin-right: 8px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .tabs { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
  .tab {
    padding: 9px 20px; border-radius: 8px; border: 2px solid #2d8a55;
    font-weight: 600; cursor: pointer; font-size: 0.92rem;
    background: white; color: #2d8a55; transition: all 0.2s;
  }
  .tab.active, .tab:hover { background: #2d8a55; color: white; }

  .tab-content { display: none; }
  .tab-content.active { display: block; }

  footer {
    text-align: center; padding: 20px;
    color: #888; font-size: 0.82rem;
    border-top: 1px solid #e0e0e0; margin-top: 20px;
  }
</style>
</head>
<body>

<div class="navbar">
  <div class="logo">🌾 Crop<span>AI</span> Bangladesh</div>
  <nav>
    <a href="/" class="{{ 'active' if page == 'home' else '' }}">🏠 Dashboard</a>
    <a href="/compare" class="{{ 'active' if page == 'compare' else '' }}">📊 Model Comparison</a>
  </nav>
</div>

{% block content %}{% endblock %}

<footer>
  Developed by Team Light Seekers &nbsp;|&nbsp; CSE-4889 (Section-D) &nbsp;|&nbsp;
  United International University, Bangladesh &nbsp;|&nbsp; 2026
</footer>
</body>
</html>
"""

HOME_HTML = BASE_HTML.replace("{% block content %}{% endblock %}", """
<div class="hero">
  <h1>🌾 Crop Yield Prediction & Recommendation</h1>
  <p>AI-powered system for Bangladesh agriculture — soil, weather & season analysis</p>
</div>

<div class="container">

  <div class="tabs">
    <div class="tab active" onclick="switchTab('recommend')">🌱 Crop Recommendation</div>
    <div class="tab" onclick="switchTab('yield')">📦 Yield Prediction</div>
  </div>

  <!-- TAB 1: Crop Recommendation -->
  <div class="tab-content active" id="tab-recommend">
    <div class="card">
      <h2>🌱 Crop Recommendation (Task B — Classification)</h2>
      <p style="font-size:0.88rem; color:#666; margin-bottom:18px;">
        Enter soil nutrients, weather data & location to get AI-powered crop recommendation.
        <strong>Best model: Gradient Boosting (F1 = 0.8538)</strong>
      </p>
      <div class="form-grid">
        <div class="form-group">
          <label>District</label>
          <select id="r_district">
            {% for d in districts %}<option>{{ d }}</option>{% endfor %}
          </select>
        </div>
        <div class="form-group">
          <label>Season</label>
          <select id="r_season">
            {% for s in seasons %}<option>{{ s }}</option>{% endfor %}
          </select>
        </div>
        <div class="form-group">
          <label>Nitrogen — N (mg/kg)</label>
          <input type="number" id="r_N" value="60" step="0.1">
        </div>
        <div class="form-group">
          <label>Phosphorus — P (mg/kg)</label>
          <input type="number" id="r_P" value="42" step="0.1">
        </div>
        <div class="form-group">
          <label>Potassium — K (mg/kg)</label>
          <input type="number" id="r_K" value="33" step="0.1">
        </div>
        <div class="form-group">
          <label>Soil pH</label>
          <input type="number" id="r_ph" value="6.3" step="0.01" min="0" max="14">
        </div>
        <div class="form-group">
          <label>Avg Temperature (°C)</label>
          <input type="number" id="r_temp" value="24" step="0.1">
        </div>
        <div class="form-group">
          <label>Avg Humidity (%)</label>
          <input type="number" id="r_humidity" value="72" step="0.1">
        </div>
        <div class="form-group">
          <label>Annual Rainfall (mm)</label>
          <input type="number" id="r_rainfall" value="178" step="0.1">
        </div>
      </div>
      <button class="btn btn-primary" onclick="getCropRecommendation()">🔍 Recommend Crop</button>

      <div class="loading" id="r_loading">
        <span class="spinner"></span> AI বিশ্লেষণ করছে...
      </div>

      <div class="result-box" id="r_result">
        <h3>✅ প্রস্তাবিত ফসল</h3>
        <div class="main-result" id="r_crop_name">—</div>
        <div class="sub">Top 3 সম্ভাব্য ফসল:</div>
        <div class="top3" id="r_top3"></div>
      </div>
    </div>
  </div>

  <!-- TAB 2: Yield Prediction -->
  <div class="tab-content" id="tab-yield">
    <div class="card">
      <h2>📦 Yield Prediction (Task A — Regression)</h2>
      <p style="font-size:0.88rem; color:#666; margin-bottom:18px;">
        Enter all features to predict expected crop production in metric tons.
        <strong>Best model: Gradient Boosting (R² = 0.9110)</strong>
      </p>
      <div class="form-grid">
        <div class="form-group">
          <label>District</label>
          <select id="y_district">
            {% for d in districts %}<option>{{ d }}</option>{% endfor %}
          </select>
        </div>
        <div class="form-group">
          <label>Season</label>
          <select id="y_season">
            {% for s in seasons %}<option>{{ s }}</option>{% endfor %}
          </select>
        </div>
        <div class="form-group">
          <label>Crop</label>
          <select id="y_crop">
            {% for c in crops %}<option>{{ c }}</option>{% endfor %}
          </select>
        </div>
        <div class="form-group">
          <label>Cultivation Area (hectares)</label>
          <input type="number" id="y_area" value="500" step="1">
        </div>
        <div class="form-group">
          <label>Nitrogen — N (mg/kg)</label>
          <input type="number" id="y_N" value="60" step="0.1">
        </div>
        <div class="form-group">
          <label>Phosphorus — P (mg/kg)</label>
          <input type="number" id="y_P" value="42" step="0.1">
        </div>
        <div class="form-group">
          <label>Potassium — K (mg/kg)</label>
          <input type="number" id="y_K" value="33" step="0.1">
        </div>
        <div class="form-group">
          <label>Soil pH</label>
          <input type="number" id="y_ph" value="6.3" step="0.01">
        </div>
        <div class="form-group">
          <label>Avg Temperature (°C)</label>
          <input type="number" id="y_avg_temp" value="24" step="0.1">
        </div>
        <div class="form-group">
          <label>Min Temperature (°C)</label>
          <input type="number" id="y_min_temp" value="17" step="0.1">
        </div>
        <div class="form-group">
          <label>Max Temperature (°C)</label>
          <input type="number" id="y_max_temp" value="30" step="0.1">
        </div>
        <div class="form-group">
          <label>Avg Humidity (%)</label>
          <input type="number" id="y_avg_hum" value="72" step="0.1">
        </div>
        <div class="form-group">
          <label>Min Relative Humidity (%)</label>
          <input type="number" id="y_min_hum" value="62" step="0.1">
        </div>
        <div class="form-group">
          <label>Max Relative Humidity (%)</label>
          <input type="number" id="y_max_hum" value="82" step="0.1">
        </div>
        <div class="form-group">
          <label>Annual Rainfall (mm)</label>
          <input type="number" id="y_rainfall" value="178" step="0.1">
        </div>
      </div>
      <button class="btn btn-secondary" onclick="getYieldPrediction()">📈 Predict Yield</button>

      <div class="loading" id="y_loading">
        <span class="spinner"></span> উৎপাদন হিসাব করা হচ্ছে...
      </div>

      <div class="result-box" id="y_result">
        <h3>📦 প্রত্যাশিত উৎপাদন</h3>
        <div class="main-result" id="y_yield_val">—</div>
        <div class="sub" id="y_sub">metric tons (approximate)</div>
      </div>
    </div>
  </div>

</div>

<script>
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}

async function getCropRecommendation() {
  const payload = {
    N: parseFloat(document.getElementById('r_N').value),
    P: parseFloat(document.getElementById('r_P').value),
    K: parseFloat(document.getElementById('r_K').value),
    ph: parseFloat(document.getElementById('r_ph').value),
    avg_temp: parseFloat(document.getElementById('r_temp').value),
    avg_humidity: parseFloat(document.getElementById('r_humidity').value),
    rainfall: parseFloat(document.getElementById('r_rainfall').value),
    season: document.getElementById('r_season').value,
    district: document.getElementById('r_district').value,
    task: 'classify'
  };
  document.getElementById('r_loading').style.display = 'block';
  document.getElementById('r_result').style.display = 'none';

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    document.getElementById('r_loading').style.display = 'none';

    if (data.error) {
      const box = document.getElementById('r_result');
      box.classList.add('error');
      box.innerHTML = '<h3>❌ Error</h3><p>' + data.error + '</p>';
      box.style.display = 'block';
      return;
    }

    document.getElementById('r_crop_name').textContent = '🌿 ' + data.crop;
    const top3Html = data.top3.map(([name, pct]) =>
      `<div class="item">
        <div class="label">${name}</div>
        <div class="bar-wrap"><div class="bar" style="width:${Math.min(pct, 100)}%"></div></div>
        <div class="pct">${pct}%</div>
       </div>`
    ).join('');
    document.getElementById('r_top3').innerHTML = top3Html;
    document.getElementById('r_result').classList.remove('error');
    document.getElementById('r_result').style.display = 'block';
  } catch(e) {
    document.getElementById('r_loading').style.display = 'none';
    alert('সার্ভার এরর: ' + e.message);
  }
}

async function getYieldPrediction() {
  const payload = {
    area: parseFloat(document.getElementById('y_area').value),
    N: parseFloat(document.getElementById('y_N').value),
    P: parseFloat(document.getElementById('y_P').value),
    K: parseFloat(document.getElementById('y_K').value),
    ph: parseFloat(document.getElementById('y_ph').value),
    avg_temp: parseFloat(document.getElementById('y_avg_temp').value),
    min_temp: parseFloat(document.getElementById('y_min_temp').value),
    max_temp: parseFloat(document.getElementById('y_max_temp').value),
    avg_humidity: parseFloat(document.getElementById('y_avg_hum').value),
    min_humidity: parseFloat(document.getElementById('y_min_hum').value),
    max_humidity: parseFloat(document.getElementById('y_max_hum').value),
    rainfall: parseFloat(document.getElementById('y_rainfall').value),
    season: document.getElementById('y_season').value,
    district: document.getElementById('y_district').value,
    crop: document.getElementById('y_crop').value,
    task: 'regress'
  };
  document.getElementById('y_loading').style.display = 'block';
  document.getElementById('y_result').style.display = 'none';

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    document.getElementById('y_loading').style.display = 'none';

    if (data.error) {
      const box = document.getElementById('y_result');
      box.classList.add('error');
      box.innerHTML = '<h3>❌ Error</h3><p>' + data.error + '</p>';
      box.style.display = 'block';
      return;
    }

    document.getElementById('y_yield_val').textContent =
      '🌾 ' + Number(data.yield_tons).toLocaleString() + ' metric tons';
    document.getElementById('y_sub').textContent =
      'Predicted for ' + payload.crop + ' in ' + payload.district + ' (' + payload.season + ')';
    document.getElementById('y_result').classList.remove('error');
    document.getElementById('y_result').style.display = 'block';
  } catch(e) {
    document.getElementById('y_loading').style.display = 'none';
    alert('সার্ভার এরর: ' + e.message);
  }
}
</script>
""")

COMPARE_HTML = BASE_HTML.replace("{% block content %}{% endblock %}", """
<div class="hero">
  <h1>📊 Model Comparison Dashboard</h1>
  <p>Random Forest vs Gradient Boosting — Performance Metrics</p>
</div>

<div class="container">

  <div class="card">
    <h2>🏆 Best Model Summary</h2>
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="title">Best Regression R²</div>
        <div class="value">{{ reg_best_r2 }}</div>
        <div class="model-name">{{ reg_best_model }}</div>
      </div>
      <div class="metric-card">
        <div class="title">Best Regression RMSE</div>
        <div class="value">{{ reg_best_rmse }}</div>
        <div class="model-name">{{ reg_best_model }}</div>
      </div>
      <div class="metric-card">
        <div class="title">Best Classification Acc</div>
        <div class="value">{{ cls_best_acc }}%</div>
        <div class="model-name">{{ cls_best_model }}</div>
      </div>
      <div class="metric-card">
        <div class="title">Best F1 Score</div>
        <div class="value">{{ cls_best_f1 }}</div>
        <div class="model-name">{{ cls_best_model }}</div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>📈 Task A — Yield Prediction (Regression)</h2>
    <div class="table-wrap">
      <table>
        <tr><th>Model</th><th>Test R²</th><th>RMSE</th><th>MAE</th><th>Status</th></tr>
        {% for row in reg_rows %}
        <tr>
          <td><strong>{{ row.model }}</strong></td>
          <td>{{ row.r2 }}</td>
          <td>{{ row.rmse }}</td>
          <td>{{ row.mae }}</td>
          <td>{% if row.best %}<span class="badge badge-best">✅ BEST</span>{% endif %}</td>
        </tr>
        {% endfor %}
      </table>
    </div>
  </div>

  <div class="card">
    <h2>🌿 Task B — Crop Recommendation (Classification)</h2>
    <div class="table-wrap">
      <table>
        <tr><th>Model</th><th>Accuracy</th><th>F1 Score (Weighted)</th><th>Status</th></tr>
        {% for row in cls_rows %}
        <tr>
          <td><strong>{{ row.model }}</strong></td>
          <td>{{ row.acc }}%</td>
          <td>{{ row.f1 }}</td>
          <td>{% if row.best %}<span class="badge badge-best">✅ BEST</span>{% endif %}</td>
        </tr>
        {% endfor %}
      </table>
    </div>
  </div>

  <div class="card">
    <h2>📋 Project Info</h2>
    <table>
      <tr><th>Item</th><th>Details</th></tr>
      <tr><td>Team</td><td>Light Seekers</td></tr>
      <tr><td>Course</td><td>CSE-4889 (Section-D) — United International University</td></tr>
      <tr><td>Dataset</td><td>SPAS-BD + Crop Recommendation + 65 Years Bangladesh Weather</td></tr>
      <tr><td>Merged Dataset</td><td>4,607 rows × 20 features</td></tr>
      <tr><td>Task A Features</td><td>15 (Area, N, P, K, ph, Temp, Humidity, Rainfall, Season, District, Crop)</td></tr>
      <tr><td>Task B Features</td><td>9 (N, P, K, ph, Avg Temp, Avg Humidity, Rainfall, Season, District)</td></tr>
      <tr><td>Classes (Task B)</td><td>72 crops across 64 Bangladesh districts</td></tr>
    </table>
  </div>
</div>
""")

LOADING_HTML = BASE_HTML.replace("{% block content %}{% endblock %}", """
<div class="hero">
  <h1>⚙️ মডেল ট্রেনিং চলছে...</h1>
  <p>প্রথমবার লোড হচ্ছে, একটু সময় লাগবে। পেজ রিফ্রেশ করুন।</p>
</div>
<div class="container">
  <div class="card" style="text-align:center; padding: 50px;">
    <div class="spinner" style="width:40px;height:40px;border-width:5px;"></div>
    <p style="margin-top:20px; font-size:1.1rem; color:#2d8a55;">
      Random Forest ও Gradient Boosting মডেল ট্রেনিং চলছে...<br>
      <small style="color:#888;">এই পেজটি 30-60 সেকেন্ড পর রিফ্রেশ করুন</small>
    </p>
    <script>setTimeout(()=>location.reload(), 5000);</script>
  </div>
</div>
""")


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route('/')
def index():
    if not _models_ready:
        return render_template_string(LOADING_HTML, page='home')
    return render_template_string(HOME_HTML, page='home',
                                  districts=ALL_DISTRICTS,
                                  crops=ALL_CROPS,
                                  seasons=ALL_SEASONS)


@app.route('/compare')
def compare():
    if not _models_ready:
        return render_template_string(LOADING_HTML, page='compare')

    best_reg   = max(_reg_results, key=lambda m: _reg_results[m]['r2'])
    best_cls   = max(_cls_results, key=lambda m: _cls_results[m]['f1'])

    reg_rows = [
        dict(model=m,
             r2=f"{v['r2']:.4f}",
             rmse=f"{v['rmse']:.4f}",
             mae=f"{v['mae']:.4f}",
             best=(m == best_reg))
        for m, v in _reg_results.items()
    ]
    cls_rows = [
        dict(model=m,
             acc=f"{v['acc']*100:.2f}",
             f1=f"{v['f1']:.4f}",
             best=(m == best_cls))
        for m, v in _cls_results.items()
    ]

    return render_template_string(COMPARE_HTML, page='compare',
        reg_rows=reg_rows, cls_rows=cls_rows,
        reg_best_model=best_reg,
        reg_best_r2=f"{_reg_results[best_reg]['r2']:.4f}",
        reg_best_rmse=f"{_reg_results[best_reg]['rmse']:.4f}",
        cls_best_model=best_cls,
        cls_best_acc=f"{_cls_results[best_cls]['acc']*100:.2f}",
        cls_best_f1=f"{_cls_results[best_cls]['f1']:.4f}",
    )


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if not _models_ready:
        return jsonify({'error': 'মডেল এখনো রেডি না। একটু অপেক্ষা করুন।'})

    data = request.get_json()
    task = data.get('task', 'classify')

    try:
        if task == 'classify':
            crop, top3 = predict_crop(
                N=data['N'], P=data['P'], K=data['K'], ph=data['ph'],
                avg_temp=data['avg_temp'], avg_humidity=data['avg_humidity'],
                rainfall=data['rainfall'], season=data['season'], district=data['district']
            )
            return jsonify({'crop': crop, 'top3': top3})

        elif task == 'regress':
            yield_tons = predict_yield(
                area=data['area'], N=data['N'], P=data['P'], K=data['K'], ph=data['ph'],
                avg_temp=data['avg_temp'], min_temp=data['min_temp'], max_temp=data['max_temp'],
                avg_hum=data['avg_humidity'], min_hum=data['min_humidity'], max_hum=data['max_humidity'],
                rainfall=data['rainfall'], season=data['season'],
                district=data['district'], crop=data['crop']
            )
            return jsonify({'yield_tons': yield_tons})

    except Exception as e:
        return jsonify({'error': str(e)})

    return jsonify({'error': 'Unknown task'})


# ─────────────────────────────────────────────
# Start
# ─────────────────────────────────────────────
if __name__ == '__main__':
    # Train models before starting server
    train_models()
    print("\n🌐 Flask সার্ভার শুরু হচ্ছে...")
    print("   → http://localhost:5000")
    print("   → Ctrl+C দিয়ে বন্ধ করুন\n")
    app.run(debug=False, host='0.0.0.0', port=5000)