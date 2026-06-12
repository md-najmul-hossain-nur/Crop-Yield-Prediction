"""
Crop Yield Prediction & Recommendation — Flask Web App
Team: Light Seekers | CSE-4889 | UIU Bangladesh

Run:  python app.py
Open: http://localhost:8080
"""

from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
import pickle, os, warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_PATH = os.path.join(BASE_DIR, 'app_models.pkl')

app = Flask(__name__)

MODEL   = {}
ENCODER = {}
CLASSES = {}

# ─────────────────────────────────────────
# TRAIN OR LOAD
# ─────────────────────────────────────────
def train_and_save():
    print("Training models ...")
    df = pd.read_csv(os.path.join(BASE_DIR, 'Data', 'Marge', 'merged_dataset.csv'))

    for col in ['Transplant', 'Growth', 'Harvest', 'AP Ratio']:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    df = df[df['Production'] > 0].copy()
    Q1, Q3 = df['Production'].quantile(0.25), df['Production'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['Production'] >= Q1 - 3*IQR) & (df['Production'] <= Q3 + 3*IQR)].copy()
    df.reset_index(drop=True, inplace=True)

    le_season   = LabelEncoder()
    le_district = LabelEncoder()
    le_crop     = LabelEncoder()

    df['Season_enc']     = le_season.fit_transform(df['Season'])
    df['District_enc']   = le_district.fit_transform(df['District'])
    df['Crop_enc']       = le_crop.fit_transform(df['Crop Name'])
    df['Production_log'] = np.log1p(df['Production'])

    REG_FEAT = ['Area','N','P','K','ph','Avg Temp','Min Temp','Max Temp',
                'Avg Humidity','Min Relative Humidity','Max Relative Humidity',
                'Rainfall','Season_enc','District_enc','Crop_enc']
    CLS_FEAT = ['N','P','K','ph','Avg Temp','Avg Humidity','Rainfall',
                'Season_enc','District_enc']

    # Regression — Gradient Boosting
    X_r = df[REG_FEAT].values
    y_r = df['Production_log'].values
    sc_r = StandardScaler()
    X_r_sc = sc_r.fit_transform(X_r)
    X_tr, X_te, y_tr, y_te = train_test_split(X_r_sc, y_r, test_size=0.2, random_state=42)
    gb = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=6, random_state=42)
    gb.fit(X_tr, y_tr)
    print("  GB Regressor done")

    # Classification — Random Forest
    X_c = df[CLS_FEAT].values
    y_c = df['Crop_enc'].values
    sc_c = StandardScaler()
    X_c_sc = sc_c.fit_transform(X_c)
    X_trc, _, y_trc, _ = train_test_split(X_c_sc, y_c, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=150, max_depth=20, min_samples_leaf=2,
                                class_weight='balanced', n_jobs=-1, random_state=42)
    rf.fit(X_trc, y_trc)
    print("  RF Classifier done")

    bundle = dict(gb=gb, rf=rf, sc_r=sc_r, sc_c=sc_c,
                  le_season=le_season, le_district=le_district, le_crop=le_crop,
                  reg_feat=REG_FEAT, cls_feat=CLS_FEAT)
    with open(PKL_PATH, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"  Saved: {PKL_PATH}")
    return bundle

def load_models():
    global MODEL, ENCODER, CLASSES
    if os.path.exists(PKL_PATH):
        print("Loading saved models ...")
        with open(PKL_PATH, 'rb') as f:
            b = pickle.load(f)
    else:
        b = train_and_save()

    MODEL['reg'] = b['gb']
    MODEL['cls'] = b['rf']
    ENCODER['sc_r']     = b['sc_r']
    ENCODER['sc_c']     = b['sc_c']
    ENCODER['season']   = b['le_season']
    ENCODER['district'] = b['le_district']
    ENCODER['crop']     = b['le_crop']
    CLASSES['crops']     = sorted(b['le_crop'].classes_.tolist())
    CLASSES['districts'] = sorted(b['le_district'].classes_.tolist())
    CLASSES['seasons']   = b['le_season'].classes_.tolist()
    print("Models ready")

# ─────────────────────────────────────────
# HTML
# ─────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AgriML-BD | Crop Yield Prediction</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background: #f0f4f0;
    color: #1c2b1e;
    min-height: 100vh;
  }

  /* ── HEADER ── */
  header {
    background: linear-gradient(135deg, #1b4332, #2d6a4f);
    color: #fff;
    padding: 0;
  }
  .header-inner {
    max-width: 1100px; margin: 0 auto;
    padding: 28px 28px 22px;
    display: flex; align-items: center; gap: 20px;
  }
  .header-icon {
    width: 52px; height: 52px; border-radius: 12px;
    background: rgba(255,255,255,0.15);
    display: flex; align-items: center; justify-content: center;
    font-size: 26px; flex-shrink: 0;
  }
  .header-text h1 { font-size: 20px; font-weight: 800; letter-spacing: -0.3px; }
  .header-text p  { font-size: 12px; opacity: 0.7; margin-top: 4px; letter-spacing: 0.2px; }
  .header-badges  { display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap; }
  .badge {
    background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.25);
    border-radius: 20px; padding: 3px 10px;
    font-size: 10.5px; font-weight: 600; letter-spacing: 0.3px;
  }

  /* ── LAYOUT ── */
  .page { max-width: 1100px; margin: 28px auto 40px; padding: 0 20px; }
  .two-col { display: grid; grid-template-columns: 1fr 380px; gap: 20px; align-items: start; }

  /* ── CARD ── */
  .card {
    background: #fff;
    border-radius: 14px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid #e4ede6;
    overflow: hidden;
  }
  .card-header {
    padding: 16px 22px;
    border-bottom: 1px solid #edf2ee;
    display: flex; align-items: center; gap: 10px;
  }
  .card-header-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #2d6a4f; flex-shrink: 0;
  }
  .card-header-title {
    font-size: 13.5px; font-weight: 700; color: #1b4332;
  }
  .card-body { padding: 20px 22px; }

  /* ── SECTION LABEL ── */
  .sec-label {
    font-size: 10px; font-weight: 700; color: #52796f;
    text-transform: uppercase; letter-spacing: 1px;
    margin: 20px 0 10px; display: flex; align-items: center; gap: 6px;
  }
  .sec-label::after {
    content: ''; flex: 1; height: 1px; background: #d8e8da;
  }
  .sec-label:first-child { margin-top: 0; }

  /* ── GRID ── */
  .grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }

  /* ── FORM FIELDS ── */
  .field label {
    display: block; font-size: 11px; font-weight: 600;
    color: #52796f; margin-bottom: 5px; letter-spacing: 0.1px;
  }
  .field input, .field select {
    width: 100%; padding: 9px 12px;
    border: 1.5px solid #ccddd0;
    border-radius: 8px; font-size: 13.5px;
    font-family: inherit;
    background: #f8fbf8;
    color: #1c2b1e;
    transition: all 0.15s;
    appearance: none;
  }
  .field input:focus, .field select:focus {
    outline: none;
    border-color: #2d6a4f;
    box-shadow: 0 0 0 3px rgba(45,106,79,0.1);
    background: #fff;
  }
  .field select {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='7' viewBox='0 0 12 7'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%2352796f' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 11px center;
    padding-right: 32px;
  }

  /* ── BUTTON ── */
  .btn-predict {
    width: 100%; margin-top: 18px; padding: 14px;
    background: linear-gradient(135deg, #1b4332, #2d6a4f);
    color: #fff; border: none; border-radius: 10px;
    font-size: 14.5px; font-weight: 700; letter-spacing: 0.2px;
    font-family: inherit;
    cursor: pointer; transition: all 0.2s;
    box-shadow: 0 4px 14px rgba(27,67,50,0.3);
  }
  .btn-predict:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(27,67,50,0.35);
  }
  .btn-predict:active:not(:disabled) { transform: translateY(0); }
  .btn-predict:disabled { opacity: 0.55; cursor: not-allowed; box-shadow: none; }

  /* ── RESULT SECTION ── */
  .result-section { display: none; margin-top: 20px; }
  .result-section.visible {
    display: block;
    animation: slideUp 0.35s ease;
  }
  @keyframes slideUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .result-divider {
    border: none; border-top: 2px dashed #d8e8da; margin: 20px 0;
  }

  .result-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 14px; }

  .res-card {
    border-radius: 10px; padding: 16px 18px; text-align: center;
    position: relative; overflow: hidden;
  }
  .res-card.crop-card {
    background: linear-gradient(135deg, #d8f3dc, #b7e4c7);
    border: 1.5px solid #95d5b2;
  }
  .res-card.yield-card {
    background: linear-gradient(135deg, #d4edda, #c3e6cb);
    border: 1.5px solid #74c69d;
  }
  .res-card .rc-tag {
    display: inline-block;
    background: rgba(27,67,50,0.12); color: #1b4332;
    font-size: 9.5px; font-weight: 700; letter-spacing: 0.8px;
    text-transform: uppercase; border-radius: 4px;
    padding: 2px 7px; margin-bottom: 10px;
  }
  .res-card .rc-value {
    font-size: 22px; font-weight: 800; color: #1b4332; line-height: 1.2;
  }
  .res-card .rc-sub { font-size: 10.5px; color: #52796f; margin-top: 5px; }

  .conf-bar-wrap { margin-bottom: 14px; }
  .conf-label {
    display: flex; justify-content: space-between; align-items: center;
    font-size: 11px; font-weight: 600; color: #52796f; margin-bottom: 5px;
  }
  .conf-bar-bg {
    height: 7px; background: #d8e8da; border-radius: 99px; overflow: hidden;
  }
  .conf-bar-fill {
    height: 100%; background: linear-gradient(90deg, #2d6a4f, #52b788);
    border-radius: 99px; transition: width 0.5s ease;
    width: 0%;
  }

  .top3-wrap { display: flex; gap: 8px; flex-wrap: wrap; }
  .top3-pill {
    padding: 5px 14px; border-radius: 20px; font-size: 12px; font-weight: 600;
    border: 1.5px solid;
  }
  .top3-pill.rank1 {
    background: #1b4332; color: #fff; border-color: #1b4332;
  }
  .top3-pill.rank2 {
    background: #d8f3dc; color: #1b4332; border-color: #95d5b2;
  }
  .top3-pill.rank3 {
    background: #f0f4f0; color: #52796f; border-color: #ccddd0;
  }

  /* ── ERROR ── */
  .error-msg {
    display: none; background: #fff5f5; border: 1.5px solid #fc8181;
    border-radius: 8px; padding: 11px 15px; color: #c53030;
    font-size: 12.5px; margin-top: 12px; font-weight: 500;
  }
  .error-msg.visible { display: block; }

  /* ── RIGHT COLUMN ── */
  .stat-grid { display: flex; flex-direction: column; gap: 12px; }
  .stat-card {
    background: #fff; border-radius: 12px;
    border: 1px solid #e4ede6;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
    padding: 16px 18px;
  }
  .stat-card .sc-label {
    font-size: 10px; font-weight: 700; color: #74c69d;
    text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px;
  }
  .stat-card .sc-title {
    font-size: 13.5px; font-weight: 700; color: #1b4332; margin-bottom: 3px;
  }
  .stat-card .sc-model { font-size: 12px; color: #52796f; margin-bottom: 5px; }
  .stat-card .sc-score {
    display: inline-block;
    background: #d8f3dc; color: #1b4332;
    font-size: 11px; font-weight: 700;
    border-radius: 6px; padding: 2px 8px;
  }

  .pipeline-card {
    background: #fff; border-radius: 12px;
    border: 1px solid #e4ede6;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
    padding: 16px 18px;
  }
  .pipeline-title {
    font-size: 12px; font-weight: 700; color: #1b4332; margin-bottom: 12px;
  }
  .pipeline-step {
    display: flex; align-items: flex-start; gap: 10px; margin-bottom: 10px;
  }
  .pipeline-step:last-child { margin-bottom: 0; }
  .step-num {
    width: 22px; height: 22px; border-radius: 50%;
    background: #1b4332; color: #fff;
    font-size: 11px; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; margin-top: 1px;
  }
  .step-text { font-size: 12px; color: #52796f; line-height: 1.5; }
  .step-text strong { color: #1b4332; display: block; font-size: 12px; }

  /* ── FOOTER ── */
  footer {
    text-align: center; padding: 20px;
    font-size: 11px; color: #74c69d;
    border-top: 1px solid #d8e8da;
    background: #fff;
    margin-top: 20px;
  }

  @media (max-width: 820px) {
    .two-col { grid-template-columns: 1fr; }
    .grid3 { grid-template-columns: 1fr 1fr; }
    .header-inner { flex-direction: column; align-items: flex-start; gap: 12px; }
  }
  @media (max-width: 500px) {
    .grid3, .grid2 { grid-template-columns: 1fr; }
    .result-row { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<header>
  <div class="header-inner">
    <div class="header-icon">&#127807;</div>
    <div class="header-text">
      <h1>AgriML-BD &mdash; Crop Yield Prediction &amp; Recommendation</h1>
      <p>Machine Learning System for Bangladesh Agricultural Data</p>
      <div class="header-badges">
        <span class="badge">Light Seekers</span>
        <span class="badge">CSE-4889</span>
        <span class="badge">United International University</span>
        <span class="badge">3 Models &bull; 9 Experiments</span>
      </div>
    </div>
  </div>
</header>

<div class="page">
<div class="two-col">

  <!-- LEFT: Form -->
  <div>
    <div class="card">
      <div class="card-header">
        <div class="card-header-dot"></div>
        <div class="card-header-title">Input Parameters</div>
      </div>
      <div class="card-body">

        <div class="sec-label">Location &amp; Season</div>
        <div class="grid2">
          <div class="field">
            <label>District</label>
            <select id="inp-district">
              {% for d in districts %}<option value="{{ d }}">{{ d }}</option>{% endfor %}
            </select>
          </div>
          <div class="field">
            <label>Season</label>
            <select id="inp-season">
              {% for s in seasons %}<option value="{{ s }}">{{ s }}</option>{% endfor %}
            </select>
          </div>
        </div>

        <div class="sec-label">Soil Nutrients</div>
        <div class="grid3">
          <div class="field">
            <label>Nitrogen — N (mg/kg)</label>
            <input type="number" id="inp-N" value="80" min="0" max="200" step="1">
          </div>
          <div class="field">
            <label>Phosphorus — P (mg/kg)</label>
            <input type="number" id="inp-P" value="45" min="0" max="200" step="1">
          </div>
          <div class="field">
            <label>Potassium — K (mg/kg)</label>
            <input type="number" id="inp-K" value="40" min="0" max="250" step="1">
          </div>
        </div>
        <div class="grid2" style="margin-top:12px">
          <div class="field">
            <label>Soil pH</label>
            <input type="number" id="inp-ph" value="6.5" min="4.0" max="9.0" step="0.1">
          </div>
          <div class="field">
            <label>Annual Rainfall (mm)</label>
            <input type="number" id="inp-rainfall" value="200" min="50" max="600" step="1">
          </div>
        </div>

        <div class="sec-label">Weather Conditions</div>
        <div class="grid3">
          <div class="field">
            <label>Avg Temperature (&#176;C)</label>
            <input type="number" id="inp-avgTemp" value="26" min="5" max="45" step="0.5">
          </div>
          <div class="field">
            <label>Min Temperature (&#176;C)</label>
            <input type="number" id="inp-minTemp" value="12" min="-5" max="40" step="0.5">
          </div>
          <div class="field">
            <label>Max Temperature (&#176;C)</label>
            <input type="number" id="inp-maxTemp" value="38" min="15" max="55" step="0.5">
          </div>
        </div>
        <div class="grid3" style="margin-top:12px">
          <div class="field">
            <label>Avg Humidity (%)</label>
            <input type="number" id="inp-avgHum" value="72" min="20" max="100" step="1">
          </div>
          <div class="field">
            <label>Min Humidity (%)</label>
            <input type="number" id="inp-minHum" value="60" min="20" max="100" step="1">
          </div>
          <div class="field">
            <label>Max Humidity (%)</label>
            <input type="number" id="inp-maxHum" value="85" min="20" max="100" step="1">
          </div>
        </div>

        <div class="sec-label">Crop &amp; Area</div>
        <div class="grid2">
          <div class="field">
            <label>Crop (Task A — yield prediction)</label>
            <select id="inp-crop">
              {% for c in crops %}<option value="{{ c }}">{{ c }}</option>{% endfor %}
            </select>
          </div>
          <div class="field">
            <label>Cultivated Area (hectares)</label>
            <input type="number" id="inp-area" value="5000" min="1" step="100">
          </div>
        </div>

        <button class="btn-predict" id="btn-run" onclick="runPrediction()">
          Run Prediction
        </button>

        <div class="error-msg" id="err-box"></div>

        <!-- RESULT -->
        <div class="result-section" id="result-section">
          <hr class="result-divider">

          <div class="result-row">
            <div class="res-card crop-card">
              <div class="rc-tag">Task B &mdash; Best Crop</div>
              <div class="rc-value" id="res-crop">—</div>
              <div class="rc-sub">Recommended for your conditions</div>
            </div>
            <div class="res-card yield-card">
              <div class="rc-tag">Task A &mdash; Yield</div>
              <div class="rc-value" id="res-yield">—</div>
              <div class="rc-sub" id="res-yield-sub">metric tons</div>
            </div>
          </div>

          <div class="conf-bar-wrap">
            <div class="conf-label">
              <span>Recommendation Confidence</span>
              <strong id="res-conf">—</strong>
            </div>
            <div class="conf-bar-bg">
              <div class="conf-bar-fill" id="conf-fill"></div>
            </div>
          </div>

          <div class="sec-label" style="margin-top:14px">Top 3 Crops</div>
          <div class="top3-wrap" id="res-top3"></div>
        </div>

      </div>
    </div>
  </div>

  <!-- RIGHT: Info -->
  <div>
    <div class="stat-grid">

      <div class="stat-card">
        <div class="sc-label">Task A</div>
        <div class="sc-title">Yield Prediction</div>
        <div class="sc-model">Gradient Boosting Regressor</div>
        <span class="sc-score">R&sup2; = 0.9455 &nbsp;&bull;&nbsp; RMSE = 0.5679</span>
      </div>

      <div class="stat-card">
        <div class="sc-label">Task B</div>
        <div class="sc-title">Crop Recommendation</div>
        <div class="sc-model">Random Forest Classifier</div>
        <span class="sc-score">Accuracy = 90.20% &nbsp;&bull;&nbsp; F1 = 0.9019</span>
      </div>

      <div class="stat-card">
        <div class="sc-label">Dataset</div>
        <div class="sc-title">Bangladesh Agricultural Data</div>
        <div class="sc-model">Merged from 3 sources</div>
        <span class="sc-score">4,607 records &nbsp;&bull;&nbsp; 72 crops &nbsp;&bull;&nbsp; 64 districts</span>
      </div>

      <div class="pipeline-card">
        <div class="pipeline-title">How It Works</div>

        <div class="pipeline-step">
          <div class="step-num">1</div>
          <div class="step-text">
            <strong>Input your data</strong>
            District, season, soil nutrients, weather conditions
          </div>
        </div>
        <div class="pipeline-step">
          <div class="step-num">2</div>
          <div class="step-text">
            <strong>Task B &mdash; Classification</strong>
            RF model predicts best crop (72 classes, 94% accuracy)
          </div>
        </div>
        <div class="pipeline-step">
          <div class="step-num">3</div>
          <div class="step-text">
            <strong>Task A &mdash; Regression</strong>
            GB model predicts yield in metric tons (R&sup2;=0.9455)
          </div>
        </div>
        <div class="pipeline-step">
          <div class="step-num">4</div>
          <div class="step-text">
            <strong>Results</strong>
            Best crop + top 3 alternatives + expected yield
          </div>
        </div>
      </div>

    </div>
  </div>

</div>
</div>

<footer>
  AgriML-BD &nbsp;&bull;&nbsp; Team Light Seekers &nbsp;&bull;&nbsp; CSE-4889 &nbsp;&bull;&nbsp; United International University Bangladesh
</footer>

<script>
async function runPrediction() {
  const btn     = document.getElementById('btn-run');
  const errBox  = document.getElementById('err-box');
  const resSec  = document.getElementById('result-section');

  // reset UI
  btn.disabled     = true;
  btn.textContent  = 'Running...';
  errBox.className = 'error-msg';
  resSec.className = 'result-section';   // hide old result

  // validate
  const fields = ['inp-N','inp-P','inp-K','inp-ph','inp-rainfall',
                  'inp-avgTemp','inp-minTemp','inp-maxTemp',
                  'inp-avgHum','inp-minHum','inp-maxHum','inp-area'];
  for (const id of fields) {
    const v = parseFloat(document.getElementById(id).value);
    if (isNaN(v)) {
      showError('Please fill all numeric fields correctly.');
      reset(); return;
    }
  }

  const payload = {
    district : document.getElementById('inp-district').value,
    season   : document.getElementById('inp-season').value,
    crop     : document.getElementById('inp-crop').value,
    N        : parseFloat(document.getElementById('inp-N').value),
    P        : parseFloat(document.getElementById('inp-P').value),
    K        : parseFloat(document.getElementById('inp-K').value),
    ph       : parseFloat(document.getElementById('inp-ph').value),
    area     : parseFloat(document.getElementById('inp-area').value),
    avgTemp  : parseFloat(document.getElementById('inp-avgTemp').value),
    minTemp  : parseFloat(document.getElementById('inp-minTemp').value),
    maxTemp  : parseFloat(document.getElementById('inp-maxTemp').value),
    avgHum   : parseFloat(document.getElementById('inp-avgHum').value),
    minHum   : parseFloat(document.getElementById('inp-minHum').value),
    maxHum   : parseFloat(document.getElementById('inp-maxHum').value),
    rainfall : parseFloat(document.getElementById('inp-rainfall').value),
  };

  try {
    const resp = await fetch('/predict', {
      method  : 'POST',
      headers : { 'Content-Type': 'application/json' },
      body    : JSON.stringify(payload)
    });

    if (!resp.ok) { showError('Server error: ' + resp.status); reset(); return; }

    const data = await resp.json();
    if (data.error) { showError(data.error); reset(); return; }

    // fill results
    document.getElementById('res-crop').textContent  = data.recommended_crop;
    document.getElementById('res-yield').textContent = Number(data.yield_tons).toLocaleString() + ' MT';
    document.getElementById('res-yield-sub').textContent =
      'metric tons — for ' + payload.crop;
    const confPct = (data.confidence * 100).toFixed(1);
    document.getElementById('res-conf').textContent = confPct + '%';
    setTimeout(function() {
      document.getElementById('conf-fill').style.width = confPct + '%';
    }, 50);

    const top3Div = document.getElementById('res-top3');
    top3Div.innerHTML = '';
    const rankClass = ['rank1','rank2','rank3'];
    data.top3.forEach(function(crop, i) {
      const span = document.createElement('span');
      span.className   = 'top3-pill ' + (rankClass[i] || 'rank3');
      span.textContent = (i + 1) + '.  ' + crop;
      top3Div.appendChild(span);
    });

    resSec.className = 'result-section visible';

  } catch (e) {
    showError('Connection failed. Make sure app.py is running. (' + e.message + ')');
  }

  reset();
}

function showError(msg) {
  const b = document.getElementById('err-box');
  b.textContent = msg;
  b.className   = 'error-msg visible';
}

function reset() {
  const btn = document.getElementById('btn-run');
  btn.disabled    = false;
  btn.textContent = 'Run Prediction';
}
</script>
</body>
</html>
"""

# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(
        HTML,
        crops     = CLASSES['crops'],
        districts = CLASSES['districts'],
        seasons   = CLASSES['seasons'],
    )

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        d = request.get_json(force=True)

        le_s = ENCODER['season']
        le_d = ENCODER['district']
        le_c = ENCODER['crop']

        if d['district'] not in le_d.classes_:
            return jsonify(error=f"Unknown district: {d['district']}")
        if d['season'] not in le_s.classes_:
            return jsonify(error=f"Unknown season: {d['season']}")
        if d['crop'] not in le_c.classes_:
            return jsonify(error=f"Unknown crop: {d['crop']}")

        s_enc = int(le_s.transform([d['season']])[0])
        d_enc = int(le_d.transform([d['district']])[0])
        c_enc = int(le_c.transform([d['crop']])[0])

        # Task B — Crop Classification
        cls_row    = np.array([[d['N'], d['P'], d['K'], d['ph'],
                                d['avgTemp'], d['avgHum'], d['rainfall'],
                                s_enc, d_enc]])
        cls_scaled = ENCODER['sc_c'].transform(cls_row)
        proba      = MODEL['cls'].predict_proba(cls_scaled)[0]
        top3_idx   = np.argsort(proba)[::-1][:3]
        top3_crops = [le_c.classes_[i] for i in top3_idx]
        confidence = float(proba[top3_idx[0]])

        # Task A — Yield Regression (for the crop the user selected)
        reg_row    = np.array([[d['area'], d['N'], d['P'], d['K'], d['ph'],
                                d['avgTemp'], d['minTemp'], d['maxTemp'],
                                d['avgHum'], d['minHum'], d['maxHum'],
                                d['rainfall'], s_enc, d_enc, c_enc]])
        reg_scaled  = ENCODER['sc_r'].transform(reg_row)
        yield_log   = float(MODEL['reg'].predict(reg_scaled)[0])
        yield_tons  = float(np.expm1(yield_log))

        return jsonify(
            recommended_crop = top3_crops[0],
            confidence       = confidence,
            top3             = top3_crops,
            yield_tons       = round(yield_tons, 1),
        )

    except Exception as e:
        return jsonify(error=str(e))


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == '__main__':
    load_models()
    print("\n" + "="*50)
    print("  App is ready!")
    print("  Open: http://localhost:8080")
    print("="*50 + "\n")
    app.run(debug=False, host='0.0.0.0', port=8080)
