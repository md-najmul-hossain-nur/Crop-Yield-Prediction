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
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #eef2f7;
    color: #2d3748;
    min-height: 100vh;
  }

  header {
    background: linear-gradient(135deg, #1a365d, #2b6cb0);
    color: #fff;
    padding: 24px 40px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
  }
  header h1 { font-size: 22px; font-weight: 700; letter-spacing: 0.3px; }
  header p  { font-size: 12px; opacity: 0.75; margin-top: 5px; }

  .page { max-width: 1000px; margin: 30px auto; padding: 0 20px; }

  .card {
    background: #fff;
    border-radius: 10px;
    box-shadow: 0 1px 8px rgba(0,0,0,0.07);
    padding: 26px;
    margin-bottom: 22px;
  }

  .card-title {
    font-size: 15px; font-weight: 700; color: #1a365d;
    border-bottom: 2px solid #e8f0fe;
    padding-bottom: 10px; margin-bottom: 20px;
  }

  .section-label {
    font-size: 11px; font-weight: 700; color: #718096;
    text-transform: uppercase; letter-spacing: 0.8px;
    margin: 18px 0 10px;
  }

  .grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }

  label {
    display: block; font-size: 11.5px; font-weight: 600;
    color: #4a5568; margin-bottom: 4px;
  }

  input[type=number], select {
    width: 100%; padding: 8px 11px;
    border: 1.5px solid #d1d9e6;
    border-radius: 6px; font-size: 14px;
    background: #fafbfc;
    transition: border-color 0.15s, box-shadow 0.15s;
  }
  input[type=number]:focus, select:focus {
    outline: none;
    border-color: #2b6cb0;
    box-shadow: 0 0 0 3px rgba(43,108,176,0.12);
    background: #fff;
  }

  .btn-predict {
    width: 100%; margin-top: 20px; padding: 13px;
    background: linear-gradient(135deg, #1a365d, #2b6cb0);
    color: #fff; border: none; border-radius: 7px;
    font-size: 15px; font-weight: 700; letter-spacing: 0.3px;
    cursor: pointer; transition: opacity 0.2s, transform 0.1s;
  }
  .btn-predict:hover:not(:disabled) { opacity: 0.92; transform: translateY(-1px); }
  .btn-predict:disabled { opacity: 0.6; cursor: not-allowed; }

  /* ── Result section ── */
  .result-section {
    display: none;
    margin-top: 22px;
    animation: fadeIn 0.3s ease;
  }
  .result-section.visible { display: block; }

  @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

  .result-top {
    display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
    margin-bottom: 14px;
  }

  .result-card {
    border-radius: 8px; padding: 18px; text-align: center;
  }
  .result-card.green { background: #f0fff4; border: 1.5px solid #9ae6b4; }
  .result-card.blue  { background: #ebf8ff; border: 1.5px solid #90cdf4; }

  .result-card .rc-label {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.6px; color: #718096; margin-bottom: 8px;
  }
  .result-card .rc-value {
    font-size: 24px; font-weight: 800; color: #1a365d;
    line-height: 1.2;
  }
  .result-card .rc-sub {
    font-size: 11px; color: #718096; margin-top: 5px;
  }

  .result-note {
    font-size: 11.5px; color: #718096;
    background: #f7fafc; border-radius: 6px;
    padding: 10px 14px; margin-bottom: 14px;
    border-left: 3px solid #2b6cb0;
  }

  .top3-row { display: flex; gap: 8px; flex-wrap: wrap; }
  .top3-tag {
    background: #ebf4ff; color: #2b6cb0;
    border: 1px solid #bee3f8;
    padding: 5px 13px; border-radius: 20px;
    font-size: 12px; font-weight: 600;
  }
  .top3-tag.first { background: #2b6cb0; color: #fff; }

  /* ── Error ── */
  .error-msg {
    display: none; background: #fff5f5; border: 1.5px solid #fc8181;
    border-radius: 7px; padding: 12px 16px; color: #c53030;
    font-size: 13px; margin-top: 14px;
  }
  .error-msg.visible { display: block; }

  /* ── Info cards ── */
  .info-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }
  .info-card {
    background: #f8fafd; border-radius: 8px;
    padding: 14px 16px; border: 1px solid #e2e8f0;
  }
  .info-card .ic-title { font-weight: 700; font-size: 13px; color: #1a365d; margin-bottom: 4px; }
  .info-card .ic-model { font-size: 12px; color: #4a5568; }
  .info-card .ic-score { font-size: 11px; color: #718096; margin-top: 3px; }

  .note-box {
    background: #fffbeb; border: 1.5px solid #f6e05e;
    border-radius: 7px; padding: 12px 16px;
    font-size: 12.5px; color: #744210;
    margin-bottom: 18px; line-height: 1.5;
  }

  footer {
    text-align: center; padding: 22px;
    font-size: 11.5px; color: #a0aec0;
  }
</style>
</head>
<body>

<header>
  <h1>AgriML-BD — Crop Yield Prediction & Recommendation System</h1>
  <p>Team: Light Seekers &nbsp;|&nbsp; CSE-4889 &nbsp;|&nbsp; United International University, Bangladesh</p>
</header>

<div class="page">

  <div class="card">
    <div class="card-title">Crop Prediction Input Form</div>

    <div class="note-box">
      <strong>How this works:</strong><br>
      Fill in your location, soil nutrients, and weather data.<br>
      &bull; <strong>Task A</strong> — Predicts yield (metric tons) for the crop you select below.<br>
      &bull; <strong>Task B</strong> — Recommends the best crop to grow based on your soil &amp; climate.
    </div>

    <div class="section-label">Location &amp; Season</div>
    <div class="grid2">
      <div>
        <label>District</label>
        <select id="inp-district">
          {% for d in districts %}<option value="{{ d }}">{{ d }}</option>{% endfor %}
        </select>
      </div>
      <div>
        <label>Season</label>
        <select id="inp-season">
          {% for s in seasons %}<option value="{{ s }}">{{ s }}</option>{% endfor %}
        </select>
      </div>
    </div>

    <div class="section-label">Soil Nutrients</div>
    <div class="grid3">
      <div>
        <label>Nitrogen — N (mg/kg)</label>
        <input type="number" id="inp-N" value="80" min="0" max="200" step="1">
      </div>
      <div>
        <label>Phosphorus — P (mg/kg)</label>
        <input type="number" id="inp-P" value="45" min="0" max="200" step="1">
      </div>
      <div>
        <label>Potassium — K (mg/kg)</label>
        <input type="number" id="inp-K" value="40" min="0" max="250" step="1">
      </div>
    </div>
    <div class="grid2" style="margin-top:14px">
      <div>
        <label>Soil pH</label>
        <input type="number" id="inp-ph" value="6.5" min="4.0" max="9.0" step="0.1">
      </div>
      <div>
        <label>Annual Rainfall (mm)</label>
        <input type="number" id="inp-rainfall" value="200" min="50" max="600" step="1">
      </div>
    </div>

    <div class="section-label">Weather Conditions</div>
    <div class="grid3">
      <div>
        <label>Avg Temperature (°C)</label>
        <input type="number" id="inp-avgTemp" value="26" min="5" max="45" step="0.5">
      </div>
      <div>
        <label>Min Temperature (°C)</label>
        <input type="number" id="inp-minTemp" value="12" min="-5" max="40" step="0.5">
      </div>
      <div>
        <label>Max Temperature (°C)</label>
        <input type="number" id="inp-maxTemp" value="38" min="15" max="55" step="0.5">
      </div>
    </div>
    <div class="grid3" style="margin-top:14px">
      <div>
        <label>Avg Humidity (%)</label>
        <input type="number" id="inp-avgHum" value="72" min="20" max="100" step="1">
      </div>
      <div>
        <label>Min Humidity (%)</label>
        <input type="number" id="inp-minHum" value="60" min="20" max="100" step="1">
      </div>
      <div>
        <label>Max Humidity (%)</label>
        <input type="number" id="inp-maxHum" value="85" min="20" max="100" step="1">
      </div>
    </div>

    <div class="section-label">Yield Prediction — Select Crop &amp; Area</div>
    <div class="grid2">
      <div>
        <label>Crop (for Task A yield prediction)</label>
        <select id="inp-crop">
          {% for c in crops %}<option value="{{ c }}">{{ c }}</option>{% endfor %}
        </select>
      </div>
      <div>
        <label>Cultivated Area (hectares)</label>
        <input type="number" id="inp-area" value="5000" min="1" step="100">
      </div>
    </div>

    <button class="btn-predict" id="btn-run" onclick="runPrediction()">
      Run Prediction
    </button>

    <div class="error-msg" id="err-box"></div>

    <!-- ── RESULT ── -->
    <div class="result-section" id="result-section">

      <div class="section-label" style="margin-top:22px">Prediction Results</div>

      <div class="result-top">
        <div class="result-card green">
          <div class="rc-label">Task B — Recommended Crop</div>
          <div class="rc-value" id="res-crop">—</div>
          <div class="rc-sub">Best crop for your soil &amp; climate conditions</div>
        </div>
        <div class="result-card blue">
          <div class="rc-label">Task A — Predicted Yield</div>
          <div class="rc-value" id="res-yield">—</div>
          <div class="rc-sub" id="res-yield-sub">metric tons for selected crop</div>
        </div>
      </div>

      <div class="result-note">
        Recommendation confidence: <strong id="res-conf">—</strong> &nbsp;|&nbsp;
        Model: Random Forest Classifier (72 crop classes) &nbsp;|&nbsp;
        Yield model: Gradient Boosting Regressor (R² = 0.9455)
      </div>

      <div class="section-label">Top 3 Recommended Crops</div>
      <div class="top3-row" id="res-top3"></div>

    </div>
  </div>

  <!-- Info -->
  <div class="card">
    <div class="card-title">Model Information</div>
    <div class="info-grid">
      <div class="info-card">
        <div class="ic-title">Task A — Yield Prediction</div>
        <div class="ic-model">Gradient Boosting (HistGB)</div>
        <div class="ic-score">R² = 0.9455 &nbsp;|&nbsp; RMSE = 0.5679</div>
      </div>
      <div class="info-card">
        <div class="ic-title">Task B — Crop Recommendation</div>
        <div class="ic-model">Random Forest Classifier</div>
        <div class="ic-score">Accuracy = 90.20% &nbsp;|&nbsp; F1 = 0.9019</div>
      </div>
      <div class="info-card">
        <div class="ic-title">Dataset</div>
        <div class="ic-model">4,607 merged records</div>
        <div class="ic-score">72 crops &nbsp;|&nbsp; 64 districts &nbsp;|&nbsp; 3 seasons</div>
      </div>
    </div>
  </div>

</div>

<footer>AgriML-BD &nbsp;|&nbsp; Light Seekers &nbsp;|&nbsp; CSE-4889 &nbsp;|&nbsp; United International University Bangladesh</footer>

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
    document.getElementById('res-conf').textContent  =
      (data.confidence * 100).toFixed(1) + '%';

    const top3Div = document.getElementById('res-top3');
    top3Div.innerHTML = '';
    data.top3.forEach(function(crop, i) {
      const span = document.createElement('span');
      span.className   = 'top3-tag' + (i === 0 ? ' first' : '');
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
