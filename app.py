from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from Data_set_creation import create_dataset
from regime_detection import detect_regimes

app = Flask(__name__)
CORS(app)

print("🚀 Initializing Model...")

# ===============================
# CONFIG
# ===============================
features = ['Fed', 'Inflation', 'US10Y', 'DXY', 'VIX', 'FSI', 'Spread', 'RealYield', 'USDINR']
target_gold = 'Gold_Return'
target_silver = 'Silver_Return'
regime_vars = ['VIX', 'FSI', 'Spread', 'Fed', 'Inflation', 'USDINR']

data = None
scaler = None
gmm = None
crisis_regime = None
model_gold_normal = None
model_gold_crisis = None
model_silver_normal = None
model_silver_crisis = None

data_source = "unknown"
startup_error = None


# ===============================
# FALLBACK DATA
# ===============================
def build_fallback_dataset():
    print("⚠️ Using fallback dataset")

    dates = pd.date_range(start="2015-01-31", periods=140, freq="ME")
    rng = np.random.default_rng(42)

    gold = np.linspace(1200, 3200, len(dates)) + rng.normal(0, 40, len(dates))
    silver = np.linspace(15, 34, len(dates)) + rng.normal(0, 0.8, len(dates))

    df = pd.DataFrame({
        "Gold": gold,
        "Silver": silver,
        "Fed": rng.uniform(0, 5, len(dates)),
        "Inflation": rng.uniform(1, 5, len(dates)),
        "US10Y": rng.uniform(1, 5, len(dates)),
        "DXY": rng.uniform(90, 110, len(dates)),
        "VIX": rng.uniform(10, 30, len(dates)),
        "FSI": rng.uniform(-1, 1, len(dates)),
        "Spread": rng.uniform(0, 2, len(dates)),
        "RealYield": rng.uniform(-1, 3, len(dates)),
        "USDINR": rng.uniform(60, 90, len(dates))
    }, index=dates)

    df["Gold_Return"] = df["Gold"].pct_change()
    df["Silver_Return"] = df["Silver"].pct_change()

    return df.dropna()


# ===============================
# INITIALIZE MODEL
# ===============================
def initialize_model():
    global data, scaler, gmm, crisis_regime
    global model_gold_normal, model_gold_crisis
    global model_silver_normal, model_silver_crisis
    global data_source, startup_error

    try:
        print("📥 Loading external dataset...")
        loaded = create_dataset()

        loaded.index = pd.to_datetime(loaded.index)
        loaded = loaded.sort_index()

        required_cols = features + [target_gold, target_silver]

        loaded = loaded.dropna(subset=required_cols)

        if loaded.empty:
            raise ValueError("Dataset empty")

        data_local = loaded
        data_source = "real"
        startup_error = None

        print(f"✅ Real dataset loaded ({len(data_local)} rows)")

    except Exception as e:
        print("⚠️ DATA LOAD FAILED:", str(e))
        data_local = build_fallback_dataset()
        data_source = "fallback"
        startup_error = str(e)

    data_local, scaler_local, gmm_local = detect_regimes(data_local, regime_vars)

    train = data_local.copy()

    crisis_regime_local = train.groupby('Regime')['VIX'].mean().idxmax()

    # GOLD MODELS
    mg_normal = GradientBoostingRegressor(random_state=42)
    mg_normal.fit(train[features], train[target_gold])

    mg_crisis = GradientBoostingRegressor(random_state=42)
    mg_crisis.fit(train[features], train[target_gold])

    # SILVER MODELS
    ms_normal = GradientBoostingRegressor(random_state=42)
    ms_normal.fit(train[features], train[target_silver])

    ms_crisis = GradientBoostingRegressor(random_state=42)
    ms_crisis.fit(train[features], train[target_silver])

    data = data_local
    scaler = scaler_local
    gmm = gmm_local
    crisis_regime = crisis_regime_local

    model_gold_normal = mg_normal
    model_gold_crisis = mg_crisis
    model_silver_normal = ms_normal
    model_silver_crisis = ms_crisis

    print("✅ Models Ready")
    print(f"📊 Data source: {data_source}")


initialize_model()


# ===============================
# SCENARIO GENERATOR
# ===============================
def generate_scenario(last_usdinr):
    return {
        'Fed': float(np.random.uniform(0, 5)),
        'Inflation': float(np.random.uniform(1, 5)),
        'US10Y': float(np.random.uniform(1, 5)),
        'DXY': float(np.random.uniform(90, 110)),
        'VIX': float(np.random.uniform(10, 30)),
        'FSI': float(np.random.uniform(-1, 1)),
        'Spread': float(np.random.uniform(0, 2)),
        'RealYield': float(np.random.uniform(-1, 3)),
        'USDINR': float(last_usdinr * (1 + np.random.normal(0, 0.01)))
    }


# ===============================
# ROUTES
# ===============================
@app.route('/')
def home():
    return render_template("index.html")


@app.route('/health')
def health():
    return jsonify({
        "data_source": data_source,
        "error": startup_error,
        "rows": len(data),
        "gold_now": float(data['Gold'].iloc[-1]),
        "silver_now": float(data['Silver'].iloc[-1]),
        "usdinr": float(data['USDINR'].iloc[-1])
    })


@app.route('/forecast', methods=['POST'])
def forecast():
    n_months = int(request.json.get("months", 6))
    n_sim = 200

    last_gold = data['Gold'].iloc[-1]
    last_silver = data['Silver'].iloc[-1]
    last_usdinr = data['USDINR'].iloc[-1]

    results = []

    for t in range(n_months):
        scenario = generate_scenario(last_usdinr)

        gold = last_gold * (1 + np.random.normal(0.02, 0.05))
        silver = last_silver * (1 + np.random.normal(0.02, 0.05))

        results.append({
            "date": str(pd.Timestamp.today() + pd.DateOffset(months=t)),
            "gold_now": float(last_gold),
            "silver_now": float(last_silver),
            "gold_usd": float(gold),
            "silver_usd": float(silver),
            "usdinr": float(scenario['USDINR'])
        })

    return jsonify(results)


# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
