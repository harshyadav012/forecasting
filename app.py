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


# ===============================
# LOAD AND TRAIN
# ===============================
def initialize_model():
    global data, scaler, gmm, crisis_regime
    global model_gold_normal, model_gold_crisis
    global model_silver_normal, model_silver_crisis

    print("📥 Loading external dataset...")
    loaded = create_dataset()
    loaded.index = pd.to_datetime(loaded.index)
    loaded = loaded.sort_index()

    print("Rows after create_dataset():", len(loaded))
    print("Columns found:", list(loaded.columns))

    required_cols = features + [target_gold, target_silver]
    missing_cols = [col for col in required_cols if col not in loaded.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    loaded = loaded.dropna(subset=required_cols)
    print("Rows after dropna on required columns:", len(loaded))

    if loaded.empty:
        raise ValueError("REAL DATA FAILED: Dataset is empty after cleaning. External series may have failed to download.")

    data_local, scaler_local, gmm_local = detect_regimes(loaded, regime_vars)
    print("Rows after detect_regimes():", len(data_local))

    train = data_local[data_local.index.year <= 2020].copy()
    print("Training rows up to 2020:", len(train))

    if train.empty:
        print("⚠️ Training set up to 2020 is empty. Using full dataset for training.")
        train = data_local.copy()

    if train.empty:
        raise ValueError("REAL DATA FAILED: No training data available.")

    if 'Regime' not in train.columns:
        raise ValueError("REAL DATA FAILED: Regime column not found after detect_regimes().")

    crisis_regime_local = train.groupby('Regime')['VIX'].mean().idxmax()

    # Gold models
    mg_normal = GradientBoostingRegressor(random_state=42)
    mg_normal.fit(train[features], train[target_gold])

    crisis_train = train[train['Regime'] == crisis_regime_local].copy()
    if crisis_train.empty:
        print("⚠️ Crisis regime subset is empty for gold. Using full training set.")
        crisis_train = train.copy()

    mg_crisis = GradientBoostingRegressor(random_state=42)
    mg_crisis.fit(crisis_train[features], crisis_train[target_gold])

    # Silver models
    ms_normal = GradientBoostingRegressor(random_state=42)
    ms_normal.fit(train[features], train[target_silver])

    crisis_train_silver = train[train['Regime'] == crisis_regime_local].copy()
    if crisis_train_silver.empty:
        print("⚠️ Crisis regime subset is empty for silver. Using full training set.")
        crisis_train_silver = train.copy()

    ms_crisis = GradientBoostingRegressor(random_state=42)
    ms_crisis.fit(crisis_train_silver[features], crisis_train_silver[target_silver])

    data = data_local
    scaler = scaler_local
    gmm = gmm_local
    crisis_regime = crisis_regime_local
    model_gold_normal = mg_normal
    model_gold_crisis = mg_crisis
    model_silver_normal = ms_normal
    model_silver_crisis = ms_crisis

    print("✅ Models Ready")


initialize_model()


# ===============================
# SCENARIO GENERATOR
# ===============================
def generate_scenario(last_usdinr):
    return {
        'Fed': float(np.clip(np.random.normal(2.0, 1.0), 0, 10)),
        'Inflation': float(np.clip(np.random.normal(3.0, 1.0), 0.5, 12)),
        'US10Y': float(np.clip(np.random.normal(3.0, 1.0), 0.1, 8)),
        'DXY': float(np.clip(np.random.normal(100, 5), 80, 120)),
        'VIX': float(np.clip(np.random.normal(20, 5), 9, 80)),
        'FSI': float(np.clip(np.random.normal(0.5, 0.5), -2, 3)),
        'Spread': float(np.clip(np.random.normal(1.0, 0.5), -1, 5)),
        'RealYield': float(np.clip(np.random.normal(1.0, 1.0), -3, 5)),
        'USDINR': float(np.clip(last_usdinr * (1 + np.random.normal(0, 0.01)), 50, 120))
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
        "status": "ok",
        "rows": int(len(data)) if data is not None else 0,
        "latest_date": str(data.index[-1].date()) if data is not None and len(data) > 0 else None,
        "latest_gold": float(data['Gold'].iloc[-1]) if data is not None and len(data) > 0 else None,
        "latest_silver": float(data['Silver'].iloc[-1]) if data is not None and len(data) > 0 else None,
        "latest_usdinr": float(data['USDINR'].iloc[-1]) if data is not None and len(data) > 0 else None
    })


@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        req = request.get_json(silent=True) or {}
        n_months = max(1, min(int(req.get("months", 6)), 36))
        n_simulations = 300

        mc_gold = np.zeros((n_months, n_simulations))
        mc_silver = np.zeros((n_months, n_simulations))
        mc_usdinr = np.zeros((n_months, n_simulations))

        last_gold = float(data['Gold'].iloc[-1])
        last_silver = float(data['Silver'].iloc[-1])
        last_usdinr = float(data['USDINR'].iloc[-1])

        for sim in range(n_simulations):
            g = last_gold
            s = last_silver
            usdinr = last_usdinr

            for t in range(n_months):
                scenario = generate_scenario(usdinr)
                usdinr = scenario['USDINR']

                row = pd.DataFrame([{f: scenario[f] for f in features}])

                regime_input = pd.DataFrame(
                    [[scenario[v] for v in regime_vars]],
                    columns=regime_vars
                )

                regime = gmm.predict(scaler.transform(regime_input))[0]

                if regime == crisis_regime:
                    g_ret = model_gold_crisis.predict(row[features])[0]
                    s_ret = model_silver_crisis.predict(row[features])[0]
                else:
                    g_ret = model_gold_normal.predict(row[features])[0]
                    s_ret = model_silver_normal.predict(row[features])[0]

                g *= (1 + g_ret)
                s *= (1 + s_ret)

                mc_gold[t, sim] = g
                mc_silver[t, sim] = s
                mc_usdinr[t, sim] = usdinr

        gold_mean = mc_gold.mean(axis=1)
        silver_mean = mc_silver.mean(axis=1)
        usdinr_mean = mc_usdinr.mean(axis=1)

        gold_p10 = np.percentile(mc_gold, 10, axis=1)
        gold_p90 = np.percentile(mc_gold, 90, axis=1)
        silver_p10 = np.percentile(mc_silver, 10, axis=1)
        silver_p90 = np.percentile(mc_silver, 90, axis=1)

        dates = pd.date_range(
            start=data.index[-1] + pd.offsets.MonthEnd(1),
            periods=n_months,
            freq='ME'
        )

        result = []
        for i in range(n_months):
            result.append({
                "date": str(dates[i]),
                "gold_now": float(last_gold),
                "silver_now": float(last_silver),
                "gold_usd": float(gold_mean[i]),
                "gold_usd_p10": float(gold_p10[i]),
                "gold_usd_p90": float(gold_p90[i]),
                "silver_usd": float(silver_mean[i]),
                "silver_usd_p10": float(silver_p10[i]),
                "silver_usd_p90": float(silver_p90[i]),
                "usdinr": float(usdinr_mean[i])
            })

        return jsonify(result)

    except Exception as e:
        print("❌ Forecast route failed:", str(e))
        return jsonify({"error": str(e)}), 500


# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
