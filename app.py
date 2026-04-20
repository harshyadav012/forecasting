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
# LOAD DATA
# ===============================
data = create_dataset()
data.index = pd.to_datetime(data.index)
data = data.sort_index()

features = ['Fed','Inflation','US10Y','DXY','VIX','FSI','Spread','RealYield','USDINR']
target_gold = 'Gold_Return'
target_silver = 'Silver_Return'

data = data.dropna(subset=features + [target_gold, target_silver])

regime_vars = ['VIX','FSI','Spread','Fed','Inflation','USDINR']
data, scaler, gmm = detect_regimes(data, regime_vars)

train = data[data.index.year <= 2020].copy()

crisis_regime = train.groupby('Regime')['VIX'].mean().idxmax()

# ===============================
# MODELS
# ===============================
# Gold
model_gold_normal = GradientBoostingRegressor(random_state=42)
model_gold_normal.fit(train[features], train[target_gold])

crisis_train = train[train['Regime'] == crisis_regime]
model_gold_crisis = GradientBoostingRegressor(random_state=42)
model_gold_crisis.fit(crisis_train[features], crisis_train[target_gold])

# Silver (NEW: regime switching)
model_silver_normal = GradientBoostingRegressor(random_state=42)
model_silver_normal.fit(train[features], train[target_silver])

crisis_train_silver = train[train['Regime'] == crisis_regime]
model_silver_crisis = GradientBoostingRegressor(random_state=42)
model_silver_crisis.fit(crisis_train_silver[features], crisis_train_silver[target_silver])

print("✅ Models Ready")

# ===============================
# SCENARIO GENERATOR
# ===============================
def generate_scenario(last_usdinr):
    return {
        'Fed': float(np.clip(np.random.normal(2.0,1.0),0,10)),
        'Inflation': float(np.clip(np.random.normal(3.0,1.0),0.5,12)),
        'US10Y': float(np.clip(np.random.normal(3.0,1.0),0.1,8)),
        'DXY': float(np.clip(np.random.normal(100,5),80,120)),
        'VIX': float(np.clip(np.random.normal(20,5),9,80)),
        'FSI': float(np.clip(np.random.normal(0.5,0.5),-2,3)),
        'Spread': float(np.clip(np.random.normal(1.0,0.5),-1,5)),
        'RealYield': float(np.clip(np.random.normal(1.0,1.0),-3,5)),
        'USDINR': float(last_usdinr * (1 + np.random.normal(0,0.01)))
    }

# ===============================
# ROUTES
# ===============================
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/forecast', methods=['POST'])
def forecast():
    req = request.get_json()
    n_months = int(req.get("months", 6))
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
                [scenario[v] for v in regime_vars],
                index=regime_vars
            ).T

            regime = gmm.predict(scaler.transform(regime_input))[0]

            # Gold regime switching
            if regime == crisis_regime:
                g_ret = model_gold_crisis.predict(row)[0]
            else:
                g_ret = model_gold_normal.predict(row)[0]

            # Silver regime switching (NEW)
            if regime == crisis_regime:
                s_ret = model_silver_crisis.predict(row)[0]
            else:
                s_ret = model_silver_normal.predict(row)[0]

            g *= (1 + g_ret)
            s *= (1 + s_ret)

            mc_gold[t, sim] = g
            mc_silver[t, sim] = s
            mc_usdinr[t, sim] = usdinr

    # ===============================
    # OUTPUT
    # ===============================
    gold_mean = mc_gold.mean(axis=1)
    silver_mean = mc_silver.mean(axis=1)
    usdinr_mean = mc_usdinr.mean(axis=1)

    dates = pd.date_range(
        start=data.index[-1] + pd.offsets.MonthEnd(1),
        periods=n_months,
        freq='ME'
    )

    result = []
    for i in range(n_months):
        result.append({
            "date": str(dates[i]),
            "gold_usd": float(gold_mean[i]),
            "silver_usd": float(silver_mean[i]),
            "usdinr": float(usdinr_mean[i])
        })

    return jsonify(result)

# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
