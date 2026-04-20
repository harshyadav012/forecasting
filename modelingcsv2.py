import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor

from Data_set_creation import create_dataset
from regime_detection import detect_regimes

print("🚀 Script Started")

# ===============================
# 1️⃣ LOAD DATA
# ===============================
data = create_dataset()
data.index = pd.to_datetime(data.index)
data = data.sort_index()

# ===============================
# 2️⃣ FEATURES
# ===============================
features = ['Fed', 'Inflation', 'US10Y', 'DXY', 'VIX', 'FSI', 'Spread', 'RealYield', 'USDINR']
target_gold = 'Gold_Return'
target_silver = 'Silver_Return'

data = data.dropna(subset=features + [target_gold, target_silver])

# ===============================
# 3️⃣ REGIME DETECTION
# ===============================
regime_vars = ['VIX', 'FSI', 'Spread', 'Fed', 'Inflation', 'USDINR']
data, scaler, gmm = detect_regimes(data, regime_vars)

# Train / test split
train = data[data.index.year <= 2020].copy()
test = data[data.index.year >= 2021].copy()

# Identify which GMM cluster corresponds to crisis (highest mean VIX)
crisis_regime = train.groupby('Regime')['VIX'].mean().idxmax()

# ===============================
# 4️⃣ MODELS
# ===============================
model_normal = GradientBoostingRegressor(random_state=42)
model_normal.fit(train[features], train[target_gold])

crisis_train = train[train['Regime'] == crisis_regime]
model_crisis = GradientBoostingRegressor(random_state=42)
model_crisis.fit(crisis_train[features], crisis_train[target_gold])

model_silver = GradientBoostingRegressor(random_state=42)
model_silver.fit(train[features], train[target_silver])

# ===============================
# 5️⃣ SCENARIO GENERATOR
# ===============================
def generate_scenario(last_usdinr: float) -> dict:
    return {
        'Fed': float(np.clip(np.random.normal(2.0, 1.0), 0.0, 10.0)),
        'Inflation': float(np.clip(np.random.normal(3.0, 1.0), 0.5, 12.0)),
        'US10Y': float(np.clip(np.random.normal(3.0, 1.0), 0.1, 8.0)),
        'DXY': float(np.clip(np.random.normal(100.0, 5.0), 80.0, 120.0)),
        'VIX': float(np.clip(np.random.normal(20.0, 5.0), 9.0, 80.0)),
        'FSI': float(np.clip(np.random.normal(0.5, 0.5), -2.0, 3.0)),
        'Spread': float(np.clip(np.random.normal(1.0, 0.5), -1.0, 5.0)),
        'RealYield': float(np.clip(np.random.normal(1.0, 1.0), -3.0, 5.0)),
        'USDINR': float(last_usdinr * (1.0 + np.random.normal(0.0, 0.01))),
    }

# ===============================
# 6️⃣ MONTE CARLO
# ===============================
def monte_carlo_forecast(n_months: int = 6, n_simulations: int = 300):
    """
    Run Monte Carlo price-path simulations for Gold, Silver, and USDINR.

    Returns
    -------
    forecast_gold   : DataFrame
    forecast_silver : DataFrame
    mc_gold         : ndarray shape (n_months, n_simulations)
    mc_silver       : ndarray shape (n_months, n_simulations)
    mc_usdinr       : ndarray shape (n_months, n_simulations)
    """
    print(f"\n📊 Running Monte Carlo ({n_simulations} sims × {n_months} months)…")

    mc_gold = np.zeros((n_months, n_simulations))
    mc_silver = np.zeros((n_months, n_simulations))
    mc_usdinr = np.zeros((n_months, n_simulations))

    last_gold = float(data['Gold'].iloc[-1])
    last_silver = float(data['Silver'].iloc[-1])
    last_usdinr = float(data['USDINR'].iloc[-1])

    for sim in range(n_simulations):
        price_gold = last_gold
        price_silver = last_silver
        usdinr = last_usdinr

        for t in range(n_months):
            scenario = generate_scenario(usdinr)
            usdinr = scenario['USDINR']

            row = pd.DataFrame([{feat: scenario[feat] for feat in features}])

            regime_input = pd.DataFrame(
                [[scenario[v] for v in regime_vars]],
                columns=regime_vars
            )
            regime = gmm.predict(scaler.transform(regime_input))[0]

            if regime == crisis_regime:
                gold_ret = model_crisis.predict(row[features])[0]
            else:
                gold_ret = model_normal.predict(row[features])[0]

            silver_ret = model_silver.predict(row[features])[0]

            price_gold *= (1.0 + gold_ret)
            price_silver *= (1.0 + silver_ret)

            mc_gold[t, sim] = price_gold
            mc_silver[t, sim] = price_silver
            mc_usdinr[t, sim] = usdinr

    # ── Build output DataFrames ──────────────────────────
    future_dates = pd.date_range(
        start=data.index[-1] + pd.offsets.MonthEnd(1),
        periods=n_months,
        freq='ME',
    )

    OZ_TO_GRAMS = 31.1035
    mean_usdinr = mc_usdinr.mean(axis=1)
    p10_usdinr = np.percentile(mc_usdinr, 10, axis=1)
    p90_usdinr = np.percentile(mc_usdinr, 90, axis=1)

    # Gold forecast
    forecast_gold = pd.DataFrame(index=future_dates)
    forecast_gold['USDINR'] = mean_usdinr
    forecast_gold['USDINR_p10'] = p10_usdinr
    forecast_gold['USDINR_p90'] = p90_usdinr
    forecast_gold['Gold_USD_per_oz'] = mc_gold.mean(axis=1)
    forecast_gold['Gold_USD_p10'] = np.percentile(mc_gold, 10, axis=1)
    forecast_gold['Gold_USD_p90'] = np.percentile(mc_gold, 90, axis=1)
    forecast_gold['Gold_INR_10g'] = (
        forecast_gold['Gold_USD_per_oz'] / OZ_TO_GRAMS * forecast_gold['USDINR'] * 10
    )

    # Silver forecast
    forecast_silver = pd.DataFrame(index=future_dates)
    forecast_silver['USDINR'] = mean_usdinr
    forecast_silver['USDINR_p10'] = p10_usdinr
    forecast_silver['USDINR_p90'] = p90_usdinr
    forecast_silver['Silver_USD_per_oz'] = mc_silver.mean(axis=1)
    forecast_silver['Silver_USD_p10'] = np.percentile(mc_silver, 10, axis=1)
    forecast_silver['Silver_USD_p90'] = np.percentile(mc_silver, 90, axis=1)
    forecast_silver['Silver_INR_kg'] = (
        forecast_silver['Silver_USD_per_oz'] / OZ_TO_GRAMS * forecast_silver['USDINR'] * 1000
    )

    print("\n📈 GOLD FORECAST:")
    print(forecast_gold.round(2))
    print("\n🪙 SILVER FORECAST:")
    print(forecast_silver.round(2))

    return forecast_gold, forecast_silver, mc_gold, mc_silver, mc_usdinr

# ===============================
# 7️⃣ JSON HELPER FOR WEB/FLASK
# ===============================
def build_forecast_payload(forecast_gold: pd.DataFrame,
                           forecast_silver: pd.DataFrame) -> list[dict]:
    payload = []

    for dt in forecast_gold.index:
        payload.append({
            "date": dt.strftime("%Y-%m-%d"),
            "usdinr": float(forecast_gold.loc[dt, "USDINR"]),
            "gold_usd": float(forecast_gold.loc[dt, "Gold_USD_per_oz"]),
            "gold_usd_p10": float(forecast_gold.loc[dt, "Gold_USD_p10"]),
            "gold_usd_p90": float(forecast_gold.loc[dt, "Gold_USD_p90"]),
            "gold_inr": float(forecast_gold.loc[dt, "Gold_INR_10g"]),
            "silver_usd": float(forecast_silver.loc[dt, "Silver_USD_per_oz"]),
            "silver_usd_p10": float(forecast_silver.loc[dt, "Silver_USD_p10"]),
            "silver_usd_p90": float(forecast_silver.loc[dt, "Silver_USD_p90"]),
            "silver_inr": float(forecast_silver.loc[dt, "Silver_INR_kg"]),
        })

    return payload

# ===============================
# 8️⃣ PLOTTING
# ===============================
def plot_forecasts(hist_data: pd.DataFrame,
                   f_gold: pd.DataFrame,
                   f_silver: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        hist_data.index[-24:],
        hist_data['Gold'].iloc[-24:],
        label='Historical Gold',
        color='black',
        linewidth=2
    )
    ax.plot(
        f_gold.index,
        f_gold['Gold_USD_per_oz'],
        label='Forecast (mean)',
        color='#D4AF37',
        linestyle='--',
        marker='o'
    )

    if 'Gold_USD_p10' in f_gold.columns:
        ax.fill_between(
            f_gold.index,
            f_gold['Gold_USD_p10'],
            f_gold['Gold_USD_p90'],
            color='#D4AF37',
            alpha=0.15,
            label='10–90th pct band'
        )

    ax.set_title('Gold Price Forecast (USD / oz)')
    ax.legend()
    fig.tight_layout()
    fig.savefig('gold_plot.png', dpi=150)
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        hist_data.index[-24:],
        hist_data['Silver'].iloc[-24:],
        label='Historical Silver',
        color='black',
        linewidth=2
    )
    ax.plot(
        f_silver.index,
        f_silver['Silver_USD_per_oz'],
        label='Forecast (mean)',
        color='#C0C0C0',
        linestyle='--',
        marker='o'
    )

    if 'Silver_USD_p10' in f_silver.columns:
        ax.fill_between(
            f_silver.index,
            f_silver['Silver_USD_p10'],
            f_silver['Silver_USD_p90'],
            color='#C0C0C0',
            alpha=0.15,
            label='10–90th pct band'
        )

    ax.set_title('Silver Price Forecast (USD / oz)')
    ax.legend()
    fig.tight_layout()
    fig.savefig('silver_plot.png', dpi=150)
    plt.show()

# ===============================
# 9️⃣ ENTRY POINT
# ===============================
if __name__ == "__main__":
    n = int(input("Enter months to forecast (1–36): "))
    assert 1 <= n <= 36, "Please enter a value between 1 and 36."

    forecast_gold, forecast_silver, _, _, _ = monte_carlo_forecast(n_months=n)

    forecast_gold.to_csv("forecast_gold.csv")
    forecast_silver.to_csv("forecast_silver.csv")
    print("✅ Forecast saved to forecast_gold.csv / forecast_silver.csv")

    plot_forecasts(data, forecast_gold, forecast_silver)
