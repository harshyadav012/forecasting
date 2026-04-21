from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import pandas as pd

def split_data(data, split_ratio=0.8):
    split_index = int(len(data) * split_ratio)
    return data.iloc[:split_index].copy(), data.iloc[split_index:].copy()

def detect_regimes(data, regime_vars, n_components=2, split_ratio=0.8):
    if data.empty:
        raise ValueError("detect_regimes received an empty dataset.")

    missing_cols = [col for col in regime_vars if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing regime columns: {missing_cols}")

    data = data.dropna(subset=regime_vars).copy()
    if data.empty:
        raise ValueError("No rows left after dropping NA values for regime variables.")

    train, test = split_data(data, split_ratio)
    if train.empty:
        raise ValueError("Training split is empty in detect_regimes.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[regime_vars])

    gmm = GaussianMixture(n_components=n_components, random_state=42)
    train["Regime"] = gmm.fit_predict(X_train)

    if not test.empty:
        X_test = scaler.transform(test[regime_vars])
        test["Regime"] = gmm.predict(X_test)
        final_data = pd.concat([train, test]).sort_index()
    else:
        final_data = train.sort_index()

    return final_data, scaler, gmm
