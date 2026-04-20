from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import pandas as pd

# ===============================
# SPLIT FUNCTION
# ===============================
def split_data(data, split_ratio=0.8):
    split_index = int(len(data) * split_ratio)
    return data.iloc[:split_index].copy(), data.iloc[split_index:].copy()

# ===============================
# REGIME DETECTION
# ===============================
def detect_regimes(data, regime_vars, n_components=2, split_ratio=0.8):

    train, test = split_data(data, split_ratio)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(train[regime_vars])
    X_test = scaler.transform(test[regime_vars])

    gmm = GaussianMixture(n_components=n_components, random_state=42)

    train['Regime'] = gmm.fit_predict(X_train)
    test['Regime'] = gmm.predict(X_test)

    final_data = pd.concat([train, test]).sort_index()

    return final_data, scaler, gmm
