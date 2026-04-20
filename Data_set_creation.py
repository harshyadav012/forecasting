import pandas as pd
import numpy as np
from fredapi import Fred
import yfinance as yf
import os

# ===============================
# CONFIG
# ===============================
start_date = '2005-01-01'

fred = Fred(api_key=os.getenv("FRED_API_KEY"))

# ===============================
# FETCH FRED DATA
# ===============================
def get_fred(series, name):
    df = fred.get_series(series, start_date=start_date).to_frame(name)
    df.index = pd.to_datetime(df.index)
    return df.resample('ME').mean()

# ===============================
# FETCH MARKET DATA
# ===============================
def get_price(ticker, name):
    df = yf.download(ticker, start=start_date, auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    price = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]

    return price.to_frame(name).resample('ME').mean()

# ===============================
# MAIN DATASET FUNCTION
# ===============================
def create_dataset():

    # Macro
    fed = get_fred('FEDFUNDS', 'Fed')
    cpi = get_fred('CPIAUCSL', 'CPI')
    us10y = get_fred('DGS10', 'US10Y')
    us2y = get_fred('DGS2', 'US2Y')
    dxy = get_fred('DTWEXBGS', 'DXY')
    vix = get_fred('VIXCLS', 'VIX')
    fsi = get_fred('STLFSI4', 'FSI')

    # Features
    inflation = cpi.pct_change(12) * 100
    inflation.columns = ['Inflation']

    spread = (us10y['US10Y'] - us2y['US2Y']).to_frame('Spread')
    real_yield = (us10y['US10Y'] - inflation['Inflation']).to_frame('RealYield')

    # Prices
    gold = get_price('GC=F', 'Gold')
    silver = get_price('SI=F', 'Silver')
    usd_inr = get_price('USDINR=X', 'USDINR')

    # Merge
    data = pd.concat([
        gold, silver,
        fed, inflation, us10y, dxy,
        vix, fsi, spread, real_yield,
        usd_inr
    ], axis=1).dropna()

    # Returns
    data['Gold_Return'] = data['Gold'].pct_change()
    data['Silver_Return'] = data['Silver'].pct_change()

    return data.dropna()
