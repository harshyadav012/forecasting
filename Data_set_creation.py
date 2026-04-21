import pandas as pd
import numpy as np
from fredapi import Fred
import yfinance as yf
import os
import time

# ===============================
# CONFIG
# ===============================
start_date = '2005-01-01'

fred_api_key = os.getenv("FRED_API_KEY")
if not fred_api_key:
    raise ValueError("FRED_API_KEY environment variable is missing.")

fred = Fred(api_key=fred_api_key)


# ===============================
# FETCH FRED DATA WITH RETRIES
# ===============================
def get_fred(series, name, retries=5, delay=2):
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            print(f"📥 Fetching FRED series {series} as {name} (attempt {attempt}/{retries})")

            data = fred.get_series(series, start_date=start_date)

            if data is None or len(data) == 0:
                raise ValueError(f"FRED series {series} returned empty data.")

            df = data.to_frame(name)
            df.index = pd.to_datetime(df.index)

            df = df.dropna()
            if df.empty:
                raise ValueError(f"FRED series {series} became empty after dropna().")

            return df.resample('ME').mean()

        except Exception as e:
            last_error = e
            print(f"⚠️ FRED fetch failed for {series}: {e}")

            if attempt < retries:
                time.sleep(delay)

    raise RuntimeError(f"FRED FAILED after {retries} attempts for series {series}: {last_error}")


# ===============================
# FETCH MARKET DATA WITH RETRIES
# ===============================
def get_price(ticker, name, retries=5, delay=2):
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            print(f"📈 Fetching Yahoo ticker {ticker} as {name} (attempt {attempt}/{retries})")

            df = yf.download(
                ticker,
                start=start_date,
                auto_adjust=True,
                progress=False
            )

            if df is None or df.empty:
                raise ValueError(f"Yahoo Finance ticker {ticker} returned empty data.")

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if 'Close' in df.columns:
                price = df['Close']
            else:
                price = df.iloc[:, 0]

            out = price.to_frame(name).dropna()

            if out.empty:
                raise ValueError(f"Ticker {ticker} has no usable price data after dropna().")

            return out.resample('ME').mean()

        except Exception as e:
            last_error = e
            print(f"⚠️ Yahoo Finance fetch failed for {ticker}: {e}")

            if attempt < retries:
                time.sleep(delay)

    raise RuntimeError(f"YFINANCE FAILED after {retries} attempts for ticker {ticker}: {last_error}")


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
    ], axis=1)

    print("Rows before fill/dropna:", len(data))
    print("Columns in merged dataset:", list(data.columns))

    # Small-gap handling
    data = data.sort_index().ffill()

    # Final clean rows only
    data = data.dropna()

    if data.empty:
        raise ValueError("Merged dataset is empty after ffill() and dropna().")

    # Returns
    data['Gold_Return'] = data['Gold'].pct_change()
    data['Silver_Return'] = data['Silver'].pct_change()

    data = data.dropna()

    if data.empty:
        raise ValueError("Dataset is empty after calculating returns.")

    print("✅ Final dataset rows:", len(data))
    print("✅ Final dataset latest date:", data.index[-1])

    return data
