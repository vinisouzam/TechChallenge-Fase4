# %%
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from scipy.stats import zscore
import os

# %%

caminho = '../data/brent.csv'

# %%

df = pd.read_csv(caminho)
df['date'] = pd.to_datetime(df['DATE'])
df = df.sort_values('date')

# %%
# Check for missing values
if df.isnull().sum().any():
    print("Warning: The dataset contains missing values.")
    print(df.isnull().sum())

# %%
# Check for outliers using z-score

df['z_score'] = zscore(df['value'])
outliers = df[(df['z_score'] > 3) | (df['z_score'] < -3)]
if not outliers.empty:
    print("Warning: The dataset contains outliers.")
    print(outliers)

# %%
# Check for stationarity using Augmented Dickey-Fuller test

adf_test = adfuller(df['value'].dropna())
if adf_test[1] > 0.05:
    print("""Warning: The dataset is not stationary.
          Consider differencing or detrending.""")
else:
    print("The dataset is stationary.")


# %%