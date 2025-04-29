# %%

import ipeadatapy as ipea
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# %%
# Load data
df = ipea.timeseries(series='EIA366_PBRENT366').reset_index()
df['date'] = pd.to_datetime(df['DATE'])
df = df.sort_values('date')

# %%

# Smooth with 30-day moving average
df['value'] = df['VALUE (US$)']
df['rolling_mean'] = df['value'].rolling(window=30, center=True).mean()
df = df.dropna(subset=['rolling_mean'])

# %%

# Detect local maxima/minima using rolling_mean
order = 30
max_idx = argrelextrema(df['rolling_mean'].values, 
                        np.greater_equal, 
                        order=order
                        )[0]
min_idx = argrelextrema(df['rolling_mean'].values, 
                        np.less_equal, 
                        order=order
                        )[0]

# %%
# Create empty columns and assign extrema points
df['max'] = np.nan
df['min'] = np.nan
df.loc[df.iloc[max_idx].index, 'max'] = df.iloc[max_idx]['rolling_mean']
df.loc[df.iloc[min_idx].index, 'min'] = df.iloc[min_idx]['rolling_mean']

# %%
# Get most relevant turning points
top_max = df.dropna(subset=['max']).nlargest(4, 'max')
top_min = df.dropna(subset=['min']).nsmallest(4, 'min')
turning_points = pd.concat([top_max, top_min]).sort_values('date')


# %%
# Explain the points
for _, row in turning_points.iterrows():
    kind = "peak" if not pd.isna(row['max']) else "trough"
    price = row['rolling_mean']
    print(f"On {row['date'].date()
          }, a {kind} occurred with Brent price around ${price:.2f}.")


# %%

"""
| Date         | Type   | Price (US$) | Notes |
|--------------|--------|-------------|-------|
| 2002-06-24   | Trough | $24.44      | Early Iraq War buildup / low demand|
| 2002-11-18   | Trough | $24.31      | Similar period, still pre-Iraq War|
| 2003-04-28   | Trough | $24.43      | End of major combat in Iraq|
| 2008-07-08   | Peak   | $136.51     | Major spike before 2008 crash|
| 2011-04-14   | Peak   | $121.80     | Arab Spring / Libya oil collapse|
| 2012-03-16   | Peak   | $125.33     | Iran sanctions and geopolitical tension|
| 2020-04-14   | Trough | $19.29      | COVID demand crash|
| 2022-06-14   | Peak   | $121.83     | Ukraine war, tight supply|
"""
# %%
