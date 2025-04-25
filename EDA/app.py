# %%

# importing necessary libraries
import pandas as pd
import ipeadatapy as ipea
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt


# %%
# getting the data from ipead
data = ipea.timeseries(series='EIA366_PBRENT366')

# %%
# Starting analysis from the data retrieved

# checking the data types
data.info()

# %%
# checking nullable values
data.isnull().sum()

# %%

# adjusting column names
data.columns = data.columns.str.replace(' ', '_')
data.columns = data.columns.str.replace('(', '')
data.columns = data.columns.str.replace(')', '')

# data['RAW_DATE'] = pd.to_datetime(data['RAW_DATE'],utc=True)

# %%
# getting the first 5 rows of the data
data.head()

# %% 
# getting the last 5 rows of the data
data.tail()

# %%
data.describe().T

# %%
fig = px.line(
    data_frame=data, 
    x=data.index,
    y='VALUE_US$',
    title= "Brent Crude Oil Price per barrel over time",
    labels={
        'VALUE_US$': 'Brent Crude Oil Price (US$)',
        'DATE':'Date'
        },
    )

fig.update_layout(
    title_x=0.5,
)

fig.show()
# %%

data_nonan = data.dropna()

# Decomposing the time series
decomposition = seasonal_decompose(
    data_nonan['VALUE_US$'],
    model='additive', 
    period=12
    )

# Plotting the decomposed components
fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
decomposition.observed.plot(ax=axes[0], title='Observed')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual')

plt.suptitle('Time Series Decomposition', fontsize=16)
plt.show()

# %%
# Checking stationarity using rolling statistics

rolling_mean = data_nonan['VALUE_US$'].rolling(window=90).mean()
rolling_std = data_nonan['VALUE_US$'].rolling(window=90).std()

# def great_change(window_data):
#     return window_data.max() - window_data.min()
# df_great_change = data_nonan['VALUE_US$'].rolling(
    # window=90).apply(great_change)

plt.figure(figsize=(10, 6))
plt.plot(data_nonan['VALUE_US$'], label='Original', color='blue')
plt.plot(rolling_mean, label='Rolling Mean', color='red')
plt.plot(rolling_std, label='Rolling Std', color='black')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show()
# %%
