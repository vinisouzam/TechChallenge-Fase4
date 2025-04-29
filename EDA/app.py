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

rolling_mean = data_nonan['VALUE_US$'].rolling(window=45).mean()
rolling_std = data_nonan['VALUE_US$'].rolling(window=45).std()
rolling_var = data_nonan['VALUE_US$'].rolling(window=45).var()

fig = go.Figure()
fig.add_trace(go.Scatter(x=data_nonan.index, 
                         y=data_nonan['VALUE_US$'], 
                         mode='lines', 
                         name='Original')
              )
fig.add_trace(go.Scatter(x=data_nonan.index, 
                         y=rolling_mean, 
                         mode='lines', 
                         name='Rolling Mean')
              )
fig.add_trace(go.Scatter(x=data_nonan.index, 
                         y=rolling_std, 
                         mode='lines', 
                         name='Rolling Std')
              )
fig.add_trace(go.Scatter(x=data_nonan.index, 
                         y=rolling_var, 
                         mode='lines', 
                         name='Rolling Var'))
fig.update_layout(
    title='Rolling Mean & Standard Deviation',
    xaxis_title='Date',
    yaxis_title='Brent Crude Oil Price (US$)',
    legend_title='Legend',
    title_x=0.5,
)
fig.show()
# %%
fig.write_html('rolling_mean_std.html')

# %%

data_nonan.loc[slice(None),'change'] = data_nonan['VALUE_US$'].pct_change().fillna(0) +1
# %%
