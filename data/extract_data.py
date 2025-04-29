# %%
import ipeadatapy as ipea
import pandas as pd

# %%
df = ipea.timeseries(series='EIA366_PBRENT366').reset_index()
df['date'] = pd.to_datetime(df['DATE'],utc=True)
df['value'] = df['VALUE (US$)']

# %%

df.to_csv('brent.csv', index=False)
# %%
