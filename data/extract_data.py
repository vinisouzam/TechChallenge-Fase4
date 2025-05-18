# %%
print('Extracting data from IPEA')
import ipeadatapy as ipea
import pandas as pd

# %%
df = ipea.timeseries(series='EIA366_PBRENT366').reset_index()
df['date'] = pd.to_datetime(df['DATE'],utc=True)
df['value'] = df['VALUE (US$)']
df.drop(['DATE', 'RAW DATE', 'CODE','DAY','MONTH','VALUE (US$)','YEAR'],axis=1, inplace=True)

# %%

df.to_csv('./data/brent.csv', index=False)
# %%

print('Extracted data from IPEA and saved to brent.csv')