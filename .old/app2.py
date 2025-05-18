import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

# --- Load data and model ---
@st.cache_data
def load_data():
    df = pd.read_csv('./data/brent.csv', parse_dates=['date'])
    df = df.set_index('date').asfreq('D')
    df['value'] = df['value'].interpolate()
    return df

@st.cache_resource
def load_model():
    return joblib.load('models/best_model.pkl')

# --- Feature engineering ---
def create_features(df):
    df_feat = df.copy()
    df_feat['lag_1'] = df_feat['value'].shift(1)
    df_feat['lag_7'] = df_feat['value'].shift(7)
    df_feat['lag_30'] = df_feat['value'].shift(30)
    df_feat['rolling_mean_7'] = df_feat['value'].rolling(7).mean()
    df_feat['rolling_mean_30'] = df_feat['value'].rolling(30).mean()
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    df_feat.drop(['DATE', 'RAW DATE', 'CODE'], axis=1, inplace=True)
    return df_feat.dropna()

# --- Forecasting logic ---
def forecast(df, model, n_days):
    last_df = df.copy()
    predictions = []

    for _ in range(n_days):
        # Generate features for the last row in the current data
        features = create_features(last_df)[-1:]  # latest features
        X_pred = features.drop(columns=['value'])  # Predicting using the features
        y_pred = model.predict(X_pred)[0]  # Model predicts the next value
        
        # Create the next date and add the prediction
        next_date = last_df.index[-1] + timedelta(days=1)
        next_row = pd.DataFrame({'value': [y_pred]}, index=[next_date])
        
        # Append the predicted value to the dataframe
        last_df = pd.concat([last_df, next_row])

        # Store the prediction
        predictions.append((next_date, y_pred))

    return pd.DataFrame(predictions, columns=['date', 'forecast']).set_index('date')

# --- Streamlit UI ---
st.set_page_config(page_title="Brent Forecast", layout="centered")
st.title("📈 Brent Oil Price Forecast")

# Load data and model
df = load_data()
model = load_model()

# Slider to select forecast range
n_days = st.slider("📆 How many days ahead to forecast?", min_value=1, max_value=90, value=30)

# Run forecast
forecast_df = forecast(df, model, n_days)

# Show forecasted values
st.subheader(f"🔮 Forecast for next {n_days} days")
st.line_chart(forecast_df)

# Show actual recent prices
st.subheader("📊 Last 90 Days of Actual Prices")
st.line_chart(df[['value']].tail(90))

# Combined view (Actual + Forecast)
st.subheader("📉 Combined: Actual + Forecast")
# Select only the 'value' column for the actual prices and 'forecast' column for the forecasted values
combined = pd.concat([df[['value']].tail(30), forecast_df[['forecast']]])
st.line_chart(combined)

# Download forecast as CSV
st.download_button("⬇️ Download Forecast", forecast_df.reset_index().to_csv(index=False), file_name="brent_forecast.csv")
