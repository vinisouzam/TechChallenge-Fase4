# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

# %%
# --- Load data ---
def load_data(path='./data/brent.csv'):
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.sort_values('date')
    df = df.set_index('date').asfreq('D')
    df['value'] = df['value'].interpolate()
    return df

# %%
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
    df_feat = df_feat.dropna()
    df_feat.drop(['DATE','RAW DATE','CODE'],axis=1, inplace=True)
    return df_feat

# %%

# --- Model evaluation ---
def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, ) # squared=False
    }

# %%
# --- Grid search training ---
def train_model(model, param_grid, X_train, y_train):
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_absolute_error')
    grid.fit(X_train, y_train)
    return grid

# %%
# --- Full pipeline ---
def run_pipeline():
    df = load_data()
    df_feat = create_features(df)

    # Split train/test
    split_date = '2023-01-01'
    train = df_feat[df_feat.index < split_date]
    test = df_feat[df_feat.index >= split_date]

    X_train = train.drop(columns=['value'])
    y_train = train['value']
    X_test = test.drop(columns=['value'])
    y_test = test['value']

    results = {}

    # --- Random Forest ---
    rf = RandomForestRegressor()
    rf_params = {'n_estimators': [50, 100], 'max_depth': [3, 5, 10]}
    rf_model = train_model(rf, rf_params, X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    results['RandomForest'] = {'model': rf_model, 'metrics': evaluate(y_test, rf_pred)}

    # --- HistGradientBoosting ---
    gb = HistGradientBoostingRegressor()
    gb_params = {'learning_rate': [0.05, 0.1], 'max_iter': [100, 200]}
    gb_model = train_model(gb, gb_params, X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    results['HistGradientBoost'] = {'model': gb_model, 'metrics': evaluate(y_test, gb_pred)}

    # --- Report Results ---
    print("\n📊 Model Evaluation:")
    for name, res in results.items():
        print(f"{name}: MAE = {res['metrics']['MAE']:.2f}, RMSE = {res['metrics']['RMSE']:.2f}")

    # --- Save best model manually ---
    best = input("\nEnter best model name to save (RandomForest / HistGradientBoost): ").strip()
    joblib.dump(results[best]['model'], 'models/best_model.pkl')
    print(f"✅ Saved {best} to models/best_model.pkl")


# %%
if __name__ == "__main__":
    os.makedirs("./models", exist_ok=True)
    run_pipeline()

# %%
