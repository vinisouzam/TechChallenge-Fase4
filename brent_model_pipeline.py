# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

RANDOM_STATE = 42

# ========================
# 1. Carrega os dados
# ========================
def load_data(path='data/brent.csv'):
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.sort_values('date').set_index('date').asfreq('D')
    df['value'] = df['value'].interpolate()
    df.drop(['DATE', 'RAW DATE', 'CODE'], axis=1, inplace=True, errors='ignore')
    return df

# ========================
# 2. Cria novas features
# ========================
def create_features(df):
    df_feat = df.copy()
    # Lags
    for lag in [1, 3, 7, 14, 30, 45, 60, 90]:
        df_feat[f'lag_{lag}'] = df_feat['value'].shift(lag)
    # Médias móveis
    for window in [3, 7, 14, 30, 45, 60, 90]:
        df_feat[f'rolling_mean_{window}'] = df_feat['value'].rolling(window).mean()
    # Desvio padrão e média exponencial
    df_feat['rolling_std_7'] = df_feat['value'].rolling(7).std()
    df_feat['exp_moving_avg_7'] = df_feat['value'].ewm(span=7).mean()
    # Calendário
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    df_feat['quarter'] = df_feat.index.quarter
    df_feat['dayofyear'] = df_feat.index.dayofyear
    df_feat['is_month_start'] = df_feat.index.is_month_start.astype(int)
    df_feat['is_month_end'] = df_feat.index.is_month_end.astype(int)
    # Tendência temporal
    df_feat['trend'] = np.arange(len(df_feat))
    df_feat = df_feat.dropna()
    
    return df_feat

# ========================
# 3. Avalia resultados
# ========================
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse}

# ========================
# 4. Forecast recursivo
# ========================
def recursive_forecast(model, last_data, steps=30):
    forecast = []
    current = last_data.copy()

    for _ in range(steps):
        last_row = current.iloc[-1:]
        new_date = last_row.index[0] + pd.Timedelta(days=1)
        new_row = pd.DataFrame(index=[new_date])

        for lag in [1, 3, 7, 14, 30, 45, 60, 90]:
            if len(current) >= lag:
                new_row[f'lag_{lag}'] = current['value'].iloc[-lag]
            else:
                new_row[f'lag_{lag}'] = np.nan

        for window in [3, 7, 14, 30]:
            if len(current) >= window:
                new_row[f'rolling_mean_{window}'] = current['value'].iloc[-window:].mean()
            else:
                new_row[f'rolling_mean_{window}'] = np.nan

        new_row['rolling_std_7'] = current['value'].iloc[-7:].std() if len(current) >= 7 else np.nan
        new_row['exp_moving_avg_7'] = current['value'].ewm(span=7).mean().iloc[-1] if len(current) >= 7 else np.nan
        new_row['dayofweek'] = new_date.dayofweek
        new_row['month'] = new_date.month
        new_row['quarter'] = (new_date.month - 1) // 3 + 1
        new_row['dayofyear'] = new_date.dayofyear
        new_row['is_month_start'] = int(new_date.is_month_start)
        new_row['is_month_end'] = int(new_date.is_month_end)
        new_row['trend'] = current['trend'].iloc[-1] + 1

        if new_row.isna().any().any():
            print("Interrompido: NaNs nas features geradas.")
            break

        y_pred = model.predict(new_row)[0]
        new_row['value'] = y_pred
        current = pd.concat([current, new_row])
        forecast.append((new_date, y_pred))

    forecast_df = pd.DataFrame(forecast, columns=['date', 'predicted_value']).set_index('date')
    return forecast_df

# ========================
# 5. Pipeline final
# ========================
def run_pipeline(df_feat):
    train = df_feat.iloc[:-90]
    test = df_feat.iloc[-90:]
    X_train = train.drop(columns=['value'])
    y_train = train['value']
    X_test = test.drop(columns=['value'])
    y_test = test['value']

    models = {
        "RandomForest": (RandomForestRegressor(random_state=RANDOM_STATE), {'n_estimators': [100]}),
        "HistGradientBoost": (HistGradientBoostingRegressor(random_state=RANDOM_STATE), {'learning_rate': [0.1]}),
        "LinearRegression": (LinearRegression(), {}),
        "Ridge": (Pipeline([('scaler', StandardScaler()), ('ridge', Ridge())]), {'ridge__alpha': [1.0]}),
        "KNeighbors": (Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor())]), {'knn__n_neighbors': [5]}),
        "SVR": (Pipeline([('scaler', StandardScaler()), ('svr', SVR())]), {'svr__C': [1]})
    }

    results = {}
    for name, (model, param_grid) in models.items():
        print(f"🔍 Treinando {name}...")
        try:
            tscv = TimeSeriesSplit(n_splits=5)
            grid = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
            grid.fit(X_train, y_train)
            pred = grid.predict(X_test)
            metrics = evaluate(y_test, pred)
            results[name] = {"model": grid, "metrics": metrics}
        except Exception as e:
            print(f" {name} falhou: {e}")

    print("\n Avaliação dos modelos:")
    for name, res in results.items():
        print(f"{name}: MAE = {res['metrics']['MAE']:.4f}, RMSE = {res['metrics']['RMSE']:.4f}")

    # Baseline
    naive_pred = X_test['lag_1']
    naive_metrics = evaluate(y_test, naive_pred)
    print(f"Naive: MAE = {naive_metrics['MAE']:.4f}, RMSE = {naive_metrics['RMSE']:.4f}")
    results['Naive'] = {"model": None, "metrics": naive_metrics}

    # Melhor modelo
    best_name = min(results, key=lambda k: results[k]['metrics']['MAE'])
    best_model = results[best_name]['model']
    print(f"\n 🏆 Melhor modelo: {best_name}")

    os.makedirs("models", exist_ok=True)
    result_df = pd.DataFrame([
        {"Model": name, "MAE": res['metrics']['MAE'], "RMSE": res['metrics']['RMSE']}
        for name, res in results.items()
    ])
    result_df.to_csv("models/model_evaluation.csv", index=False)

    if best_model:
        joblib.dump(best_model, "models/best_model.pkl")
        print("✅ Modelo salvo em models/best_model.pkl")

        forecast_df = recursive_forecast(best_model, df_feat.iloc[-60:], steps=30)
        forecast_df.to_csv("models/forecast_30d.csv")
        print("📈 Previsão futura salva em models/forecast_30d.csv")

# ========================
# Execução
# ========================
# %%

df = load_data()
# %%

df_feat = create_features(df)

# %%
run_pipeline(df_feat)
