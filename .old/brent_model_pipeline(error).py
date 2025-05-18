# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

RANDOM_STATE = 42
pd.options.display.max_columns = 100

# %%
# 1. Carrega dados
def load_data(path='data/brent.csv'):
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.sort_values('date').set_index('date').asfreq('D')
    df['value'] = df['value'].interpolate()
    df.drop(['DATE', 'RAW DATE', 'CODE'], axis=1, inplace=True, errors='ignore')
    return df

# %%
# 2. Cria features
def create_features(df):
    df_feat = df.copy()
    df_feat['lag_1'] = df_feat['value'].shift(1)
    df_feat['lag_7'] = df_feat['value'].shift(7)
    df_feat['lag_30'] = df_feat['value'].shift(30)
    df_feat['rolling_mean_7'] = df_feat['value'].rolling(7).mean()
    df_feat['rolling_mean_30'] = df_feat['value'].rolling(30).mean()
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    df_feat['dayofyear'] = df_feat.index.dayofyear
    df_feat['trend'] = np.arange(len(df_feat))
    return df_feat.dropna()

# %%
# 3. Avaliação
def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
    }

# %%
# 4. Baseline ingênuo: valor de hoje = valor de ontem
def naive_forecast(df, steps=90):
    df['naive'] = df['value'].shift(1)
    test = df.iloc[-steps:]
    return evaluate(test['value'], test['naive'])

# %%
# 5. Previsão recursiva multi-step
def recursive_forecast(model, last_known_data, steps=30):
    forecast = []
    current_data = last_known_data.copy()

    for _ in range(steps):
        last_row = current_data.iloc[-1:]
        new_date = last_row.index[0] + pd.Timedelta(days=1)
        new_row = pd.DataFrame(index=[new_date])
        new_row['lag_1'] = last_row['value'].values[0]
        new_row['lag_7'] = current_data['value'].iloc[-7] if len(current_data) >= 7 else np.nan
        new_row['lag_30'] = current_data['value'].iloc[-30] if len(current_data) >= 30 else np.nan
        new_row['rolling_mean_7'] = current_data['value'].iloc[-7:].mean()
        new_row['rolling_mean_30'] = current_data['value'].iloc[-30:].mean()
        new_row['dayofweek'] = new_date.dayofweek
        new_row['month'] = new_date.month
        new_row['dayofyear'] = new_date.dayofyear
        new_row['trend'] = current_data['trend'].iloc[-1] + 1

        if new_row.isna().any().any():
            break

        X_pred = new_row.copy()
        y_pred = model.predict(X_pred)[0]
        new_row['value'] = y_pred
        forecast.append((new_date, y_pred))
        current_data = pd.concat([current_data, new_row])

    return pd.DataFrame(forecast, columns=['date', 'predicted_value']).set_index('date')

# %%
# 6. Treinamento e avaliação
def train_model(name, model, param_grid, X_train, y_train):
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid

# %%
# 7. Pipeline completo
def run_pipeline(df_feat):
    os.makedirs('models', exist_ok=True)

    train = df_feat.iloc[:-90]
    test = df_feat.iloc[-90:]

    X_train = train.drop(columns=['value'])
    y_train = train['value']
    X_test = test.drop(columns=['value'])
    y_test = test['value']

    results = {}

    # Modelos
    models = {
        "RandomForest": (
            RandomForestRegressor(random_state=RANDOM_STATE),
            {'n_estimators': [100], 'max_depth': [10]},
            False
        ),
        "HistGradientBoost": (
            HistGradientBoostingRegressor(random_state=RANDOM_STATE),
            {'learning_rate': [0.1], 'max_iter': [200]},
            False
        ),
        "LinearRegression": (
            LinearRegression(),
            {},
            True
        ),
        "Ridge": (
            Ridge(),
            {'alpha': [1.0]},
            True
        ),
        "KNeighbors": (
            KNeighborsRegressor(),
            {'n_neighbors': [5]},
            True
        ),
        "SVR": (
            SVR(),
            {'C': [1], 'kernel': ['rbf']},
            True
        )
    }

    # Baseline
    naive_metrics = naive_forecast(df_feat)
    results['Naive'] = {'metrics': naive_metrics}
    print(f"📉 Naive baseline: MAE = {naive_metrics['MAE']:.4f}, RMSE = {naive_metrics['RMSE']:.4f}")

    for name, (base_model, param_grid, scale) in models.items():
        steps = [('model', base_model)]
        if scale:
            steps.insert(0, ('scaler', StandardScaler()))
        pipeline = Pipeline(steps)
        try:
            grid = train_model(name, pipeline, param_grid, X_train, y_train)
            pred = grid.predict(X_test)
            metrics = evaluate(y_test, pred)
            results[name] = {
                'model': grid,
                'metrics': metrics
            }
        except Exception as e:
            print(f"❌ Erro ao treinar {name}: {e}")

    print("\n📊 Comparação de modelos:")
    for name, res in results.items():
        if 'metrics' in res:
            print(f"{name}: MAE = {res['metrics']['MAE']:.4f}, RMSE = {res['metrics']['RMSE']:.4f}")

    # Melhor modelo
    valid_models = {k: v for k, v in results.items() if 'model' in v}
    best_name = min(valid_models, key=lambda k: valid_models[k]['metrics']['MAE'])
    best_model = valid_models[best_name]['model']
    print(f"\n🏆 Melhor modelo: {best_name}")

    # Salva avaliação
    df_results = pd.DataFrame([
        {'Model': name, **res['metrics']}
        for name, res in results.items() if 'metrics' in res
    ])
    df_results.to_csv('models/model_evaluation.csv', index=False)

    # Salva modelo campeão
    joblib.dump(best_model, 'models/best_model.pkl')

    # Previsão recursiva com base nos últimos 30 dias
    print("\n📈 Gerando previsão de 30 dias à frente...")
    last_known_data = df_feat.iloc[-30:].copy()
    forecast_df = recursive_forecast(best_model, last_known_data, steps=30)
    forecast_df.to_csv('models/forecast_30d.csv')

# %%
# Execução
df = load_data()
df_feat = create_features(df)
run_pipeline(df_feat)
