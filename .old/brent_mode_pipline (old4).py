# oldAntes erros
# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

pd.options.display.max_columns = 999

# %%
RANDOM_STATE = 42

# %%
# ===============================
# 1. Carregamento e limpeza da base
# ===============================
def load_data(path='data/brent.csv'):
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.sort_values('date').set_index('date').asfreq('D')
    df['value'] = df['value'].interpolate()
    df.drop(['DATE', 'RAW DATE', 'CODE'], axis=1, inplace=True, errors='ignore')
    return df

# %%
# ===============================
# 2. Criação de variáveis preditoras
# ===============================
def create_features(df):
    df_feat = df.copy()
    
    # Features baseadas no histórico da variável alvo
    df_feat['lag_1'] = df_feat['value'].shift(1)
    df_feat['lag_7'] = df_feat['value'].shift(7)
    df_feat['lag_30'] = df_feat['value'].shift(30)
    df_feat['rolling_mean_7'] = df_feat['value'].rolling(7).mean()
    df_feat['rolling_mean_30'] = df_feat['value'].rolling(30).mean()

    # Sazonalidade temporal
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    df_feat['dayofyear'] = df_feat.index.dayofyear

    # Tendência temporal
    df_feat['trend'] = np.arange(len(df_feat))

    # Limpa colunas que não existem
    for col in ['DATE', 'RAW DATE', 'CODE']:
        if col in df_feat.columns:
            df_feat.drop(columns=[col], inplace=True)

    df_feat = df_feat.dropna()
    return df_feat


# %%
# ===============================
# 3. Avaliação de desempenho
# ===============================
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse}

# %%
# ===============================
# 4. Treinamento com validação temporal
# ===============================
def train_model(name, model, param_grid, X_train, y_train):
    print(f"🔍 Training {name}...")
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(
        model, param_grid, cv=tscv,
        scoring='neg_mean_absolute_error', n_jobs=-1
        )
    grid.fit(X_train, y_train)
    return grid


# %%
# ===============================
# 5. Previsão recursiva multi-step
# ===============================
def recursive_forecast(model, last_known_data, steps=30):
    """
    Gera previsões para N dias à frente usando o último período conhecido (ex: últimos 30 dias).
    A cada passo, usa a previsão anterior para compor as features do próximo.
    """
    forecast = []
    current_data = last_known_data.copy()

    for i in range(steps):
        last_row = current_data.iloc[-1:]

        # Cria a nova linha com base na última data + 1
        new_date = last_row.index[0] + pd.Timedelta(days=1)
        new_row = pd.DataFrame(index=[new_date])

        # Gera features manualmente com base na lógica do create_features
        new_row['lag_1'] = last_row['value'].values[0]
        new_row['lag_7'] = current_data['value'].iloc[-7] if len(current_data) >= 7 else np.nan
        new_row['lag_30'] = current_data['value'].iloc[-30] if len(current_data) >= 30 else np.nan
        new_row['rolling_mean_7'] = current_data['value'].iloc[-7:].mean()
        new_row['rolling_mean_30'] = current_data['value'].iloc[-30:].mean()
        new_row['dayofweek'] = new_date.dayofweek
        new_row['month'] = new_date.month
        new_row['dayofyear'] = new_date.dayofyear
        new_row['trend'] = current_data['trend'].iloc[-1] + 1

        # Checa se há valores ausentes
        if new_row.isna().any().any():
            print("⚠️ Aviso: valores NaN em features de previsão. Interrompendo forecast.")
            break

        # Faz previsão
        X_pred = new_row.copy()
        y_pred = model.predict(X_pred)[0]

        # Adiciona valor previsto
        new_row['value'] = y_pred
        forecast.append((new_date, y_pred))

        # Adiciona nova linha à base atual para usar nos próximos passos
        current_data = pd.concat([current_data, new_row])

    # Converte para DataFrame
    forecast_df = pd.DataFrame(forecast, columns=['date', 'predicted_value']).set_index('date')
    return forecast_df

# %%
# ===============================
# 6. Pipeline completo
# ===============================
def run_pipeline(df):
    
    # Separação de treino/teste (últimos 90 dias para teste)
    train = df_feat.iloc[:-90]
    test = df_feat.iloc[-90:]

    X_train = train.drop(columns=['value'])
    y_train = train['value']
    X_test = test.drop(columns=['value'])
    y_test = test['value']

    results = {}

    # Modelos e hiperparâmetros
    models = {
        "RandomForest": (
            RandomForestRegressor(random_state=RANDOM_STATE),
            {'n_estimators': [100], 'max_depth': [10]}
        ),
        "HistGradientBoost": (
            HistGradientBoostingRegressor(random_state=RANDOM_STATE),
            {'learning_rate': [0.1], 'max_iter': [200]}
        ),
        "LinearRegression": (
            LinearRegression(),
            {}
        ),
        "Ridge": (
            Ridge(),
            {'alpha': [1.0]}
        ),
        "KNeighbors": (
            KNeighborsRegressor(),
            {'n_neighbors': [5]}
        ),
        "SVR": (
            SVR(),
            {'C': [1], 'kernel': ['rbf']}
        )
    }

    for name, (model, param_grid) in models.items():
        try:
            grid = train_model(name, model, param_grid, X_train, y_train)
            pred = grid.predict(X_test)
            metrics = evaluate(y_test, pred)
            results[name] = {
                'model': grid,
                'metrics': metrics
            }
        except Exception as e:
            print(f"❌ Error training {name}: {e}")

    print("\n📊 Model Evaluation Summary:")
    for name, res in results.items():
        print(f"{name}: MAE = {res['metrics']['MAE']:.4f}, RMSE = {res['metrics']['RMSE']:.4f}")

    best_name = min(results, key=lambda k: results[k]['metrics']['MAE'])
    best_model = results[best_name]['model']
    print(f"\n🏆 Best model: {best_name} (MAE = {results[best_name]['metrics']['MAE']:.4f})")

    # Salvar avaliação
    os.makedirs('models', exist_ok=True)
    results_df = pd.DataFrame([
        {
            'Model': name,
            'MAE': res['metrics']['MAE'],
            'RMSE': res['metrics']['RMSE']
        }
        for name, res in results.items()
    ])
    results_df.to_csv('models/model_evaluation.csv', index=False)
    print("📁 Avaliações salvas em models/model_evaluation.csv")

    # Salvar modelo campeão
    joblib.dump(best_model, 'models/best_model.pkl')
    print("✅ Modelo salvo em models/best_model.pkl")

    # Forecast de 30 dias à frente com modelo vencedor
    print("\n📈 Gerando previsão de 30 dias à frente com o melhor modelo...")
    last_known_data = df_feat.iloc[-30:]  # base para começar a previsão
    forecast_df = recursive_forecast(best_model, last_known_data, steps=30)
    forecast_df.to_csv('models/forecast_30d.csv')
    print("📁 Previsão futura salva em models/forecast_30d.csv")

# %%S
# ===============================
# 7. Execução do pipeline
# ===============================
# if __name__ == "__main__":
# %%
df = load_data()
# %%
df_feat = create_features(df)
    
# %%
run_pipeline(df_feat)

# %%
