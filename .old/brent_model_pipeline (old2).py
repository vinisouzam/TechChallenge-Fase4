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

RANDOM_STATE = 42

# --- Load data ---
def load_data(path='data/brent.csv'):
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.sort_values('date').set_index('date').asfreq('D')
    df['value'] = df['value'].interpolate()
    return df

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

    # Remove columns only if they exist
    for col in ['DATE', 'RAW DATE', 'CODE']:
        if col in df_feat.columns:
            df_feat.drop(col, axis=1, inplace=True)

    return df_feat

# --- Evaluation metrics ---
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)  # RMSE manualmente
    return {"MAE": mae, "RMSE": rmse}

# --- Grid search for model ---
def train_model(name, model, param_grid, X_train, y_train):
    print(f"🔍 Training {name}...")
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(
        model, param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X_train, y_train)
    return grid

# --- Main pipeline ---
def run_pipeline():
    df = load_data()
    df_feat = create_features(df)

    # Train/test split
    split_date = '2023-01-01'
    train = df_feat[df_feat.index < split_date]
    test = df_feat[df_feat.index >= split_date]

    X_train = train.drop(columns=['value'])
    y_train = train['value']
    X_test = test.drop(columns=['value'])
    y_test = test['value']

    results = {}

    models = {
        "RandomForest": (
            RandomForestRegressor(random_state=RANDOM_STATE),
            {'n_estimators': [50, 100], 'max_depth': [5, 10]}
        ),
        "HistGradientBoost": (
            HistGradientBoostingRegressor(random_state=RANDOM_STATE),
            {'learning_rate': [0.05, 0.1], 'max_iter': [100, 200]}
        ),
        "LinearRegression": (
            LinearRegression(),
            {}
        ),
        "Ridge": (
            Ridge(),
            {'alpha': [0.1, 1.0, 10]}
        ),
        "KNeighbors": (
            KNeighborsRegressor(),
            {'n_neighbors': [3, 5, 10]}
        ),
        "SVR": (
            SVR(),
            {'C': [0.1, 1, 10], 'kernel': ['rbf']}
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

    # Print all results
    print("\n📊 Model Evaluation Summary:")
    for name, res in results.items():
        print(f"{name}: MAE = {res['metrics']['MAE']:.4f}, RMSE = {res['metrics']['RMSE']:.4f}")

    # Choose best by MAE
    best_name = min(results, key=lambda k: results[k]['metrics']['MAE'])
    best_model = results[best_name]['model']
    print(f"\n🏆 Best model: {best_name} (MAE = {results[best_name]['metrics']['MAE']:.4f})")

    # Exportar os resultados para CSV
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
    
    # Save
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    print("✅ Model saved to models/best_model.pkl")

if __name__ == "__main__":
    run_pipeline()

    
