from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


file_paths = [
    "GLUSAP/data/OhioT1DM_559.csv",
    "GLUSAP/data/OhioT1DM_563.csv",
    "GLUSAP/data/OhioT1DM_570.csv",
    "GLUSAP/data/OhioT1DM_575.csv",
    "GLUSAP/data/OhioT1DM_588.csv",
    "GLUSAP/data/OhioT1DM_591.csv",
]

def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Time'])
    df.set_index('Time', inplace=True)
    return df

data = {os.path.basename(file): load_data(file) for file in file_paths}

def prepare_lagged_features(df, lags=10, freq='5min'):
    df = df.resample(freq).mean()
    df = df.dropna()
    df = df.reset_index()
    df.columns = ['ds', 'y']

    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df['y'].shift(lag)

    df['hour'] = df['ds'].dt.hour
    df['dayofweek'] = df['ds'].dt.dayofweek

    df = df.dropna()
    return df

def combined_forecasting_with_naive_as_feature(df, season_length=96, lags=10, freq='5min', catboost_weight=0.9):
    data = prepare_lagged_features(df, lags, freq)

    X = data.drop(columns=['y', 'ds'])
    y = data['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    naive_model = SeasonalNaive(season_length=season_length)
    sf = StatsForecast(models=[naive_model], freq=freq)

    train_naive = pd.DataFrame({'unique_id': '1', 'ds': data['ds'][:len(X_train)], 'y': y_train})
    naive_forecast_train = sf.forecast(df=train_naive, h=len(X_train))['SeasonalNaive']

    X_train['naive_forecast'] = naive_forecast_train.values

    train_naive_test = pd.DataFrame({'unique_id': '1', 'ds': data['ds'][len(X_train):], 'y': y_test})
    naive_forecast_test = sf.forecast(df=train_naive_test, h=len(X_test))['SeasonalNaive']
    X_test['naive_forecast'] = naive_forecast_test.values

    catboost_model = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, verbose=0)
    catboost_model.fit(X_train, y_train)

    predictions_catboost = catboost_model.predict(X_test)

    predictions_combined = catboost_weight * predictions_catboost + (1 - catboost_weight) * naive_forecast_test.values

    return X_test, y_test, predictions_combined, predictions_catboost, naive_forecast_test


try:
    fig = make_subplots(
        rows=2, cols=3,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=[f"Pacient {file_name.split('.')[0].split('_')[1]}" for file_name in data.keys()]
    )

    max_subplots = 6
    for idx, (file_name, df) in enumerate(data.items(), start=1):
        if idx > max_subplots:
            break

        try:
            season_length = int(24 * 60 / 5)
            lags = 10
            catboost_weight = 0.7

            X_test, y_test, predictions_combined, predictions_catboost, predictions_naive = combined_forecasting_with_naive_as_feature(
                df, season_length, lags, catboost_weight=catboost_weight
            )

            mae = mean_absolute_error(y_test, predictions_combined)
            mse = mean_squared_error(y_test, predictions_combined)
            r2 = r2_score(y_test, predictions_combined)

            print(f"Results for patient {file_name}:")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"R² Score: {r2:.4f}")
            print("-" * 50)

            fig.add_trace(go.Scatter(
                x=X_test.index, y=y_test, mode='lines', name='Test Data', line=dict(color='black', width=1)),
                row=(idx - 1) // 3 + 1, col=(idx - 1) % 3 + 1  # Assignar les coordenades correctes
            )

            fig.add_trace(go.Scatter(
                x=X_test.index, y=predictions_combined, mode='lines', name='Combined Predictions', line=dict(dash='dash', width=2)),
                row=(idx - 1) // 3 + 1, col=(idx - 1) % 3 + 1  # Assignar les coordenades correctes
            )

        except Exception as e:
            print(f"Error amb el fitxer {file_name}: {e}")

    fig.update_layout(
        title='Prediccions combinades SeasonalNaive + CatBoost',
        xaxis_title='',
        yaxis_title='Nivell de Glucosa',
        template='plotly',
        showlegend=False,
    )

    fig.show()
    fig.write_html("prediccio_seasonalnaive.html")

except Exception as e:
    print(f"Error en la creació del gràfic global: {e}")