import statsmodels.api as sm
import pandas as pd
from catboost import CatBoostRegressor
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def prepare_lagged_features(df, lags=10, freq='15min'):
    df = df.set_index('Time').resample(freq).mean().dropna()  # Resampleo
    df = df.reset_index()
    df.columns = ['ds', 'y']
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df['y'].shift(lag)

    df['hour'] = df['ds'].dt.hour
    df['dayofweek'] = df['ds'].dt.dayofweek
    df = df.dropna()
    return df

def forecast_sarimax(train, test, pdq=(4, 0, 3), PDQm=(1, 1, 1, 96)):
    train = train.asfreq('15min')
    test = test.asfreq('15min')
    train_model = sm.tsa.SARIMAX(train, order=pdq, seasonal_order=PDQm)
    train_model_fit = train_model.fit()

    test_model = sm.tsa.SARIMAX(
        pd.concat([train, test]),
        order=pdq,
        seasonal_order=PDQm
    )
    test_model_fit = test_model.filter(train_model_fit.params)
    predict_test = test_model_fit.get_prediction(start=len(train), end=len(train) + len(test) - 1, dynamic=True)

    return predict_test.predicted_mean


files = glob.glob("GLUSAP/data/OhioT1DM_*.csv")
results_dict = {}
fig = make_subplots(
    rows=2, cols=3,
    shared_xaxes=False,
    vertical_spacing=0.08,
    subplot_titles=[f"Paciente {file.split('_')[1].split('.')[0]}" for file in files]
)
for idx, file in enumerate(files, start=1):
    try:
        df = pd.read_csv(file)

        df['Time'] = pd.to_datetime(df['Time'])
        scaler = MinMaxScaler()
        df['CGM'] = scaler.fit_transform(df[['CGM']])

        data = prepare_lagged_features(df, lags=10, freq='15min')
        last_date = data['ds'].max()

        train_start_date = last_date - pd.Timedelta(days=25)
        train_end_date = last_date - pd.Timedelta(days=5)
        train_data = data[(data['ds'] >= train_start_date) & (data['ds'] < train_end_date)]
        test_data = data[data['ds'] >= train_end_date]

        train_p = train_data['ds'].min().strftime("%Y-%m-%d")
        test_p = test_data['ds'].min().strftime("%Y-%m-%d")

        X_train = train_data.drop(columns=['y', 'ds'])
        y_train = train_data['y']
        X_test = test_data.drop(columns=['y', 'ds'])
        y_test = test_data['y']

        sarimax_test_predictions = forecast_sarimax(
            y_train.set_axis(pd.date_range(start=train_p, periods=len(y_train), freq="15min")),
            y_test.set_axis(pd.date_range(start=test_p, periods=len(y_test), freq="15min"))
        )

        catboost_model = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, verbose=0)
        catboost_model.fit(X_train, y_train)

        catboost_predictions = catboost_model.predict(X_test)

        catboost_weight = 0.7
        combined_predictions = catboost_weight * catboost_predictions + (1 - catboost_weight) * sarimax_test_predictions
        combined_denorm = scaler.inverse_transform(pd.DataFrame(combined_predictions))
        real_values = scaler.inverse_transform(pd.DataFrame(y_test.to_numpy()))

        mse = mean_squared_error(real_values, combined_denorm)
        mae = mean_absolute_error(real_values, combined_denorm)
        r2 = r2_score(real_values, combined_denorm)

        print(f"Paciente {file.split('_')[1].split('.')[0]} - MSE: {mse:.5f}, MAE: {mae:.5f}, R²: {r2:.5f}")

        results_dict[file] = {
            'real_values': real_values.flatten().tolist(),
            'sarimax_predictions': scaler.inverse_transform(pd.DataFrame(sarimax_test_predictions)).flatten().tolist(),
            'catboost_predictions': scaler.inverse_transform(pd.DataFrame(catboost_predictions)).flatten().tolist(),
            'combined_predictions': combined_denorm.flatten().tolist()
        }
        fig.add_trace(go.Scatter(
            x=test_data['ds'], y=real_values.flatten(), mode='lines',
            name=f'Paciente {file.split("_")[1].split(".")[0]} - Datos reales', line=dict(color='black', width=1)),
            row=(idx - 1) // 3 + 1, col=(idx - 1) % 3 + 1
        )
        fig.add_trace(go.Scatter(
            x=test_data['ds'], y=combined_denorm.flatten(), mode='lines',
            name=f'Pacient {file.split("_")[1].split(".")[0]} - Predicción combinada',
            line=dict(dash='dash', width=2)),
            row=(idx - 1) // 3 + 1, col=(idx - 1) % 3 + 1
        )
    except Exception as e:
        print(f"Error con el archivo {file}: {e}")

fig.update_layout(
    title='Prediccions combinades SARIMA + CatBoost',
    xaxis_title='',
    yaxis_title='Nivell de Glucosa',
    showlegend=False,
    template='plotly'
)
fig.show()
fig.write_html("prediccions_sarimax.html")
# Imprimir resultados del diccionario
print(results_dict)