from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import glob

# Funcions de càrrega de dades
def carregar_dades(archivo):
    """
    Llegeix un fitxer CSV i converteix la columna 'Time' a tipus datetime.
    """
    df = pd.read_csv(archivo)
    df['Time'] = pd.to_datetime(df['Time'])
    return df

def carregar_dades_pacient(pacient):
    """
    Carrega dades específiques del pacient des de fitxers CSV.
    """
    dades_train = pd.read_csv(f"train_data_{pacient}.csv")
    dades_test = pd.read_csv(f"test_data_{pacient}.csv")
    prediccions = pd.read_csv(f"predictions_patient_{pacient}.csv")
    return dades_train, dades_test, prediccions

def llistar_pacients():
    """
    Llista els pacients disponibles basant-se en els fitxers de dades.
    """
    fitxers_train = glob.glob("train_data_*.csv")
    pacients = [f.split('_')[-1].split('.')[0] for f in fitxers_train]
    return pacients

def llistar_fitxers():
    """
    Llista els fitxers disponibles per a simulacions en diferents categories.
    """
    rutes = {
        "heterogeneous": 'src/data/processed/simglucose/heterogeneous_/adult#00*.csv',
        "homogeneous_low": 'src/data/processed/simglucose/homogeneous_low/adult#00*.csv',
        "homogeneous_high": 'src/data/processed/simglucose/homogeneous_high/adult#00*.csv',
    }
    fitxers = {categoria: glob.glob(path) for categoria, path in rutes.items()}
    return fitxers

# Càrrega inicial de dades
fitxers_simulacio = llistar_fitxers()
dades_simulacio = {
    categoria: [carregar_dades(f) for f in fitxers_simulacio[categoria] if "model" not in f]
    for categoria in fitxers_simulacio
}

rutes_ohio = glob.glob('data/OhioT1DM_*.csv')
dades_ohio = {f"Pacient {f.split('_')[1].split('.')[0]}": carregar_dades(f) for f in rutes_ohio}

rutes_ohio_processed = glob.glob('data/processed/OhioT1DM_*_processed.csv')
dades_ohio_processed = {f"Pacient {f.split('_')[1].split('.')[0]}": carregar_dades(f) for f in rutes_ohio_processed}

# Configuració de l'aplicació Dash
app = Dash(__name__)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <title>Gráfic Interactiu de Predicció</title>
        <link href="https://fonts.googleapis.com/css2?family=Raleway:wght@400;600&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Raleway', sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f8f9fa;
            }
            h1 {
                color: #2c3e50;
                font-weight: 600;
                text-align: center;
            }
            .container {
                position: relative;
                margin: 50px;
                padding-top: 20px;
            }
            .main-graph {
                width: 100%;
                height: 600px;
            }
            .zoom-graph {
                position: absolute;
                top: 10px; /* Posicionat a la part superior */
                right: 20px; /* Alineat a la dreta */
                width: 35%; /* Anchura ajustada */
                height: 300px;
                border: 1px solid #ccc;
                background-color: white;
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
            }
            .info-box {
                margin-top: 20px;
                font-size: 18px;
                text-align: center;
            }
            /* Controls al final */
            .controls {
                margin-top: 40px;
                text-align: center;
            }
            .control-container {
                margin-bottom: 30px;
            }
            .label-container {
                margin-bottom: 10px;
            }
        </style>
    </head>
    <body>
        <div id="react-entry-point">
            {%app_entry%}
        </div>
        <footer>
            {%config%}
            {%scripts%}
            <script id="_dash-renderer" type="application/javascript">
                new DashRenderer();
            </script>
        </footer>
    </body>
</html>
'''

# Disseny de l'aplicació
app.layout = html.Div([
    html.H1("GLUSAP Dashboard"),
    dcc.Tabs(id="tabs", value="prediccio", children=[
        dcc.Tab(label="Predicció", value="prediccio", children=[
            html.Div(className="container", children=[
                html.Button("Play", id="play-pause-button", n_clicks=0, style={
                    "padding": "10px 20px", "font-size": "16px", "cursor": "pointer",
                    "margin-top": "20px", "background-color": "#007bff", "color": "white",
                    "border": "none", "border-radius": "5px",
                }),
                dcc.Graph(id="grafic-principal", className="main-graph"),
                dcc.Graph(id="grafic-zoom", className="zoom-graph"),
                html.Div(className="controls", children=[
                    html.Label("Selecciona el pacient:"),
                    dcc.Dropdown(
                        id="pacient-dropdown",
                        options=[{'label': p, 'value': p} for p in llistar_pacients()],
                        value=llistar_pacients()[0], clearable=False
                    ),
                    dcc.Interval(id="interval-actualitzacio", interval=500, disabled=True)
                ])
            ])
        ]),
        dcc.Tab(label="Simulacions", value="simulacions", children=[
            html.Label("Selecciona tipus de simulació:"),
            dcc.Dropdown(
                id="categoria-dropdown",
                options=[
                    {'label': 'Heterogeneous', 'value': 'heterogeneous'},
                    {'label': 'Homogeneous Low', 'value': 'homogeneous_low'},
                    {'label': 'Homogeneous High', 'value': 'homogeneous_high'}
                ],
                value='heterogeneous'
            ),
            html.Div(id="info-simulacions"),
            dcc.Graph(id="grafic-simulacios")
        ]),
        dcc.Tab(label="Dades reals - OhioT1DM", value="dadesreals", children=[
            html.Label("Selecciona pacients:"),
            dcc.Checklist(id="pacients-checklist", options=[], value=[]),
            dcc.Graph(id="grafico-ohio")
        ])
    ])
])

@app.callback(
    Output('play-pause-button', 'children'),
    Output('intervalo-actualizacion', 'disabled', allow_duplicate=True),
    Input('play-pause-button', 'n_clicks'),
    State('intervalo-actualizacion', 'disabled'),
    prevent_initial_call='initial_duplicate'
)
def toggle_play_pause(n_clicks, is_disabled):
    if n_clicks % 2 == 1:
        return "Pausa", False
    else:
        return "Play", True


@app.callback(
    Output("adultos-checklist", "options"),
    [Input("categoria-dropdown", "value")]
)
def actualitzar_checklist(categoria):
    adultos = [f"Adult {i + 1}" for i in range(len(dades_ohio[categoria]))]
    return [{'label': adulto, 'value': adulto} for adulto in adultos]


@app.callback(
    Output("adults-ohio-checklist", "options"),
    [Input("tabs", "value")]
)
def actualitzar_checklist(tab_value):
    if tab_value == "dadesreals":
        adultos = [f"Pacient {i.split('_')[1].split('.')[0]}" for i in rutes_ohio]
        return [{'label': adulto, 'value': adulto} for adulto in adultos]
    return []

@app.callback(
    Output("grafico-real", "figure"),
    [Input("adults-ohio-checklist", "value"),
     Input("processed-checklist", "value")]
)
def actualitzar_grafic_real(seleccionados, procesar):
    fig = go.Figure()

    for adulto in seleccionados:
        if procesar and 'process' in procesar:
            df = dades_ohio_processed[adulto]
        else:
            df = dades_ohio[adulto]

        fig.add_trace(go.Scatter(x=df['Time'], y=df['CGM'], mode='lines', name=f'Glucosa {adulto}'))

    fig.add_hline(y=70, line=dict(color='red'), name="Hipoglucemia", showlegend=True)
    fig.add_hline(y=180, line=dict(color='orange'), name="Hiperglucemia", showlegend=True)

    fig.update_layout(
        title=f"OhioT1DM",
        xaxis_title="Temps",
        yaxis_title="Glucosa (CGM)",
        template="plotly_white"
    )

    return fig

@app.callback(
    Output("grafic-simulacio", "figure"),
    [Input("categoria-dropdown", "value"),
     Input("adultos-checklist", "value")]
)
def actualitzar_grafic_simulacion(categoria, seleccionados):
    fig = go.Figure()

    mostrar_comidas = len(seleccionados) == 1

    for i, adulto in enumerate(seleccionados):
        df = dades_ohio[categoria][i]

        fig.add_trace(go.Scatter(x=df['Time'], y=df['CGM'], mode='lines', name=f'Glucosa {adulto}'))

        if mostrar_comidas:
            comidas = df[df['CHO'] > 0]

            for j, comida in comidas.iterrows():
                fig.add_vline(x=comida['Time'], line=dict(color="black", width=1, dash="dash"), showlegend=False)

            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name="Menjar", line=dict(color="black", dash="dash"), showlegend=True))


    fig.add_hline(y=70, line=dict(color='red'), name="Hipoglucemia", showlegend=True)
    fig.add_hline(y=180, line=dict(color='orange'), name="Hiperglucemia", showlegend=True)

    fig.update_layout(
        title=f"Simulacions de Glucosa - {categoria}",
        xaxis_title="Temps",
        yaxis_title="Glucosa (CGM)",
        template="plotly_white"
    )

    return fig

@app.callback(
    Output("intervalo-actualizacion", "disabled"),
    [Input("play-pause-button", "n_clicks")],
    prevent_initial_call=True
)
def alternar_pausa_play(n_clicks):
    return n_clicks % 2 == 0


@app.callback(
    Output("grafic-principal", "figure"),
    [Input("pacient-dropdown", "value"),
        Input("intervalo-actualizacion", "n_intervals")],
    prevent_initial_call=True
)
def actualitzar_grafic_progresiu(pacient, n_intervals):
    train_data, test_data, predictions = carregar_dades_pacient(pacient)
    predictions = predictions[['Time', 'Pred_1']]
    predictions = predictions.rename(columns={'Pred_1': 'Predictions'})

    train_limit = int(len(train_data) * 0.05)
    train_data = train_data.iloc[-train_limit:]

    max_points = n_intervals + 1
    test_data_limited = test_data.iloc[:max_points]
    predictions_limited = predictions.iloc[:max_points]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train_data['Time'],
        y=train_data['CGM'],
        mode='lines',
        name='Train Real',
    ))

    if not test_data_limited.empty:
        fig.add_trace(go.Scatter(
            x=test_data_limited['Time'],
            y=test_data_limited['CGM'],
            mode='lines',
            name='Test Real',
        ))

    if not predictions_limited.empty:
        fig.add_trace(go.Scatter(
            x=predictions_limited['Time'],
            y=predictions_limited['Predictions'],
            mode='lines',
            name='Prediccions',
            opacity=0.7
        ))

    fig.update_layout(
        title="Predicció LSTM - Pacient 559",
        xaxis_title="Temps",
        yaxis_title="CGM",
        template="plotly_white",
        legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.2),
        xaxis=dict(
            range=[train_data['Time'].iloc[0], test_data['Time'].iloc[-1]]  # Reservar espacio en el eje X
        )
    )

    return fig


@app.callback(
    Output("grafic-zoom", "figure"),
    [
        Input("intervalo-actualizacion", "n_intervals"),
        Input("pacient-dropdown", "value")
    ],
)
def actualitzar_grafic_zoom(n_intervals, pacient):
    train_data, test_data, predictions = carregar_dades_pacient(pacient)
    predictions = predictions.rename(columns={'Pred_1': 'Predictions'})

    # Obtenir l'índex corresponent al temps de la predicció actual
    max_points = n_intervals + 1
    predictions_limited = predictions.iloc[:max_points]

    if predictions_limited.empty:
        return go.Figure()  # Si no hi ha dades, retornar un gràfic buit

    # Obtenir el timestamp de la darrera predicció
    time_selected = predictions_limited['Time'].iloc[-1]
    time_selected = pd.to_datetime(time_selected)

    predictions['Time'] = pd.to_datetime(predictions['Time'], errors='coerce')
    test_data['Time'] = pd.to_datetime(test_data['Time'], errors='coerce')

    fig = go.Figure()

    idx_selected = predictions.loc[predictions['Time'] == time_selected].index

    if not idx_selected.empty:
        idx_selected = idx_selected[0]

        # Índex de les prediccions +5 i +25
        idx_pred_1 = idx_selected
        idx_pred_5 = idx_selected + 4

        if idx_pred_5 < len(predictions):
            # Calcular la pendent entre +5 i +25
            y1 = predictions.iloc[idx_pred_1]['Predictions']
            y2 = predictions.iloc[idx_pred_5]['Pred_5']
            pendiente = (y2 - y1) / 20  # Suposant un interval de 20 minuts

            # Comprovar els valors de glucosa predits
            glucosa_maxima = predictions.iloc[idx_pred_1:idx_pred_5 + 1]["Predictions"].max()

            # Definir els colors segons la pendent i la glucosa màxima
            if glucosa_maxima > 200:
                color = "rgba(255, 0, 0, 0.2)"  # Vermell per glucosa > 200
            elif glucosa_maxima > 180:
                color = "rgba(255, 255, 0, 0.2)"  # Groc per glucosa > 180
            elif abs(pendiente) < 5:
                color = "rgba(0, 255, 0, 0.2)"  # Verd per pendent baixa
            elif 5 <= abs(pendiente) < 10:
                color = "rgba(255, 255, 0, 0.2)"  # Groc per pendent moderada
            else:
                color = "rgba(255, 0, 0, 0.2)"  # Vermell per pendent alta

            # Afegir l'ombra al gràfic
            fig.add_shape(
                type="rect",
                x0=predictions.iloc[idx_pred_1]['Time'],
                x1=predictions.iloc[idx_pred_5]['Time'],
                y0=test_data['CGM'].min() - 20,
                y1=test_data['CGM'].max() + 20,
                fillcolor=color,
                line=dict(width=0),
                layer="below"
            )

            # Afegir els punts de predicció com abans
            for i, pred_col in enumerate(["Predictions", "Pred_2", "Pred_3", "Pred_4", "Pred_5"]):
                pred_idx = idx_selected + i
                fig.add_trace(go.Scatter(
                    x=[predictions.iloc[pred_idx]['Time']],
                    y=[predictions.iloc[pred_idx][pred_col]],
                    mode='markers+text',
                    name=f"+{(i+1)*5} min",
                    marker=dict(size=10),
                    opacity=0.8
                ))

            # Afegir les dades reals
            mask = (test_data['Time'] >= (predictions.iloc[idx_pred_1]['Time'] - pd.Timedelta(minutes=5))) & (
                test_data['Time'] <= predictions.iloc[idx_pred_5]['Time'])
            real_data = test_data[mask]
            fig.add_trace(go.Scatter(
                x=real_data['Time'],
                y=real_data['CGM'],
                mode='lines+markers',
                name="Test real",
                line=dict(color='crimson'),
                marker=dict(size=2, color='crimson'),
                opacity=0.6
            ))

    fig.update_layout(
        title="Predicció a diferents horitzons vs. Dades reals",
        xaxis_title="Temps",
        yaxis_title="CGM",
        template="plotly_white",
        xaxis=dict(
            range=[time_selected - pd.Timedelta(minutes=10), time_selected + pd.Timedelta(minutes=30)]
        ),
        yaxis=dict(
            range=[test_data['CGM'].min() - 20, test_data['CGM'].max() + 20]
        )
    )

    return fig

if __name__ == "__main__":
    app.run_server(debug=True)
