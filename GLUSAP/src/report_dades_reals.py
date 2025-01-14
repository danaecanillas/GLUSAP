import os
import plotly.express as px
import pandas as pd
from statsmodels.tsa.seasonal import STL
import pandas as pd
import plotly.graph_objects as go

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

combined_data = []
for file_name, df in data.items():
    temp_df = df.copy()
    temp_df['Pacient'] = file_name
    combined_data.append(temp_df)

combined_df = pd.concat(combined_data, ignore_index=True)

fig = px.histogram(
    combined_df,
    x='CGM',
    color='Pacient',
    nbins=30,
    marginal='box',
    opacity=0.75,
    title="Distribució de CGM per Pacient",
    labels={'CGM': 'CGM', 'Pacient': 'Pacient'}
)

fig.update_layout(
    xaxis_title="CGM",
    yaxis_title="Freqüència",
    legend_title="Pacient",
    template="plotly_white"
)

fig.show()

fig.write_html("distribucio_cgm.html")

############################################################################################################
fig = go.Figure()

colors = px.colors.qualitative.Plotly

for file_name, df in data.items():
    df = df.resample('5T').mean()

    df['CGM'] = df['CGM'].interpolate(method='time')

    # Descomposició STL
    stl = STL(df['CGM'], period=288)
    result = stl.fit()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=result.trend,
        mode='lines',
        name=f"Tendència - {file_name}",
        line=dict(width=2)
    ))

fig.update_layout(
    title="Tendència diària per Pacient",
    xaxis_title="Temps",
    yaxis_title="Valors de CGM",
    template="plotly_white",
    legend_title="Tendència"
)

fig.show()

fig.write_html("descomposicio_temporal.html")



