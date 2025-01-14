import pandas as pd
import glob
import plotly.express as px
import plotly.graph_objects as go


def load_data(filepaths):
    dataframes = {}
    for path in filepaths:
        category = path.split("/")[-2]  # Extreure la categoria de la ruta
        files = glob.glob(path)
        dfs = [pd.read_csv(file) for file in files]
        combined_df = pd.concat(dfs, ignore_index=True)
        dataframes[category] = combined_df
    return dataframes


filepaths = [
    'GLUSAP/data/processed/simglucose/heterogeneous_/adult#00*.csv',
    'GLUSAP/data/processed/simglucose/homogeneous_low/adult#00*.csv',
    'GLUSAP/data/processed/simglucose/homogeneous_high/adult#00*.csv'
]

data = load_data(filepaths)

# 1. Gràfic de Distribució de BG (Glucosa en Sang)
bg_fig = go.Figure()
for category, df in data.items():
    df = df[df["BG"] > 10]
    bg_fig.add_trace(go.Histogram(
        x=df["BG"],
        histnorm='probability',
        name=category.replace('_', ' ').title(),
        opacity=0.7
    ))

bg_fig.update_layout(
    title="Distribució de BG (Glucosa en Sang)",
    xaxis_title="BG (mg/dL)",
    yaxis_title="Densitat",
    barmode='overlay'
)
bg_fig.write_html("distribucio_bg.html")
bg_fig.show()


percentages = []
normal_range_min = 70  # Mínim valor de BG normal
normal_range_max = 180  # Màxim valor de BG normal

ranges = {
    "70 <= BG <= 180": (normal_range_min, normal_range_max),
    "BG > 180": (normal_range_max, float('inf')),
    "BG < 70": (float('-inf'), normal_range_min),
    "BG > 250": (250, float('inf')),
    "BG < 50": (float('-inf'), 50)
}

for category, df in data.items():
    # Percentatges per cada rang de BG
    percentage_by_range = {}

    for range_label, (lower, upper) in ranges.items():
        if lower == float('-inf'):
            count_in_range = (df["BG"] < upper).sum()
        elif upper == float('inf'):
            count_in_range = (df["BG"] > lower).sum()
        else:
            count_in_range = ((df["BG"] >= lower) & (df["BG"] <= upper)).sum()

        percentage_by_range[range_label] = (count_in_range / len(df)) * 100 if len(df) > 0 else 0

    # Modificar el nom de la categoria (substituir _ per espais i posar en majúscula)
    formatted_category = category.replace('_', ' ').title()

    # Afegir els percentatges a la llista
    percentages.append({
        "Categoria": formatted_category,
        "70 <= BG <= 180 (%)": percentage_by_range["70 <= BG <= 180"],
        "BG > 180 (%)": percentage_by_range["BG > 180"],
        "BG < 70 (%)": percentage_by_range["BG < 70"],
        "BG > 250 (%)": percentage_by_range["BG > 250"],
        "BG < 50 (%)": percentage_by_range["BG < 50"]
    })

percentages_df = pd.DataFrame(percentages)

bg_percentage_fig = px.bar(
    percentages_df,
    x="Categoria",
    y=["70 <= BG <= 180 (%)", "BG > 180 (%)", "BG < 70 (%)", "BG > 250 (%)", "BG < 50 (%)"],
    title="Percentatge de BG en diferents intervals per Categoria",
    labels={"value": "Percentatge (%)", "Categoria": "Categoria"},
    barmode="group"
)

bg_percentage_fig.write_html("percentatges_bg_ranges.html")
bg_percentage_fig.show()
