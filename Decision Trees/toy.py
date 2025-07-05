import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from data_generator import generate_data
from utils import seed_everything, plot_dash_data
seed_everything()

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Blob Configuration Visualizer", style={'textAlign': 'center'}),
    
    html.Div([
        html.Label("Number of Clusters:"),
        dcc.Slider(
            id='cluster-slider',
            min=2,
            max=8,
            step=1,
            value=3,
            marks={i: str(i) for i in range(2, 9)},
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], style={'width': '50%', 'margin': 'auto', 'padding': '20px'}),
    
    html.Div([
        dcc.Graph(id='blob-plot')
    ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'})
])


@app.callback(
    Output('blob-plot', 'figure'),
    Input('cluster-slider', 'value')
)
def update_blob_plot(n_clusters: int):
    X, y, centers = generate_data(n_clusters)
    fig = plot_dash_data(X, y, centers)
    return fig

if __name__ == "__main__":
    print("Starting Dash app for blob visualization...")
    app.run_server(debug=True)
