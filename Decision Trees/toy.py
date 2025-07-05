import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from typing import Tuple

np.random.seed(42)

app = dash.Dash(__name__)

def generate_blobs_data(n_clusters: int = 3, sample_sizes: list[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if sample_sizes is None:
        sample_sizes = [100] * n_clusters
    
    # Generate blobs for each cluster with specified sample sizes
    X_list = []
    y_list = []
    
    for i in range(n_clusters):
        cluster_X, cluster_y = make_blobs(
            n_samples=sample_sizes[i], 
            centers=1, 
            cluster_std=1.0,
            center_box=(-10.0, 10.0), 
            random_state=42 + i  # Different seed for each cluster
        )
        X_list.append(cluster_X)
        y_list.append(np.full(sample_sizes[i], i))  # Assign cluster labels
    
    # Combine all clusters
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    return X, y

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
def update_blob_plot(n_clusters: int, proportions: list = None):
    if proportions is None:
        proportions = [1/n_clusters] * n_clusters
    
    # Calculate sample sizes for each cluster based on proportions
    total_samples = 300
    sample_sizes = [int(prop * total_samples) for prop in proportions]
    
    # Adjust for rounding errors to ensure total samples = 300
    sample_sizes[0] += total_samples - sum(sample_sizes)
    
    X, y = generate_blobs_data(n_clusters, sample_sizes)
    
    fig = px.scatter(
        x=X[:, 0], 
        y=X[:, 1], 
        color=y,
        title=f'Blob Configuration with {n_clusters} Clusters',
        labels={'x': 'Feature 1', 'y': 'Feature 2'},
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        title_x=0.5,
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        showlegend=False,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        width=600,
        height=600,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

if __name__ == "__main__":
    print("Starting Dash app for blob visualization...")
    app.run_server(debug=True)
