import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from utils import plot_dash_data, plot_impurity_vs_depth, seed_everything
from simple_gini_visualization import truncate_tree_to_depth, visualize_tree
from plot_tree import plot_tree_graph
seed_everything(12)
# seed_everything(191)
# seed_everything(191)
app = dash.Dash(__name__)
# Configuration parameters
BASE = 0.4  # Decay factor - each cluster has 40% of the previous cluster's size
DEFAULT_N_CLUSTERS = 2
TOTAL_SAMPLES = 16

def calculate_proportions(n_clusters: int) -> np.ndarray:
    """Calculate cluster proportions based on decay factor."""
    proportions = [BASE**i for i in range(n_clusters)]
    proportions = np.array(proportions) / np.sum(proportions)
    return proportions

def main():
    """
    Main function to demonstrate Gini impurity visualization.
    """
    # Load dataset using data_generator
    from data_generator import generate_data, generate_blobs_data, generate_chessboard_data
    
    # proportions = calculate_proportions(DEFAULT_N_CLUSTERS)
    # X, y, centers = generate_data(DEFAULT_N_CLUSTERS, TOTAL_SAMPLES, proportions)
    X, y, centers = generate_chessboard_data(TOTAL_SAMPLES, 2, 4)
    # centers = np.array([[-1, 0], [1, 0]])
    # X, y = generate_blobs_data(centers)
    feature_names, target_names = ['X0', 'X1'], ['0', '1']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y
    )
    # fig = plot_dash_data(X_train, y_train, centers)
    # fig.write_html('my_plot.html', auto_open=True)
    
    # Create and fit decision tree
    print("Training Decision Tree Classifier...")
    dt = DecisionTreeClassifier(
        criterion='gini',
        max_depth=7,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    dt.fit(X_train, y_train)
    # print(f"depth={None}, y={np.sum(dt.predict(X_train))}")
    # visualize_tree(dt, feature_names, target_names)
    # for d in [0,1,2]:
    #     truncated_tree = truncate_tree_to_depth(dt, d)
    #     print(f"depth={d}, y={np.sum(truncated_tree.predict(X_train))}")
    #     visualize_tree(truncated_tree, feature_names, target_names)
    
    fig = plot_dash_data(X_train, y_train, centers, dt)
    impurity_fig = plot_impurity_vs_depth(dt)
    print('Plotting impurity vs depth')
    return fig, impurity_fig, dt, X_train, y_train, centers, feature_names, target_names

data_fig, impurity_fig, dt, X_train, y_train, centers, feature_names, target_names = main()
app.layout = html.Div([
    html.Div([
        dcc.Graph(
            id='tree-plot',
            figure=plot_tree_graph(dt, feature_names, target_names),
            style={'width': '30%', 'display': 'inline-block'}
        ),
        dcc.Graph(
            id='impurity-vs-depth-plot',
            figure=impurity_fig,
            style={'width': '30%', 'display': 'inline-block'}
        ),
        dcc.Graph(
            id='data-decision-boundaries-plot',
            style={'width': '30%', 'display': 'inline-block'}
        )
    ]),
    html.Div([
        dcc.Graph(
            id='data-decision-boundaries-full-plot',
            figure=data_fig,
            style={'width': '32%', 'display': 'inline-block'}
        ),
    ])
])
server = app.server

@app.callback(
    Output('data-decision-boundaries-plot', 'figure'),
    Input('impurity-vs-depth-plot', 'hoverData')
)
def update_details_plot(hoverData):
    if hoverData is None:
        return go.Figure()

    point = hoverData['points'][0]
    # x_val = point['x']
    hovered_depth = point['y']
    # print(point)
    
    truncated_tree = truncate_tree_to_depth(dt, hovered_depth)
    data_fig = plot_dash_data(X_train, y_train, centers, truncated_tree, title_s='Decision regions for subtree')
    print(f'Plotting data with truncated decision tree with depth = {hovered_depth}')
    return data_fig

if __name__ == '__main__':
    app.run_server(debug=True)
