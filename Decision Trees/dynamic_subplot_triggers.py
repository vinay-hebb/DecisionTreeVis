import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from dash import callback_context

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from utils import plot_dash_data, plot_impurity_vs_depth, seed_everything
from simple_gini_visualization import truncate_tree_to_depth, visualize_tree
from plot_tree import plot_tree_graph
# seed_everything(132)  # Balanced tree, explainable
# seed_everything(12)    # Interesting yet confusing example where root and its child have same predictions
# seed_everything(142)    # Unbalanced but simple to explain as cut is only along X1
# seed_everything(191)    # Interesting yet confusing example where root and its child have same predictions
# seed_everything(11)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# Configuration parameters
BASE = 0.4  # Decay factor - each cluster has 40% of the previous cluster's size
DEFAULT_N_CLUSTERS = 2
TOTAL_SAMPLES = 16

def calculate_proportions(n_clusters: int) -> np.ndarray:
    """Calculate cluster proportions based on decay factor."""
    proportions = [BASE**i for i in range(n_clusters)]
    proportions = np.array(proportions) / np.sum(proportions)
    return proportions

def main(seed):
    """
    Main function to demonstrate Gini impurity visualization.
    """
    seed_everything(seed)
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

X_train, y_train, centers, dt = None, None, None, None
app.layout = html.Div([
    html.Div([
        html.Label(
            'Random Seed (for reproducible data generation):',
            style={'marginRight': '10px', 'fontWeight': 'bold'}
        ),
        dcc.Input(
            id='seed-input',
            type='number',
            placeholder='Enter random seed',
            value=None,
            style={'marginRight': '10px'}
        ),
        html.Button('Submit', id='submit-seed', n_clicks=0),
    ]),
    html.Div([
        dcc.Graph(
            id='tree-plot',
            style={'width': '30%', 'display': 'inline-block'},
            config={'displayModeBar': True, 'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines', 'toImage', 'sendDataToCloud', 'toggleHover', 'resetViews', 'resetViewMapbox', 'zoom3d', 'pan3d', 'orbitRotation', 'tableRotation', 'resetCameraDefault3d', 'resetCameraLastSave3d', 'hoverClosest3d', 'hoverClosestGeo', 'hoverClosestGl2d', 'hoverClosestPie', 'toggleHover', 'resetViews', 'resetViewMapbox', 'zoom3d', 'pan3d', 'orbitRotation', 'tableRotation', 'resetCameraDefault3d', 'resetCameraLastSave3d', 'hoverClosest3d', 'hoverClosestGeo', 'hoverClosestGl2d', 'hoverClosestPie']}
        ),
        dcc.Graph(
            id='impurity-vs-depth-plot',
            style={'width': '30%', 'display': 'inline-block'},
            config={'displayModeBar': True, 'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines', 'toImage', 'sendDataToCloud', 'toggleHover', 'resetViews', 'resetViewMapbox', 'zoom3d', 'pan3d', 'orbitRotation', 'tableRotation', 'resetCameraDefault3d', 'resetCameraLastSave3d', 'hoverClosest3d', 'hoverClosestGeo', 'hoverClosestGl2d', 'hoverClosestPie', 'toggleHover', 'resetViews', 'resetViewMapbox', 'zoom3d', 'pan3d', 'orbitRotation', 'tableRotation', 'resetCameraDefault3d', 'resetCameraLastSave3d', 'hoverClosest3d', 'hoverClosestGeo', 'hoverClosestGl2d', 'hoverClosestPie']}
        ),
        dcc.Graph(
            id='data-decision-boundaries-plot',
            style={'width': '30%', 'display': 'inline-block'},
            config={'displayModeBar': True, 'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines', 'toImage', 'sendDataToCloud', 'toggleHover', 'resetViews', 'resetViewMapbox', 'zoom3d', 'pan3d', 'orbitRotation', 'tableRotation', 'resetCameraDefault3d', 'resetCameraLastSave3d', 'hoverClosest3d', 'hoverClosestGeo', 'hoverClosestGl2d', 'hoverClosestPie', 'toggleHover', 'resetViews', 'resetViewMapbox', 'zoom3d', 'pan3d', 'orbitRotation', 'tableRotation', 'resetCameraDefault3d', 'resetCameraLastSave3d', 'hoverClosest3d', 'hoverClosestGeo', 'hoverClosestGl2d', 'hoverClosestPie']}
        )
    ]),
])
server = app.server
@app.callback(
    [
        Output('tree-plot', 'figure'),
        Output('impurity-vs-depth-plot', 'figure'),
        Output('data-decision-boundaries-plot', 'figure'),
    ],
    [
        Input('submit-seed', 'n_clicks'),
        Input('impurity-vs-depth-plot', 'hoverData')
    ],
    State('seed-input', 'value')
)
def update_all(n_clicks, hoverData, seed):
    # On initial load, just return empty figures
    if n_clicks == 0 or seed is None:
        return go.Figure(), go.Figure(), go.Figure()

    global X_train, y_train, centers, dt

    ctx = callback_context
    if ctx.triggered and ctx.triggered[0]['prop_id'].startswith('impurity-vs-depth-plot.hoverData') and \
        hoverData is not None and 'points' in hoverData and len(hoverData['points']) > 0:
        point = hoverData['points'][0]
        hovered_depth = point['y']
        truncated_tree = truncate_tree_to_depth(dt, hovered_depth)
        data_decision_fig = plot_dash_data(
            X_train, y_train, centers, truncated_tree, title_s='Decision regions for subtree'
        )
        print(f'Plotting data with truncated decision tree with depth = {hovered_depth}')
        tree_fig = dash.no_update
        impurity_fig = dash.no_update
    else:
        data_fig, impurity_fig, dt, X_train, y_train, centers, feature_names, target_names = main(seed)
        tree_fig = plot_tree_graph(dt, feature_names, target_names)
        # Default: show full tree's decision boundaries
        data_decision_fig = plot_dash_data(
            X_train, y_train, centers, dt, title_s='Decision regions for full tree'
        )

    return tree_fig, impurity_fig, data_decision_fig

if __name__ == '__main__':
    app.run(debug=True, port=2221)
