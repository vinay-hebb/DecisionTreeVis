import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from utils import plot_dash_data, plot_impurity_vs_depth
from simple_gini_visualization import truncate_tree_to_depth, visualize_tree

app = dash.Dash(__name__)
def main():
    """
    Main function to demonstrate Gini impurity visualization.
    """
    print("Gini Impurity Reduction Visualization")
    print("=" * 50)
    
    # Load dataset using data_generator
    from data_generator import generate_data, generate_blobs_data
    
    # X, y, _ = generate_data(2)
    centers = np.array([[-1, 0], [1, 0]])
    X, y = generate_blobs_data(centers)
    feature_names, target_names = ['X0', 'X1'], ['1', '0']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y
    )
    # fig = plot_dash_data(X_train, y_train, centers)
    # fig.write_html('my_plot.html', auto_open=True)
    
    # Create and fit decision tree
    print("Training Decision Tree Classifier...")
    dt = DecisionTreeClassifier(
        criterion='gini',
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    dt.fit(X_train, y_train)
    # print(f"depth={None}, y={np.sum(dt.predict(X_train))}")
    # visualize_tree(dt, ['X0', 'X1'], ['0', '1'])
    # for d in [0,1,2]:
    #     truncated_tree = truncate_tree_to_depth(dt, d)
    #     print(f"depth={d}, y={np.sum(truncated_tree.predict(X_train))}")
    #     visualize_tree(truncated_tree, ['X0', 'X1'], ['0', '1'])
    
    fig = plot_dash_data(X_train, y_train, centers, dt)
    impurity_fig = plot_impurity_vs_depth(dt)
    print('Plotting impurity vs depth')
    return fig, impurity_fig, dt, X_train, y_train, centers

data_fig, impurity_fig, dt, X_train, y_train, centers = main()
app.layout = html.Div([
    html.Div([
        dcc.Graph(
            id='data-decision-boundaries-full-plot',
            figure=data_fig,
            style={'width': '32%', 'display': 'inline-block'}
        ),
        dcc.Graph(
            id='impurity-vs-depth-plot',
            figure=impurity_fig,
            style={'width': '32%', 'display': 'inline-block'}
        ),
        dcc.Graph(
            id='data-decision-boundaries-plot',
            style={'width': '32%', 'display': 'inline-block'}
        )
    ])
])

@app.callback(
    Output('data-decision-boundaries-plot', 'figure'),
    Input('impurity-vs-depth-plot', 'hoverData')
)
def update_details_plot(hoverData):
    if hoverData is None:
        return go.Figure()

    point = hoverData['points'][0]
    x_val = point['x']
    # y_val = point['y']
    # print(point)
    
    # hovered_row = df[(df['sepal_width'] == x_val) & (df['sepal_length'] == y_val)]
    # if hovered_row.empty:
    #     return go.Figure()
    
    # hovered_row = hovered_row.iloc[0]
    # print(f"Hovered point: x={x_val}, y={y_val}, species={hovered_row['species']}")
    # filtered_df = df[df['species'] == hovered_row['species']]

    # fig = px.scatter(filtered_df, x='petal_width', y='petal_length', color='species',
    #                  title=f"Details for {hovered_row['species']}")
    truncated_tree = truncate_tree_to_depth(dt, x_val)
    # new tree seems to be generating but contour does not change
    data_fig = plot_dash_data(X_train, y_train, centers, truncated_tree)
    # print(f"{data_fig.data[2].name}, z={np.sum(data_fig.data[2].z)}, id={id(truncated_tree)}, {truncated_tree}")
    print(f'Plotting data with truncated decision tree with depth = {x_val}')
    return data_fig

if __name__ == '__main__':
    app.run_server(debug=True)
