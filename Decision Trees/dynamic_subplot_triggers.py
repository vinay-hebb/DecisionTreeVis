import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from utils import plot_impurity_vs_depth

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
    from utils import plot_dash_data
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
    
    fig, impurity_fig = plot_dash_data(X_train, y_train, centers, dt)
    print('Plotting impurity vs depth')
    return fig, impurity_fig, dt, X_train, y_train

data_fig, impurity_fig, dt, X_train, y_train = main()
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

    # point = hoverData['points'][0]
    # x_val = point['x']
    # y_val = point['y']
    
    # hovered_row = df[(df['sepal_width'] == x_val) & (df['sepal_length'] == y_val)]
    # if hovered_row.empty:
    #     return go.Figure()
    
    # hovered_row = hovered_row.iloc[0]
    # print(f"Hovered point: x={x_val}, y={y_val}, species={hovered_row['species']}")
    # filtered_df = df[df['species'] == hovered_row['species']]

    # fig = px.scatter(filtered_df, x='petal_width', y='petal_length', color='species',
    #                  title=f"Details for {hovered_row['species']}")
    print('Plotting data')
    fig = data_fig
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
