import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

df = px.data.iris()

app = dash.Dash(__name__)
app.layout = html.Div([
    html.Div([
        dcc.Graph(
            id='scatter-plot',
            figure=px.scatter(df, x='sepal_width', y='sepal_length', color='species'),
            style={'width': '48%', 'display': 'inline-block'}
        ),
        dcc.Graph(
            id='details-plot',
            style={'width': '48%', 'display': 'inline-block'}
        )
    ])
])

@app.callback(
    Output('details-plot', 'figure'),
    Input('scatter-plot', 'hoverData')
)
def update_details_plot(hoverData):
    if hoverData is None:
        return go.Figure()

    point = hoverData['points'][0]
    x_val = point['x']
    y_val = point['y']
    
    hovered_row = df[(df['sepal_width'] == x_val) & (df['sepal_length'] == y_val)]
    if hovered_row.empty:
        return go.Figure()
    
    hovered_row = hovered_row.iloc[0]
    print(f"Hovered point: x={x_val}, y={y_val}, species={hovered_row['species']}")
    filtered_df = df[df['species'] == hovered_row['species']]

    fig = px.scatter(filtered_df, x='petal_width', y='petal_length', color='species',
                     title=f"Details for {hovered_row['species']}")

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
