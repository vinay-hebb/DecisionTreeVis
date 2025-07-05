import os
import random
import numpy as np

def seed_everything(seed=1111):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def plot_dash_data(X, y, centers):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode='markers',
            marker=dict(
                color=y,
                colorscale='viridis',
            ),
            name='Data Points',
            visible=True
        )
    )
    
    # Add centers as a separate trace
    fig.add_trace(
        go.Scatter(
            x=centers[:, 0],
            y=centers[:, 1],
            mode='markers',
            marker=dict(
                size=15,
                symbol='x',
                color='red',
                line=dict(width=2, color='darkred')
            ),
            name='Centers',
            visible=True
        )
    )
    
    fig.update_layout(
        title_x=0.5,
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        showlegend=True,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        width=600,
        height=600,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig
