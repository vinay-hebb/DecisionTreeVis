import os
import random
import numpy as np

def seed_everything(seed=1111):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def plot_impurity_vs_depth(tree):
    import plotly.graph_objects as go
    from simple_gini_visualization import extract_gini_impurity_info, calculate_node_depths
    node_info = extract_gini_impurity_info(tree)
    depths = calculate_node_depths(tree)
    
    # Separate split nodes and leaf nodes
    split_nodes = []
    leaf_nodes = []
    
    # Calculate weighted impurity for each depth
    depth_impurities = {}
    depth_samples = {}
    
    for node_id, info in node_info.items():
        depth = depths[node_id]
        impurity = info['impurity']
        samples = info['n_samples']
        
        if depth not in depth_impurities:
            depth_impurities[depth] = []
            depth_samples[depth] = []
        
        depth_impurities[depth].append(impurity)
        depth_samples[depth].append(samples)
    
    # Calculate weighted average impurity for each depth
    weighted_impurities = []
    depths_list = []
    
    total_samples = np.sum(depth_samples[0])
    for depth in sorted(depth_impurities.keys()):
        impurities = depth_impurities[depth]
        samples = depth_samples[depth]
        
        # # Calculate weighted average impurity
        # total_samples = sum(samples)
        weighted_impurity = sum(imp * sample for imp, sample in zip(impurities, samples)) / total_samples
        
        weighted_impurities.append(weighted_impurity)
        depths_list.append(depth)
    
    # Create impurity vs depth figure
    impurity_fig = go.Figure()
    
    # Add weighted impurity line
    impurity_fig.add_trace(
        go.Scatter(
            x=weighted_impurities,
            y=depths_list,
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(size=8, color='blue'),
            name='Weighted Average Impurity',
            visible=True,
            showlegend=False,
        )
    )
    
    impurity_fig.update_layout(
        title='Node Impurity vs Tree Depth',
        xaxis_title='Gini Impurity',
        yaxis_title='Tree Depth',
        showlegend=True,
        # width=600,
        height=600,
        margin=dict(l=50, r=50, t=50, b=50),
        yaxis=dict(autorange='reversed'),
        xaxis=dict(autorange='reversed'),
    )
    return impurity_fig

def plot_dash_data(X, y, centers, tree=None, feature_names=['X0', 'X1'], target_names=[0, 1], title_s=None):
    import plotly.graph_objects as go
    import numpy as np

    fig = go.Figure()
    
    # Add data points
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
    if centers is not None:
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
    
    # Add decision boundaries if tree is provided
    if tree is not None:
        # Create a mesh grid for decision boundaries
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100),
            np.linspace(y_min, y_max, 100)
        )
        
        # Get predictions for the mesh grid
        Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
        # print(f'plot_dash_data: {np.sum(Z)}')
        Z = Z.reshape(xx.shape)
        
        # Add decision boundary contour
        fig.add_trace(
            go.Contour(
                x=np.linspace(x_min, x_max, 100),
                y=np.linspace(y_min, y_max, 100),
                z=Z,
                colorscale='viridis',
                opacity=0.3,
                showscale=False,
                name='Decision Boundary',
                visible=True
            )
        )
    
    fig.update_layout(
        title=title_s,
        title_x=0.5,
        xaxis_title=feature_names[0],
        yaxis_title=feature_names[1],
        showlegend=True,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        width=600,
        height=600,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig
