from sklearn.tree import _tree
import plotly.graph_objects as go

def plot_tree_graph(dt, feature_names=None, class_names=None):
    """
    Build a networkx-like graph structure for a decision tree and return a Plotly figure.
    """
    nodes = []
    edges = []
    labels = {}

    tree_ = dt.tree_
    n_nodes = tree_.node_count

    # Compute node positions for visualization (simple vertical tree)
    def compute_positions(node_id, depth, x, dx, positions):
        positions[node_id] = (x, -depth)
        left = tree_.children_left[node_id]
        right = tree_.children_right[node_id]
        if left != -1:
            compute_positions(left, depth+1, x - dx/(2**depth), dx, positions)
        if right != -1:
            compute_positions(right, depth+1, x + dx/(2**depth), dx, positions)

    positions = {}
    compute_positions(0, 0, 0, 2, positions)

    for node_id in range(n_nodes):
        x, y = positions[node_id]
        nodes.append((x, y))
        value = tree_.value[node_id][0]
        predicted_class_idx = value.argmax()
        if class_names is not None: predicted_class = class_names[predicted_class_idx]
        else:                       predicted_class = str(predicted_class_idx)
        if tree_.children_left[node_id] == _tree.TREE_LEAF:
            label = (
                f"Leaf<br>samples={value*tree_.n_node_samples[node_id]}"
                f"<br>predict: {predicted_class}"
            )
        else:
            feature = feature_names[tree_.feature[node_id]] if feature_names is not None else f"X[{tree_.feature[node_id]}]"
            threshold = tree_.threshold[node_id]
            label = f"{feature} <= {threshold:.2f}<br>samples={value*tree_.n_node_samples[node_id]}<br>predict: {predicted_class}"
        labels[node_id] = label

        # Edges
        for child in [tree_.children_left[node_id], tree_.children_right[node_id]]:
            if child != _tree.TREE_LEAF and child != -1:
                x0, y0 = positions[node_id]
                x1, y1 = positions[child]
                edges.append(((x0, y0), (x1, y1)))

    # Plotly scatter for nodes
    node_x = [x for x, y in nodes]
    node_y = [y for x, y in nodes]
    node_text = [labels[i] for i in range(n_nodes)]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[f"{i}" for i in range(n_nodes)],
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(size=30, color='lightblue', line=dict(width=2, color='darkblue')),
        textposition="middle center",
        showlegend=False
    )

    # Plotly scatter for edges
    edge_x = []
    edge_y = []
    for (x0, y0), (x1, y1) in edges:
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=2, color='gray'),
        hoverinfo='none',
        showlegend=False
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Decision Tree Structure",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        margin=dict(l=20, r=20, t=40, b=20),
        height=600
    )
    return fig

