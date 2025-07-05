import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import plotly.graph_objects as go
import plotly.express as px
from data_generator import generate_data

def create_decision_tree_classifier(X, y, max_depth=None, random_state=42):
    """
    Create and train a decision tree classifier.
    
    Args:
        X: Feature matrix
        y: Target labels
        max_depth: Maximum depth of the tree (None for unlimited)
        random_state: Random seed for reproducibility
        
    Returns:
        Trained DecisionTreeClassifier
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    
    # Create and train the decision tree
    dt_classifier = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state,
        criterion='gini'
    )
    dt_classifier.fit(X_train, y_train)
    
    return dt_classifier, X_train, X_test, y_train, y_test

def evaluate_classifier(classifier, X_test, y_test):
    """
    Evaluate the classifier and print metrics.
    
    Args:
        classifier: Trained classifier
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = classifier.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'confusion_matrix': cm,
        'classification_report': report
    }

def visualize_decision_boundary(classifier, X, y, centers, title="Decision Tree Decision Boundary"):
    """
    Create an interactive plot showing the decision boundary of the classifier.
    
    Args:
        classifier: Trained classifier
        X: Feature matrix
        y: Target labels
        centers: Cluster centers
        title: Plot title
    """
    # Create a mesh grid for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Make predictions on the mesh grid
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Create the plot
    fig = go.Figure()
    
    # Add decision boundary
    fig.add_trace(
        go.Contour(
            x=xx[0, :],
            y=yy[:, 0],
            z=Z,
            colorscale='viridis',
            opacity=0.3,
            showscale=False,
            name='Decision Boundary'
        )
    )
    
    # Add data points
    fig.add_trace(
        go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode='markers',
            marker=dict(
                color=y,
                colorscale='viridis',
                size=8,
                line=dict(width=1, color='white')
            ),
            name='Data Points',
            text=[f'Cluster {label}' for label in y],
            hovertemplate='<b>Cluster %{text}</b><br>' +
                         'Feature 1: %{x:.2f}<br>' +
                         'Feature 2: %{y:.2f}<extra></extra>'
        )
    )
    
    # Add centers
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
            text=[f'Center {i}' for i in range(len(centers))],
            hovertemplate='<b>%{text}</b><br>' +
                         'Feature 1: %{x:.2f}<br>' +
                         'Feature 2: %{y:.2f}<extra></extra>'
        )
    )
    
    fig.update_layout(
        title=title,
        title_x=0.5,
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        showlegend=True,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        width=700,
        height=600,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def plot_confusion_matrix(cm, class_names=None):
    """
    Create a confusion matrix plot.
    
    Args:
        cm: Confusion matrix
        class_names: Names of the classes
    """
    if class_names is None:
        class_names = [f'Cluster {i}' for i in range(len(cm))]
    
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        x=class_names,
        y=class_names,
        title="Confusion Matrix",
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        title_x=0.5,
        xaxis_title="Predicted",
        yaxis_title="Actual"
    )
    
    return fig

def main():
    """Main function to run the decision tree classification."""
    print("Decision Tree Classification with Generated Data")
    print("=" * 50)
    
    # Generate data with different numbers of clusters
    n_clusters_list = [2, 3, 4, 5]
    
    for n_clusters in n_clusters_list:
        print(f"\n{'='*20} {n_clusters} Clusters {'='*20}")
        
        # Generate data
        X, y, centers = generate_data(n_clusters)
        print(f"Generated {len(X)} samples with {n_clusters} clusters")
        
        # Create and train classifier
        classifier, X_train, X_test, y_train, y_test = create_decision_tree_classifier(
            X, y, max_depth=5
        )
        
        # Evaluate the classifier
        results = evaluate_classifier(classifier, X_test, y_test)
        
        # Print tree information
        print(f"\nTree depth: {classifier.get_depth()}")
        print(f"Number of leaves: {classifier.get_n_leaves()}")
        
        # Create visualizations
        fig_boundary = visualize_decision_boundary(
            classifier, X, y, centers, 
            f"Decision Tree Decision Boundary - {n_clusters} Clusters"
        )
        
        fig_cm = plot_confusion_matrix(
            results['confusion_matrix'],
            [f'Cluster {i}' for i in range(n_clusters)]
        )
        
        # Save plots
        fig_boundary.write_html(f"decision_boundary_{n_clusters}_clusters.html")
        fig_cm.write_html(f"confusion_matrix_{n_clusters}_clusters.html")
        
        print(f"Plots saved as:")
        print(f"  - decision_boundary_{n_clusters}_clusters.html")
        print(f"  - confusion_matrix_{n_clusters}_clusters.html")

if __name__ == "__main__":
    main() 