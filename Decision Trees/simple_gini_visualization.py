"""
Simple Gini Impurity Reduction Visualization for Decision Trees

This script shows how to extract and visualize Gini impurity reduction
for each split in a scikit-learn DecisionTreeClassifier.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from utils import seed_everything

def extract_gini_impurity_info(tree_model):
    """
    Extract Gini impurity information for each node in the decision tree.
    
    Parameters:
    -----------
    tree_model : DecisionTreeClassifier
        A fitted DecisionTreeClassifier instance
    
    Returns:
    --------
    dict : Dictionary containing node information with impurity details
    """
    tree = tree_model.tree_
    node_info = {}
    
    for node_id in range(tree.node_count):
        # Get node properties
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        
        # Check if it's a leaf node
        is_leaf = left_child == -1
        
        # Get impurity and samples
        impurity = tree.impurity[node_id]
        n_samples = tree.n_node_samples[node_id]
        
        # Calculate weighted impurity reduction for split nodes
        impurity_reduction = 0.0
        if not is_leaf:
            left_impurity = tree.impurity[left_child]
            right_impurity = tree.impurity[right_child]
            left_samples = tree.n_node_samples[left_child]
            right_samples = tree.n_node_samples[right_child]
            
            # Weighted impurity reduction
            impurity_reduction = impurity - (
                (left_samples * left_impurity + right_samples * right_impurity) / n_samples
            )
        
        node_info[node_id] = {
            'node_id': node_id,
            'impurity': impurity,
            'n_samples': n_samples,
            'is_leaf': is_leaf,
            'impurity_reduction': impurity_reduction,
            'left_child': left_child,
            'right_child': right_child
        }
    
    return node_info

def calculate_node_depths(tree_model):
    """
    Calculate the depth of each node in the tree.
    
    Parameters:
    -----------
    tree_model : DecisionTreeClassifier
        A fitted DecisionTreeClassifier instance
    
    Returns:
    --------
    dict : Dictionary mapping node_id to depth
    """
    tree = tree_model.tree_
    depths = {0: 0}  # Root node has depth 0
    
    def calculate_depth(node_id, depth):
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        
        if left_child != -1:
            depths[left_child] = depth + 1
            calculate_depth(left_child, depth + 1)
        
        if right_child != -1:
            depths[right_child] = depth + 1
            calculate_depth(right_child, depth + 1)
    
    calculate_depth(0, 0)
    return depths

def visualize_gini_impurity_reduction(tree_model, feature_names=None):
    """
    Create comprehensive visualization of Gini impurity reduction.
    
    Parameters:
    -----------
    tree_model : DecisionTreeClassifier
        A fitted DecisionTreeClassifier instance
    feature_names : list, optional
        Names of features for better visualization
    """
    node_info = extract_gini_impurity_info(tree_model)
    depths = calculate_node_depths(tree_model)
    
    # Separate split nodes and leaf nodes
    split_nodes = []
    leaf_nodes = []
    
    for node_id, info in node_info.items():
        depth = depths[node_id]
        if info['is_leaf']:
            leaf_nodes.append((depth, info['impurity']))
        else:
            split_nodes.append((depth, info['impurity_reduction']))
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Impurity reduction by depth
    if split_nodes:
        depths_split, reductions = zip(*split_nodes)
        ax1.scatter(depths_split, reductions, alpha=0.7, s=100, color='blue')
        ax1.set_xlabel('Tree Depth')
        ax1.set_ylabel('Gini Impurity Reduction')
        ax1.set_title('Impurity Reduction by Tree Depth')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        if len(depths_split) > 1:
            z = np.polyfit(depths_split, reductions, 1)
            p = np.poly1d(z)
            ax1.plot(depths_split, p(depths_split), "r--", alpha=0.8, label='Trend')
            ax1.legend()
    
    # Plot 2: Node impurity by depth
    depths_leaf, impurities_leaf = zip(*leaf_nodes)
    ax2.scatter(depths_leaf, impurities_leaf, alpha=0.7, s=100, color='red', label='Leaf Nodes')
    
    if split_nodes:
        depths_split, _ = zip(*split_nodes)
        split_impurities = [node_info[node_id]['impurity'] for node_id in range(len(depths_split))]
        ax2.scatter(depths_split, split_impurities, alpha=0.7, s=100, color='green', label='Split Nodes')
    
    ax2.set_xlabel('Tree Depth')
    ax2.set_ylabel('Gini Impurity')
    ax2.set_title('Node Impurity by Tree Depth')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distribution of impurity reduction
    if split_nodes:
        _, reductions = zip(*split_nodes)
        ax3.hist(reductions, bins=min(8, len(reductions)), alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_xlabel('Impurity Reduction')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Impurity Reduction')
        ax3.grid(True, alpha=0.3)
        
        # Add statistics
        mean_reduction = np.mean(reductions)
        ax3.axvline(mean_reduction, color='red', linestyle='--', label=f'Mean: {mean_reduction:.4f}')
        ax3.legend()
    
    # Plot 4: Cumulative impurity reduction
    if split_nodes:
        # Sort by impurity reduction (descending)
        sorted_nodes = sorted(split_nodes, key=lambda x: x[1], reverse=True)
        _, reductions = zip(*sorted_nodes)
        cumulative_reduction = np.cumsum(reductions)
        
        ax4.plot(range(1, len(reductions) + 1), cumulative_reduction, 
                marker='o', linewidth=2, markersize=6)
        ax4.set_xlabel('Number of Splits')
        ax4.set_ylabel('Cumulative Impurity Reduction')
        ax4.set_title('Cumulative Impurity Reduction')
        ax4.grid(True, alpha=0.3)
        
        # Add percentage lines
        total_reduction = np.sum(reductions)
        ax4.axhline(y=total_reduction * 0.8, color='red', linestyle='--', 
                   alpha=0.7, label='80% of total')
        ax4.axhline(y=total_reduction * 0.9, color='orange', linestyle='--', 
                   alpha=0.7, label='90% of total')
        ax4.legend()
    
    plt.tight_layout()
    plt.show()

def print_detailed_split_analysis(tree_model, feature_names=None):
    """
    Print detailed analysis of all splits in the tree.
    
    Parameters:
    -----------
    tree_model : DecisionTreeClassifier
        A fitted DecisionTreeClassifier instance
    feature_names : list, optional
        Names of features
    """
    node_info = extract_gini_impurity_info(tree_model)
    depths = calculate_node_depths(tree_model)
    
    # Get split nodes only
    split_nodes = {k: v for k, v in node_info.items() if not v['is_leaf']}
    
    if not split_nodes:
        print("No split nodes found in the tree.")
        return
    
    # Sort by impurity reduction (descending)
    sorted_nodes = sorted(split_nodes.items(), 
                        key=lambda x: x[1]['impurity_reduction'], 
                        reverse=True)
    
    print("=" * 80)
    print("DETAILED SPLIT ANALYSIS")
    print("=" * 80)
    
    print(f"{'Node':<6} {'Depth':<6} {'Samples':<8} {'Impurity':<10} {'Reduction':<12} {'% of Total':<12}")
    print("-" * 80)
    
    total_reduction = sum(node[1]['impurity_reduction'] for node in sorted_nodes)
    
    for node_id, info in sorted_nodes:
        depth = depths[node_id]
        reduction = info['impurity_reduction']
        percentage = (reduction / total_reduction) * 100
        
        print(f"{node_id:<6} {depth:<6} {info['n_samples']:<8} "
              f"{info['impurity']:<10.4f} {reduction:<12.4f} {percentage:<12.2f}%")
    
    print("-" * 80)
    print(f"Total impurity reduction: {total_reduction:.4f}")
    print(f"Number of splits: {len(split_nodes)}")
    print(f"Average impurity reduction per split: {total_reduction/len(split_nodes):.4f}")
    
    # Find how many splits give 80% and 90% of total reduction
    reductions = [node[1]['impurity_reduction'] for node in sorted_nodes]
    cumulative_reduction = np.cumsum(reductions)
    
    splits_80 = np.argmax(cumulative_reduction >= total_reduction * 0.8) + 1
    splits_90 = np.argmax(cumulative_reduction >= total_reduction * 0.9) + 1
    
    print(f"80% of total impurity reduction achieved with {splits_80} splits")
    print(f"90% of total impurity reduction achieved with {splits_90} splits")

def main():
    """
    Main function to demonstrate Gini impurity visualization.
    """
    print("Gini Impurity Reduction Visualization")
    print("=" * 50)
    
    # Load dataset using data_generator
    from data_generator import generate_data
    
    X, y, _ = generate_data(2)
    feature_names, target_names = ['X0', 'X1'], ['1', '0']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y
    )
    
    # Create and fit decision tree
    print("Training Decision Tree Classifier...")
    dt = DecisionTreeClassifier(
        criterion='gini',
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    dt.fit(X_train, y_train)
    
    # Print basic information
    print(f"Tree depth: {dt.get_depth()}")
    print(f"Number of leaves: {dt.get_n_leaves()}")
    print(f"Training accuracy: {dt.score(X_train, y_train):.4f}")
    print(f"Test accuracy: {dt.score(X_test, y_test):.4f}")
    print()
    
    # Print detailed split analysis
    print_detailed_split_analysis(dt, feature_names)
    print()
    
    # Create visualizations
    print("Creating visualizations...")
    visualize_gini_impurity_reduction(dt, feature_names)
    
    # Show the tree structure
    print("Decision Tree Structure:")
    print("=" * 40)
    tree_text = export_text(dt, feature_names=feature_names, show_weights=True)
    print(tree_text)
    
    # Visualize the tree
    plt.figure(figsize=(20, 12))
    plot_tree(dt, 
             feature_names=feature_names,
             class_names=target_names,
             filled=True,
             rounded=True,
             fontsize=10)
    plt.title('Decision Tree with Gini Impurity Information', fontsize=14, pad=20)
    plt.show()

if __name__ == "__main__":
    seed_everything()
    main() 