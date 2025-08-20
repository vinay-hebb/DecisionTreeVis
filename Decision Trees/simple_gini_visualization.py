import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import warnings
warnings.filterwarnings('ignore')

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


def visualize_tree(dt, feature_names, target_names):
    plt.figure()
    plot_tree(dt, 
             feature_names=feature_names,
             class_names=target_names,
             filled=True,
             rounded=True,
             fontsize=10)
    plt.title('Decision Tree with Gini Impurity Information', fontsize=14, pad=20)
    plt.show()

def truncate_tree_to_depth(tree, max_depth):
    """
    Truncate a trained decision tree to a specified maximum depth.
    
    Args:
        tree: sklearn DecisionTreeClassifier (already fitted)
        max_depth: int, maximum depth to truncate to
        
    Returns:
        sklearn DecisionTreeClassifier: truncated tree with same structure up to max_depth
    """
    from sklearn.tree import DecisionTreeClassifier
    
    # Create a new tree with the same parameters
    truncated_tree = DecisionTreeClassifier(
        criterion=tree.criterion,
        max_depth=max_depth,
        min_samples_split=tree.min_samples_split,
        min_samples_leaf=tree.min_samples_leaf,
        random_state=tree.random_state
    )
    
    # Create a deep copy of the tree structure
    from copy import deepcopy
    truncated_tree.tree_ = deepcopy(tree.tree_)
    
    # Copy essential attributes from the original tree
    truncated_tree.n_outputs_ = tree.n_outputs_
    truncated_tree.n_classes_ = tree.n_classes_
    truncated_tree.classes_ = tree.classes_
    
    # Get the tree structure
    tree_structure = truncated_tree.tree_
    
    # Find nodes at max_depth and convert them to leaves
    def truncate_nodes(node_id, current_depth):
        if current_depth >= max_depth:
            # Convert this node to a leaf
            tree_structure.children_left[node_id] = -1
            tree_structure.children_right[node_id] = -1
            return
        
        # Recursively process children if they exist
        left_child = tree_structure.children_left[node_id]
        right_child = tree_structure.children_right[node_id]
        
        if left_child != -1:
            truncate_nodes(left_child, current_depth + 1)
        if right_child != -1:
            truncate_nodes(right_child, current_depth + 1)
    
    # Start truncation from root
    truncate_nodes(0, 0)
    
    return truncated_tree
