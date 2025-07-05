"""
Simple example of using decision tree classification with generated data.
This script demonstrates the basic workflow without all the advanced visualizations.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from data_generator import generate_data

def simple_decision_tree_example():
    """Simple example of decision tree classification."""
    print("Simple Decision Tree Classification Example")
    print("=" * 40)
    
    # Generate data with 3 clusters
    print("Generating data with 3 clusters...")
    X, y, centers = generate_data(n_clusters=3)
    print(f"Generated {len(X)} samples")
    print(f"Data shape: {X.shape}")
    print(f"Number of unique classes: {len(np.unique(y))}")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create and train decision tree
    print("\nTraining decision tree...")
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(X_train, y_train)
    
    # Make predictions
    y_pred = dt.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print tree information
    print(f"\nTree depth: {dt.get_depth()}")
    print(f"Number of leaves: {dt.get_n_leaves()}")
    
    # Show feature importances
    feature_importances = dt.feature_importances_
    print(f"\nFeature importances:")
    print(f"Feature 1: {feature_importances[0]:.4f}")
    print(f"Feature 2: {feature_importances[1]:.4f}")
    
    return dt, X, y, centers

if __name__ == "__main__":
    # Run the simple example
    classifier, X, y, centers = simple_decision_tree_example()
    
    print("\n" + "="*40)
    print("Example completed successfully!")
    print("You can now run the full script 'decision_tree_classifier.py'")
    print("for more advanced visualizations and analysis.") 