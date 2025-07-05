# Decision Tree Classification with Generated Data

This project demonstrates decision tree classification using synthetically generated data from the `data_generator` module.

## Files Overview

- `data_generator.py` - Generates synthetic blob data with configurable clusters
- `decision_tree_classifier.py` - Full-featured decision tree classification with visualizations
- `simple_example.py` - Basic example showing core functionality
- `toy.py` - Interactive Dash app for blob visualization
- `requirements.txt` - Required Python packages

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the simple example:**
   ```bash
   python simple_example.py
   ```

3. **Run the full classification script:**
   ```bash
   python decision_tree_classifier.py
   ```

4. **Run the interactive visualization:**
   ```bash
   python toy.py
   ```

## Scripts Description

### `simple_example.py`
A basic example that demonstrates:
- Data generation with 3 clusters
- Training a decision tree classifier
- Basic evaluation metrics (accuracy, classification report)
- Feature importance analysis

### `decision_tree_classifier.py`
A comprehensive script that:
- Tests classification with 2-5 clusters
- Creates interactive decision boundary visualizations
- Generates confusion matrix plots
- Saves results as HTML files
- Provides detailed evaluation metrics

### `toy.py`
An interactive Dash web application that:
- Allows users to adjust the number of clusters via a slider
- Shows real-time blob visualization
- Displays cluster centers and data points

## Output Files

The `decision_tree_classifier.py` script generates:
- `decision_boundary_X_clusters.html` - Interactive decision boundary plots
- `confusion_matrix_X_clusters.html` - Confusion matrix visualizations

## Data Generation

The `data_generator.py` module creates synthetic data with:
- Configurable number of clusters (2-8)
- Decay factor for cluster sizes (40% reduction per cluster)
- 20 total samples distributed across clusters
- 2D feature space with random centers

## Key Features

- **Decision Tree Classification**: Uses scikit-learn's DecisionTreeClassifier
- **Interactive Visualizations**: Plotly-based plots with hover information
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score
- **Decision Boundaries**: Visual representation of how the tree splits the feature space
- **Confusion Matrices**: Detailed error analysis

## Example Usage

```python
from data_generator import generate_data
from sklearn.tree import DecisionTreeClassifier

# Generate data
X, y, centers = generate_data(n_clusters=3)

# Train classifier
dt = DecisionTreeClassifier(max_depth=4)
dt.fit(X_train, y_train)

# Make predictions
predictions = dt.predict(X_test)
```

## Requirements

- Python 3.7+
- numpy
- scikit-learn
- matplotlib
- seaborn
- plotly
- dash

## Notes

- The data generator uses a fixed random seed (42) for reproducibility
- Decision trees are limited to max_depth=5 to prevent overfitting
- All visualizations are interactive and saved as HTML files
- The classification uses stratified sampling to maintain class balance 