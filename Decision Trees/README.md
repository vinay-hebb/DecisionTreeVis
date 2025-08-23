# Decision Tree Classification with Interactive Visualization

This project demonstrates decision tree classification using synthetically generated data with interactive visualizations and Gini impurity analysis.

## Files Overview

- `app.py` - Interactive Dash web application for decision tree visualization
- `data_generator.py` - Generates synthetic data with configurable clusters and chessboard patterns
- `utils.py` - Utility functions for plotting and impurity analysis
- `simple_gini_visualization.py` - Gini impurity visualization and tree truncation utilities
- `plot_tree.py` - Interactive Plotly-based decision tree structure visualization
- `requirements.txt` - Required Python packages

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the interactive web application:**
   ```bash
   python app.py
   ```

3. **Generate and visualize data:**
   ```bash
   python data_generator.py
   ```

## Scripts Description

### `app.py`
An interactive Dash web application that provides:
- Interactive decision tree visualization with Plotly
- Gini impurity vs. depth analysis
- Configurable random seed for reproducible data generation
- Real-time data visualization with decision boundaries
- Support for different data generation patterns (clusters, chessboard)

### `data_generator.py`
Generates synthetic datasets including:
- Configurable number of clusters (2-8)
- Chessboard pattern data
- Blob-based cluster data
- Configurable sample sizes and cluster proportions
- Reproducible data generation with random seeds

### `utils.py`
Utility functions for:
- Setting random seeds for reproducibility
- Plotting impurity vs. depth relationships
- Creating interactive data visualizations with Plotly
- Decision boundary plotting

### `simple_gini_visualization.py`
Gini impurity analysis tools:
- Extract Gini impurity information from decision trees
- Calculate node depths and impurity reductions
- Visualize trees with matplotlib
- Tree truncation utilities for depth analysis

### `plot_tree.py`
Interactive decision tree visualization:
- Creates Plotly-based tree structure plots
- Shows node information, splits, and predictions
- Interactive hover information for each node
- Clean tree layout with proper positioning

## Key Features

- **Interactive Web Interface**: Dash-based application with real-time updates
- **Decision Tree Visualization**: Multiple visualization approaches (matplotlib and Plotly)
- **Gini Impurity Analysis**: Detailed impurity analysis across tree depths
- **Synthetic Data Generation**: Multiple data generation strategies
- **Reproducible Results**: Configurable random seeds for consistent outputs
- **Multiple Data Patterns**: Support for clusters, chessboard, and blob patterns

## Data Generation

The `data_generator.py` module creates synthetic data with:
- Configurable number of clusters (2-8)
- Chessboard patterns with customizable dimensions
- Blob-based cluster data with Gaussian distributions
- Configurable sample sizes and cluster proportions
- Reproducible generation using random seeds

## Example Usage

```python
from data_generator import generate_data, generate_chessboard_data
from sklearn.tree import DecisionTreeClassifier
from utils import seed_everything

# Set seed for reproducibility
seed_everything(42)

# Generate different types of data
X, y, centers = generate_data(n_clusters=3, total_samples=16)
# or
X, y = generate_chessboard_data(total_samples=16, n_rows=2, n_cols=4)

# Train classifier
dt = DecisionTreeClassifier(max_depth=7, random_state=42)
dt.fit(X_train, y_train)
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

- The application uses a fixed random seed (42) by default for reproducibility
- Decision trees are limited to max_depth=7 to prevent overfitting
- All visualizations are interactive and use Plotly for enhanced user experience
- The classification uses stratified sampling to maintain class balance
- Multiple visualization approaches are available for different use cases 