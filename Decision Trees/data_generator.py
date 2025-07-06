import numpy as np
from sklearn.datasets import make_blobs
from typing import Tuple, Optional

def generate_blobs_data(centers: Optional[np.ndarray] = None, 
                       sample_sizes: Optional[list[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate blob data with specified centers and sample sizes.
    
    Args:
        centers: Array of cluster centers, shape (n_clusters, n_features)
        sample_sizes: List of sample sizes for each cluster
        
    Returns:
        Tuple of (X, y) where X is the feature matrix and y contains cluster labels
    """
    if centers is None:
        raise ValueError("centers must be provided")
    
    n_clusters = len(centers)
    if sample_sizes is None:
        sample_sizes = [100] * n_clusters
    
    X_list = []
    y_list = []
    
    for i in range(n_clusters):
        center = centers[i].reshape(1, -1)
        cluster_data = make_blobs(
            n_samples=sample_sizes[i], 
            centers=center, 
            cluster_std=1.0,
        )
        cluster_X, _ = cluster_data[0], cluster_data[1]
        X_list.append(cluster_X)
        y_list.append(np.full(sample_sizes[i], i))  # Assign cluster labels
    
    # Combine all clusters
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    return X, y

def generate_data(n_clusters: int, n_samples: int, proportions) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate complete dataset with specified number of clusters.
    
    Args:
        n_clusters: Number of clusters to generate
        
    Returns:
        Tuple of (X, y, centers) where X is the feature matrix, 
        y contains cluster labels, and centers are the cluster centers
    """
    
    # Calculate sample sizes based on proportions
    sample_sizes = [int(prop * n_samples) for prop in proportions]
    sample_sizes[0] += n_samples - sum(sample_sizes)  # Ensure total samples is correct
    
    # Generate random centers
    centers = np.random.uniform(-5, 5, size=(n_clusters, 2))
    
    # Generate the data
    X, y = generate_blobs_data(centers, sample_sizes)
    
    return X, y, centers

def generate_chessboard_data(n_samples: int = 500, n_rows: int = 2, n_cols: int = 4, stdev: float = 1.0):
    """
    Generate a classification dataset with centers arranged like a chessboard (2 rows x 4 columns).
    Returns X, y, centers.
    """
    from sklearn.datasets import make_classification
    import numpy as np

    n_clusters = n_rows * n_cols
    n_classes = 2  # For chessboard, alternate classes

    # Generate centers in a grid (chessboard)
    x_coords = np.linspace(-6, 6, n_cols)
    y_coords = np.linspace(-3, 3, n_rows)
    centers = np.array([[x, y] for y in y_coords for x in x_coords])

    # Assign class labels in a chessboard pattern
    labels = []
    for row in range(n_rows):
        for col in range(n_cols):
            # Alternate classes like a chessboard
            labels.append((row + col) % n_classes)
    labels = np.array(labels)

    # Assign samples per cluster as evenly as possible
    samples_per_cluster = [n_samples // n_clusters] * n_clusters
    for i in range(n_samples % n_clusters):
        samples_per_cluster[i] += 1

    X_list = []
    y_list = []
    for i, (center, label) in enumerate(zip(centers, labels)):
        Xi = center + np.random.randn(samples_per_cluster[i], 2) * stdev
        X_list.append(Xi)
        y_list.append(np.full(samples_per_cluster[i], label))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y, centers

if __name__ == "__main__":
    from utils import seed_everything
    from dynamic_subplot_triggers import calculate_proportions
    seed_everything()
    BASE = 0.4  # Decay factor - each cluster has 40% of the previous cluster's size
    DEFAULT_N_CLUSTERS = 2
    TOTAL_SAMPLES = 500
    # Test the data generation
    print("Testing data generation...")
    proportions = calculate_proportions(DEFAULT_N_CLUSTERS)
    print(f"Proportions for {DEFAULT_N_CLUSTERS} clusters: {proportions}")
    
    X, y, centers = generate_data(DEFAULT_N_CLUSTERS, TOTAL_SAMPLES, proportions)
    print(f"Generated data shape: {X.shape}")
    print(f"Number of unique labels: {len(np.unique(y))}")
    print(f"Centers shape: {centers.shape}") 