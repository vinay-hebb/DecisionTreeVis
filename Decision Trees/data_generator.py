import numpy as np
from sklearn.datasets import make_blobs
from typing import Tuple, Optional

# Set random seed for reproducibility
np.random.seed(42)

# Configuration parameters
BASE = 0.4  # Decay factor - each cluster has 40% of the previous cluster's size
DEFAULT_N_CLUSTERS = 3
TOTAL_SAMPLES = 20

def calculate_proportions(n_clusters: int) -> np.ndarray:
    """Calculate cluster proportions based on decay factor."""
    proportions = [BASE**i for i in range(n_clusters)]
    proportions = np.array(proportions) / np.sum(proportions)
    return proportions

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
        cluster_X, cluster_y = cluster_data[0], cluster_data[1]
        X_list.append(cluster_X)
        y_list.append(np.full(sample_sizes[i], i))  # Assign cluster labels
    
    # Combine all clusters
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    return X, y

def generate_data(n_clusters: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate complete dataset with specified number of clusters.
    
    Args:
        n_clusters: Number of clusters to generate
        
    Returns:
        Tuple of (X, y, centers) where X is the feature matrix, 
        y contains cluster labels, and centers are the cluster centers
    """
    # Calculate proportions for the given number of clusters
    proportions = calculate_proportions(n_clusters)
    
    # Calculate sample sizes based on proportions
    sample_sizes = [int(prop * TOTAL_SAMPLES) for prop in proportions]
    sample_sizes[0] += TOTAL_SAMPLES - sum(sample_sizes)  # Ensure total samples is correct
    
    # Generate random centers
    centers = np.random.uniform(-5, 5, size=(n_clusters, 2))
    
    # Generate the data
    X, y = generate_blobs_data(centers, sample_sizes)
    
    return X, y, centers

if __name__ == "__main__":
    # Test the data generation
    print("Testing data generation...")
    proportions = calculate_proportions(DEFAULT_N_CLUSTERS)
    print(f"Proportions for {DEFAULT_N_CLUSTERS} clusters: {proportions}")
    
    X, y, centers = generate_data(DEFAULT_N_CLUSTERS)
    print(f"Generated data shape: {X.shape}")
    print(f"Number of unique labels: {len(np.unique(y))}")
    print(f"Centers shape: {centers.shape}") 