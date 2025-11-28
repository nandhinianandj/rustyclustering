"""
HDBSCAN clustering example with visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from ghdbscan import HDBSCAN

def make_varied_density_blobs(random_state=42):
    """Generate clusters with varying densities."""
    np.random.seed(random_state)
    
    # Dense cluster
    dense = np.random.randn(50, 2) * 0.5 + np.array([0, 0])
    
    # Medium density cluster
    medium = np.random.randn(40, 2) * 1.5 + np.array([10, 0])
    
    # Sparse cluster
    sparse = np.random.randn(30, 2) * 2.5 + np.array([0, 10])
    
    # Noise
    noise = np.random.randn(20, 2) * 10
    
    X = np.vstack([dense, medium, sparse, noise])
    return X

def main():
    print("HDBSCAN Clustering Example with Visualization")
    print("=" * 50)
    
    # Generate sample data with varying densities
    X = make_varied_density_blobs(random_state=42)
    print(f"\nGenerated {len(X)} data points with varying density clusters")
    
    # Run HDBSCAN
    min_cluster_size = 10
    min_samples = 5
    print(f"\nRunning HDBSCAN with min_cluster_size={min_cluster_size}, min_samples={min_samples}")
    
    hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean")
    labels = hdbscan.fit_predict(X)
    
    # Print results
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"\nResults:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Number of noise points: {n_noise}")
    
    # Visualize results
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = 'gray'
        
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        
        if k == -1:
            ax.scatter(xy[:, 0], xy[:, 1], c=[col], s=50, alpha=0.3, 
                      edgecolors='k', marker='x', label='Noise')
        else:
            ax.scatter(xy[:, 0], xy[:, 1], c=[col], s=50, alpha=0.6, 
                      edgecolors='k', label=f'Cluster {k}')
    
    ax.set_title(f'HDBSCAN Results\n(min_cluster_size={min_cluster_size}, min_samples={min_samples})')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hdbscan_results.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to 'hdbscan_results.png'")
    plt.show()

if __name__ == "__main__":
    main()
