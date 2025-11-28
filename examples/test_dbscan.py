"""
DBSCAN clustering example with visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from ghdbscan import DBSCAN

def make_blobs(n_samples=100, centers=3, random_state=42):
    """Generate isotropic Gaussian blobs for clustering."""
    np.random.seed(random_state)
    
    # Generate cluster centers
    cluster_centers = np.random.randn(centers, 2) * 10
    
    # Generate points around each center
    points = []
    labels = []
    for i, center in enumerate(cluster_centers):
        cluster_points = np.random.randn(n_samples // centers, 2) + center
        points.append(cluster_points)
        labels.extend([i] * (n_samples // centers))
    
    # Add some noise points
    noise_points = np.random.randn(10, 2) * 20
    points.append(noise_points)
    labels.extend([-1] * 10)
    
    X = np.vstack(points)
    return X, np.array(labels)

def main():
    print("DBSCAN Clustering Example with Visualization")
    print("=" * 50)
    
    # Generate sample data
    X, true_labels = make_blobs(n_samples=150, centers=3, random_state=42)
    print(f"\nGenerated {len(X)} data points with 3 clusters + noise")
    
    # Run DBSCAN
    eps = 3.0
    min_samples = 5
    print(f"\nRunning DBSCAN with eps={eps}, min_samples={min_samples}")
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = dbscan.fit_predict(X)
    
    # Print results
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"\nResults:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Number of noise points: {n_noise}")
    
    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot original data
    unique_labels = set(true_labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = 'gray'
        
        class_member_mask = (true_labels == k)
        xy = X[class_member_mask]
        ax1.scatter(xy[:, 0], xy[:, 1], c=[col], s=50, alpha=0.6, edgecolors='k')
    
    ax1.set_title('Original Data')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    
    # Plot DBSCAN results
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = 'gray'
        
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        
        if k == -1:
            ax2.scatter(xy[:, 0], xy[:, 1], c=[col], s=50, alpha=0.3, 
                       edgecolors='k', marker='x', label='Noise')
        else:
            ax2.scatter(xy[:, 0], xy[:, 1], c=[col], s=50, alpha=0.6, 
                       edgecolors='k', label=f'Cluster {k}')
    
    ax2.set_title(f'DBSCAN Results (eps={eps}, min_samples={min_samples})')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('dbscan_results.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to 'dbscan_results.png'")
    plt.show()

if __name__ == "__main__":
    main()
