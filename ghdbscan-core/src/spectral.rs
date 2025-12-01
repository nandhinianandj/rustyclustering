//! Spectral Clustering implementation.
//!
//! Uses nalgebra for eigenvalue decomposition.

use ndarray::{Array1, Array2, ArrayView2, Axis};
use crate::kmeans::{KMeans, KMeansParams};
use crate::distance::{DistanceMetric, Distance};
use nalgebra::{DMatrix, SymmetricEigen};

/// Parameters for Spectral Clustering.
#[derive(Debug, Clone)]
pub struct SpectralParams {
    /// Number of clusters.
    pub n_clusters: usize,
    /// Number of neighbors for affinity matrix (if using KNN).
    pub n_neighbors: usize,
    /// Gamma for RBF kernel (if using RBF).
    pub gamma: f64,
    /// Random seed.
    pub seed: Option<u64>,
}

impl SpectralParams {
    /// Create new Spectral Clustering parameters.
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            n_neighbors: 10,
            gamma: 1.0,
            seed: None,
        }
    }

    /// Set number of neighbors.
    pub fn with_n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set gamma.
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Result of Spectral Clustering.
#[derive(Debug, Clone)]
pub struct SpectralResult {
    /// Cluster labels for each point.
    pub labels: Array1<usize>,
}

/// Spectral Clustering algorithm.
pub struct SpectralClustering {
    params: SpectralParams,
}

impl SpectralClustering {
    /// Create a new SpectralClustering instance.
    pub fn new(params: SpectralParams) -> Self {
        Self { params }
    }

    /// Fit Spectral Clustering to the data.
    pub fn fit(&self, data: &ArrayView2<f64>) -> SpectralResult {
        let n_samples = data.nrows();
        
        // 1. Construct Affinity Matrix (RBF Kernel)
        // For simplicity, we use full RBF kernel here. 
        // For large N, KNN graph is better but requires sparse matrix support in eigendecomposition.
        // nalgebra's SymmetricEigen works on dense matrices.
        let mut affinity = DMatrix::zeros(n_samples, n_samples);
        let gamma = self.params.gamma;

        for i in 0..n_samples {
            for j in 0..n_samples {
                if i == j {
                    affinity[(i, j)] = 1.0;
                } else if i < j {
                    let d = DistanceMetric::Euclidean.distance(&data.row(i), &data.row(j));
                    let sim = (-gamma * d * d).exp();
                    affinity[(i, j)] = sim;
                    affinity[(j, i)] = sim;
                }
            }
        }

        // 2. Compute Laplacian
        // Unnormalized Laplacian: L = D - A
        // Normalized Laplacian (Random Walk): L_rw = I - D^-1 * A
        // Normalized Laplacian (Symmetric): L_sym = I - D^-1/2 * A * D^-1/2
        // We use L_sym as it's symmetric and works well with SymmetricEigen
        
        // Compute Degree Matrix D
        let mut d_inv_sqrt = DMatrix::zeros(n_samples, n_samples);
        for i in 0..n_samples {
            let mut degree = 0.0;
            for j in 0..n_samples {
                degree += affinity[(i, j)];
            }
            if degree > 1e-10 {
                d_inv_sqrt[(i, i)] = 1.0 / degree.sqrt();
            }
        }

        // L_sym = I - D^-1/2 * A * D^-1/2
        // Actually, we want eigenvectors of L corresponding to smallest eigenvalues.
        // Equivalently, eigenvectors of D^-1/2 * A * D^-1/2 corresponding to LARGEST eigenvalues.
        // Let M = D^-1/2 * A * D^-1/2
        let m = &d_inv_sqrt * &affinity * &d_inv_sqrt;

        // 3. Eigenvalue Decomposition
        let eigen = SymmetricEigen::new(m);
        let eigenvalues = eigen.eigenvalues;
        let eigenvectors = eigen.eigenvectors;

        // Sort eigenvalues and get indices of k largest
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());
        
        let k = self.params.n_clusters;
        let top_indices = &indices[0..k];

        // Form matrix U from top k eigenvectors
        let mut u = Array2::zeros((n_samples, k));
        for (col_idx, &eig_idx) in top_indices.iter().enumerate() {
            for row_idx in 0..n_samples {
                u[[row_idx, col_idx]] = eigenvectors[(row_idx, eig_idx)];
            }
        }

        // Normalize rows of U
        for mut row in u.outer_iter_mut() {
            let norm = row.mapv(|x| x * x).sum().sqrt();
            if norm > 1e-10 {
                row /= norm;
            }
        }

        // 4. K-Means on U
        let kmeans_params = KMeansParams::new(k)
            .with_seed(self.params.seed.unwrap_or(0));
        let kmeans = KMeans::new(kmeans_params);
        let result = kmeans.fit(&u.view());

        SpectralResult {
            labels: result.labels,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_spectral_simple() {
        // Two concentric circles (or just separated groups)
        // Spectral is good at non-convex clusters, but here we test basic separation
        let data = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
        ];

        let params = SpectralParams::new(2).with_seed(42).with_gamma(1.0);
        let spectral = SpectralClustering::new(params);
        let result = spectral.fit(&data.view());

        // Check labels
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[0], result.labels[2]);
        
        assert_eq!(result.labels[3], result.labels[4]);
        assert_eq!(result.labels[3], result.labels[5]);
        
        assert_ne!(result.labels[0], result.labels[3]);
    }
}
