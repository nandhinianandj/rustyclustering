//! Gaussian Mixture Model (GMM) implementation.
//!
//! Currently supports diagonal covariance matrices for efficiency and to avoid
//! heavy linear algebra dependencies (BLAS/LAPACK).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use rand::Rng;
use rand::seq::SliceRandom;
use std::f64::consts::PI;
use crate::kmeans::{KMeans, KMeansParams};

/// Covariance type for GMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CovarianceType {
    /// Diagonal covariance matrix (features are independent).
    Diag,
    // Full covariance requires matrix inversion/determinant (needs ndarray-linalg)
    // Full, 
}

/// Parameters for GMM.
#[derive(Debug, Clone)]
pub struct GMMParams {
    /// Number of components.
    pub n_components: usize,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Tolerance for convergence.
    pub tol: f64,
    /// Covariance type.
    pub covariance_type: CovarianceType,
    /// Random seed.
    pub seed: Option<u64>,
}

impl GMMParams {
    /// Create new GMM parameters.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_iter: 100,
            tol: 1e-3,
            covariance_type: CovarianceType::Diag,
            seed: None,
        }
    }

    /// Set maximum iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Result of GMM clustering.
#[derive(Debug, Clone)]
pub struct GMMResult {
    /// Means of each component (n_components x n_features).
    pub means: Array2<f64>,
    /// Covariances of each component.
    /// For Diag: (n_components x n_features) - diagonal elements.
    pub covariances: Array2<f64>,
    /// Weights of each component (n_components).
    pub weights: Array1<f64>,
    /// Predicted labels for the training data.
    pub labels: Array1<usize>,
    /// Log-likelihood of the model.
    pub log_likelihood: f64,
}

/// Gaussian Mixture Model algorithm.
pub struct GMM {
    params: GMMParams,
}

impl GMM {
    /// Create a new GMM instance.
    pub fn new(params: GMMParams) -> Self {
        Self { params }
    }

    /// Fit GMM to the data using Expectation-Maximization (EM).
    pub fn fit(&self, data: &ArrayView2<f64>) -> GMMResult {
        let n_samples = data.nrows();
        let n_features = data.ncols();
        let k = self.params.n_components;

        // 1. Initialization
        // Use K-Means to initialize means
        let kmeans_params = KMeansParams::new(k)
            .with_max_iter(10)
            .with_seed(self.params.seed.unwrap_or(0)); // Propagate seed if possible
        let kmeans = KMeans::new(kmeans_params);
        let kmeans_result = kmeans.fit(data);

        let mut means = kmeans_result.centroids;
        let mut weights = Array1::from_elem(k, 1.0 / k as f64);
        
        // Initialize covariances (diagonal) based on cluster variance
        let mut covariances = Array2::zeros((k, n_features));
        for i in 0..k {
            // Find points in this cluster
            let cluster_points: Vec<_> = data.outer_iter()
                .zip(kmeans_result.labels.iter())
                .filter(|(_, &l)| l == i)
                .map(|(p, _)| p)
                .collect();
            
            if cluster_points.is_empty() {
                // Fallback for empty cluster: use global variance
                let global_var = data.var_axis(Axis(0), 0.0);
                covariances.row_mut(i).assign(&global_var);
            } else {
                // Compute variance for this cluster
                // Manual variance computation for robustness
                let mean = means.row(i);
                let mut var = Array1::zeros(n_features);
                for p in &cluster_points {
                    for j in 0..n_features {
                        var[j] += (p[j] - mean[j]).powi(2);
                    }
                }
                var /= cluster_points.len() as f64;
                // Add small epsilon to avoid zero variance
                var += 1e-6;
                covariances.row_mut(i).assign(&var);
            }
        }

        let mut log_likelihood = -f64::INFINITY;
        let mut responsibilities = Array2::zeros((n_samples, k));

        for _iter in 0..self.params.max_iter {
            // 2. E-Step: Compute responsibilities
            let (new_resp, new_ll) = self.e_step(data, &means, &covariances, &weights);
            responsibilities = new_resp;

            // Check convergence
            if (new_ll - log_likelihood).abs() < self.params.tol {
                log_likelihood = new_ll;
                break;
            }
            log_likelihood = new_ll;

            // 3. M-Step: Update parameters
            let (new_means, new_covs, new_weights) = self.m_step(data, &responsibilities);
            means = new_means;
            covariances = new_covs;
            weights = new_weights;
        }

        // Compute final labels
        let labels = responsibilities.map_axis(Axis(1), |row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        });

        GMMResult {
            means,
            covariances,
            weights,
            labels,
            log_likelihood,
        }
    }

    fn e_step(
        &self,
        data: &ArrayView2<f64>,
        means: &Array2<f64>,
        covariances: &Array2<f64>,
        weights: &Array1<f64>,
    ) -> (Array2<f64>, f64) {
        let n_samples = data.nrows();
        let k = self.params.n_components;
        let mut responsibilities = Array2::zeros((n_samples, k));
        let mut log_likelihood = 0.0;

        for i in 0..n_samples {
            let x = data.row(i);
            let mut row_sum = 0.0;
            
            for j in 0..k {
                let mean = means.row(j);
                let cov = covariances.row(j);
                let weight = weights[j];
                
                let prob = self.gaussian_pdf_diag(&x, &mean, &cov);
                let weighted_prob = weight * prob;
                
                responsibilities[[i, j]] = weighted_prob;
                row_sum += weighted_prob;
            }

            // Normalize responsibilities
            if row_sum > 0.0 {
                let mut row = responsibilities.row_mut(i);
                row /= row_sum;
                log_likelihood += row_sum.ln();
            }
        }

        (responsibilities, log_likelihood)
    }

    fn m_step(
        &self,
        data: &ArrayView2<f64>,
        responsibilities: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>, Array1<f64>) {
        let n_samples = data.nrows();
        let n_features = data.ncols();
        let k = self.params.n_components;

        let nk = responsibilities.sum_axis(Axis(0)); // Sum of responsibilities for each component
        
        // Update weights
        let weights = &nk / n_samples as f64;

        // Update means
        let mut means = Array2::zeros((k, n_features));
        for j in 0..k {
            if nk[j] > 1e-10 {
                for i in 0..n_samples {
                    let resp = responsibilities[[i, j]];
                    let x = data.row(i);
                    let mut mean_row = means.row_mut(j);
                    // mean_row += x * resp
                    for d in 0..n_features {
                        mean_row[d] += x[d] * resp;
                    }
                }
                let mut mean_row = means.row_mut(j);
                mean_row /= nk[j];
            }
        }

        // Update covariances (Diagonal)
        let mut covariances = Array2::zeros((k, n_features));
        for j in 0..k {
            if nk[j] > 1e-10 {
                for i in 0..n_samples {
                    let resp = responsibilities[[i, j]];
                    let x = data.row(i);
                    let mean = means.row(j);
                    let mut cov_row = covariances.row_mut(j);
                    
                    for d in 0..n_features {
                        let diff = x[d] - mean[d];
                        cov_row[d] += resp * diff * diff;
                    }
                }
                let mut cov_row = covariances.row_mut(j);
                cov_row /= nk[j];
                // Add epsilon for numerical stability
                cov_row += 1e-6;
            } else {
                // Reset covariance if component died
                 covariances.row_mut(j).fill(1.0);
            }
        }

        (means, covariances, weights)
    }

    fn gaussian_pdf_diag(&self, x: &ArrayView1<f64>, mean: &ArrayView1<f64>, cov: &ArrayView1<f64>) -> f64 {
        let n_features = x.len();
        let mut exponent = 0.0;
        let mut det_sqrt = 1.0;

        for i in 0..n_features {
            let diff = x[i] - mean[i];
            exponent += (diff * diff) / cov[i];
            det_sqrt *= cov[i].sqrt();
        }
        
        let norm_const = (2.0 * PI).powf(n_features as f64 / 2.0) * det_sqrt;
        (-0.5 * exponent).exp() / norm_const
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gmm_simple() {
        // Two clusters: one around (0,0), one around (10,10)
        let data = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],
            [11.0, 10.0],
            [10.0, 11.0],
        ];

        let params = GMMParams::new(2).with_seed(42);
        let gmm = GMM::new(params);
        let result = gmm.fit(&data.view());

        assert_eq!(result.means.nrows(), 2);
        
        // Check labels
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[0], result.labels[2]);
        
        assert_eq!(result.labels[3], result.labels[4]);
        assert_eq!(result.labels[3], result.labels[5]);
        
        assert_ne!(result.labels[0], result.labels[3]);
    }
}
