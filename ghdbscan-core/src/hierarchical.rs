//! Agglomerative Hierarchical Clustering implementation.

use ndarray::{Array1, Array2, ArrayView2};
use crate::distance::{DistanceMetric, Distance};
use ordered_float::OrderedFloat;
use std::collections::HashMap;

/// Linkage criterion for hierarchical clustering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linkage {
    /// Minimum distance between points in two clusters.
    Single,
    /// Maximum distance between points in two clusters.
    Complete,
    /// Average distance between points in two clusters.
    Average,
    /// Ward's minimum variance method.
    Ward,
}

/// Parameters for Hierarchical Clustering.
#[derive(Debug, Clone)]
pub struct HierarchicalParams {
    /// Number of clusters to find.
    pub n_clusters: usize,
    /// Linkage criterion.
    pub linkage: Linkage,
    /// Distance metric.
    pub metric: DistanceMetric,
}

impl HierarchicalParams {
    /// Create new Hierarchical Clustering parameters.
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            linkage: Linkage::Ward,
            metric: DistanceMetric::Euclidean,
        }
    }

    /// Set linkage criterion.
    pub fn with_linkage(mut self, linkage: Linkage) -> Self {
        self.linkage = linkage;
        self
    }

    /// Set distance metric.
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }
}

/// Result of Hierarchical Clustering.
#[derive(Debug, Clone)]
pub struct HierarchicalResult {
    /// Cluster labels for each point.
    pub labels: Array1<usize>,
    /// The children of each non-leaf node.
    /// Values less than n_samples correspond to leaves of the tree.
    /// A node i greater than or equal to n_samples is a non-leaf node and has children children[i - n_samples].
    pub children: Vec<[usize; 2]>,
}

/// Agglomerative Hierarchical Clustering algorithm.
pub struct AgglomerativeClustering {
    params: HierarchicalParams,
}

impl AgglomerativeClustering {
    /// Create a new AgglomerativeClustering instance.
    pub fn new(params: HierarchicalParams) -> Self {
        Self { params }
    }

    /// Fit the algorithm to the data.
    ///
    /// Note: This is a naive O(N^3) implementation for demonstration.
    /// Optimized versions use priority queues and spatial indexes (O(N^2) or O(N^2 log N)).
    pub fn fit(&self, data: &ArrayView2<f64>) -> HierarchicalResult {
        let n_samples = data.nrows();
        if n_samples < self.params.n_clusters {
            panic!("Number of samples must be >= n_clusters");
        }

        // Initialize each point as a cluster
        let mut clusters: Vec<Vec<usize>> = (0..n_samples).map(|i| vec![i]).collect();
        // Map current cluster index to original cluster ID (which becomes node ID in tree)
        let mut cluster_ids: Vec<usize> = (0..n_samples).collect();
        let mut active_clusters: Vec<bool> = vec![true; n_samples];
        let mut children = Vec::with_capacity(n_samples - 1);
        
        let mut next_cluster_id = n_samples;
        let mut n_active_clusters = n_samples;

        // Precompute distance matrix (upper triangle)
        // For Ward, we need squared Euclidean distances
        let mut dist_matrix = Array2::<f64>::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let d = self.params.metric.distance(&data.row(i), &data.row(j));
                dist_matrix[[i, j]] = d;
                dist_matrix[[j, i]] = d;
            }
        }

        // Main loop: merge clusters until we reach n_clusters
        while n_active_clusters > self.params.n_clusters {
            let mut min_dist = f64::MAX;
            let mut merge_pair = (0, 0);

            // Find closest pair of active clusters
            // Naive search: O(N^2) per iteration -> O(N^3) total
            for i in 0..active_clusters.len() {
                if !active_clusters[i] { continue; }
                for j in (i + 1)..active_clusters.len() {
                    if !active_clusters[j] { continue; }

                    let dist = self.compute_linkage_distance(
                        &clusters[i], &clusters[j], &dist_matrix, data
                    );

                    if dist < min_dist {
                        min_dist = dist;
                        merge_pair = (i, j);
                    }
                }
            }

            let (c1, c2) = merge_pair;
            
            // Record merge
            children.push([cluster_ids[c1], cluster_ids[c2]]);
            
            // Merge c2 into c1
            let mut new_cluster = clusters[c1].clone();
            new_cluster.extend_from_slice(&clusters[c2]);
            clusters[c1] = new_cluster;
            
            // Update cluster ID
            cluster_ids[c1] = next_cluster_id;
            next_cluster_id += 1;
            
            // Deactivate c2
            active_clusters[c2] = false;
            n_active_clusters -= 1;
        }

        // Assign labels based on remaining active clusters
        let mut labels = Array1::zeros(n_samples);
        let mut label_id = 0;
        for i in 0..active_clusters.len() {
            if active_clusters[i] {
                for &sample_idx in &clusters[i] {
                    labels[sample_idx] = label_id;
                }
                label_id += 1;
            }
        }

        HierarchicalResult {
            labels,
            children,
        }
    }

    fn compute_linkage_distance(
        &self,
        c1: &[usize],
        c2: &[usize],
        dist_matrix: &Array2<f64>,
        data: &ArrayView2<f64>, // Needed for Ward
    ) -> f64 {
        match self.params.linkage {
            Linkage::Single => {
                let mut min_d = f64::MAX;
                for &i in c1 {
                    for &j in c2 {
                        let d = dist_matrix[[i, j]];
                        if d < min_d { min_d = d; }
                    }
                }
                min_d
            },
            Linkage::Complete => {
                let mut max_d = f64::MIN;
                for &i in c1 {
                    for &j in c2 {
                        let d = dist_matrix[[i, j]];
                        if d > max_d { max_d = d; }
                    }
                }
                max_d
            },
            Linkage::Average => {
                let mut sum_d = 0.0;
                for &i in c1 {
                    for &j in c2 {
                        sum_d += dist_matrix[[i, j]];
                    }
                }
                sum_d / (c1.len() * c2.len()) as f64
            },
            Linkage::Ward => {
                // Ward's method minimizes the increase in variance
                // Delta = ((n1 * n2) / (n1 + n2)) * ||centroid1 - centroid2||^2
                let n1 = c1.len() as f64;
                let n2 = c2.len() as f64;
                
                let mut centroid1 = Array1::<f64>::zeros(data.ncols());
                for &i in c1 { centroid1 += &data.row(i); }
                centroid1 /= n1;
                
                let mut centroid2 = Array1::<f64>::zeros(data.ncols());
                for &i in c2 { centroid2 += &data.row(i); }
                centroid2 /= n2;
                
                let dist_sq = self.params.metric.distance(&centroid1.view(), &centroid2.view()).powi(2);
                
                ((n1 * n2) / (n1 + n2)) * dist_sq
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_hierarchical_simple() {
        let data = array![
            [0.0, 0.0],
            [1.0, 0.0], // Cluster 1
            [10.0, 10.0],
            [11.0, 10.0], // Cluster 2
        ];
        
        let params = HierarchicalParams::new(2).with_linkage(Linkage::Single);
        let model = AgglomerativeClustering::new(params);
        let result = model.fit(&data.view());
        
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[2], result.labels[3]);
        assert_ne!(result.labels[0], result.labels[2]);
    }

    #[test]
    fn test_hierarchical_ward() {
        let data = array![
            [0.0, 0.0],
            [0.5, 0.5],
            [10.0, 10.0],
            [10.5, 10.5],
        ];
        
        let params = HierarchicalParams::new(2).with_linkage(Linkage::Ward);
        let model = AgglomerativeClustering::new(params);
        let result = model.fit(&data.view());
        
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[2], result.labels[3]);
    }
}
