//! DBSCAN (Density-Based Spatial Clustering of Applications with Noise) implementation.

use ndarray::{ArrayView2, Array1};
use crate::distance::DistanceMetric;
use crate::spatial_index::SpatialIndex;
use rayon::prelude::*;

/// Parameters for DBSCAN clustering.
#[derive(Debug, Clone)]
pub struct DBSCANParams {
    /// Maximum distance between two points to be considered neighbors (epsilon).
    pub eps: f64,
    /// Minimum number of points required to form a dense region.
    pub min_samples: usize,
    /// Distance metric to use.
    pub metric: DistanceMetric,
}

impl DBSCANParams {
    /// Create new DBSCAN parameters.
    ///
    /// # Arguments
    ///
    /// * `eps` - Maximum distance between neighbors
    /// * `min_samples` - Minimum points to form a cluster
    pub fn new(eps: f64, min_samples: usize) -> Self {
        Self {
            eps,
            min_samples,
            metric: DistanceMetric::Euclidean,
        }
    }

    /// Set the distance metric.
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }
}

/// Result of DBSCAN clustering.
#[derive(Debug, Clone)]
pub struct ClusterResult {
    /// Cluster labels for each point (-1 for noise).
    pub labels: Array1<i32>,
    /// Indicates whether each point is a core point.
    pub core_points: Array1<bool>,
}

impl ClusterResult {
    /// Get the number of clusters (excluding noise).
    pub fn n_clusters(&self) -> usize {
        let max_label = self.labels.iter().max().copied().unwrap_or(-1);
        if max_label < 0 {
            0
        } else {
            (max_label + 1) as usize
        }
    }

    /// Get the number of noise points.
    pub fn n_noise(&self) -> usize {
        self.labels.iter().filter(|&&label| label == -1).count()
    }
}

/// DBSCAN clustering algorithm.
pub struct DBSCAN {
    params: DBSCANParams,
}

impl DBSCAN {
    /// Create a new DBSCAN instance with the given parameters.
    pub fn new(params: DBSCANParams) -> Self {
        Self { params }
    }

    /// Fit the DBSCAN algorithm to the data.
    ///
    /// # Arguments
    ///
    /// * `data` - 2D array where each row is a data point
    ///
    /// # Returns
    ///
    /// ClusterResult containing cluster labels and core point indicators
    pub fn fit(&self, data: &ArrayView2<f64>) -> ClusterResult {
        let n_points = data.nrows();
        
        // Initialize labels (-2 = unvisited, -1 = noise, >= 0 = cluster id)
        let mut labels = Array1::from_elem(n_points, -2);
        let mut core_points = Array1::from_elem(n_points, false);
        
        // Build spatial index for efficient neighbor queries
        let index = SpatialIndex::new(data);
        
        // Find neighbors for all points
        let neighbors: Vec<Vec<usize>> = (0..n_points)
            .into_par_iter()
            .map(|i| {
                let point = data.row(i);
                index.radius_search(&point, self.params.eps)
                    .into_iter()
                    .map(|(_, idx)| idx)
                    .collect()
            })
            .collect();
        
        // Identify core points
        for (i, neighbor_list) in neighbors.iter().enumerate() {
            if neighbor_list.len() >= self.params.min_samples {
                core_points[i] = true;
            }
        }
        
        // Cluster expansion
        let mut cluster_id = 0;
        for i in 0..n_points {
            if labels[i] != -2 {
                continue; // Already processed
            }
            
            if !core_points[i] {
                labels[i] = -1; // Mark as noise
                continue;
            }
            
            // Start a new cluster
            self.expand_cluster(i, cluster_id, &neighbors, &mut labels, &core_points);
            cluster_id += 1;
        }
        
        ClusterResult { labels, core_points }
    }

    /// Expand a cluster from a core point using BFS.
    fn expand_cluster(
        &self,
        start_point: usize,
        cluster_id: i32,
        neighbors: &[Vec<usize>],
        labels: &mut Array1<i32>,
        core_points: &Array1<bool>,
    ) {
        let mut queue = vec![start_point];
        labels[start_point] = cluster_id;
        
        while let Some(point) = queue.pop() {
            if !core_points[point] {
                continue;
            }
            
            for &neighbor in &neighbors[point] {
                if labels[neighbor] == -2 {
                    // Unvisited point
                    labels[neighbor] = cluster_id;
                    queue.push(neighbor);
                } else if labels[neighbor] == -1 {
                    // Noise point that can be absorbed into cluster
                    labels[neighbor] = cluster_id;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_dbscan_simple_clusters() {
        // Two clear clusters
        let data = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],
            [11.0, 10.0],
            [10.0, 11.0],
        ];
        
        let params = DBSCANParams::new(2.0, 2);
        let dbscan = DBSCAN::new(params);
        let result = dbscan.fit(&data.view());
        
        assert_eq!(result.n_clusters(), 2);
        assert_eq!(result.n_noise(), 0);
    }

    #[test]
    fn test_dbscan_with_noise() {
        let data = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [100.0, 100.0], // Noise point
        ];
        
        let params = DBSCANParams::new(2.0, 2);
        let dbscan = DBSCAN::new(params);
        let result = dbscan.fit(&data.view());
        
        assert_eq!(result.n_clusters(), 1);
        assert_eq!(result.n_noise(), 1);
        assert_eq!(result.labels[3], -1); // Last point is noise
    }

    #[test]
    fn test_dbscan_all_noise() {
        let data = array![
            [0.0, 0.0],
            [10.0, 10.0],
            [20.0, 20.0],
        ];
        
        let params = DBSCANParams::new(2.0, 2);
        let dbscan = DBSCAN::new(params);
        let result = dbscan.fit(&data.view());
        
        assert_eq!(result.n_clusters(), 0);
        assert_eq!(result.n_noise(), 3);
    }
}
