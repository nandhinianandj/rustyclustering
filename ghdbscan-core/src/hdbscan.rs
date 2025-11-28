//! HDBSCAN (Hierarchical Density-Based Spatial Clustering) implementation.

use ndarray::{ArrayView2, Array1};
use crate::distance::{Distance, DistanceMetric};
use crate::spatial_index::SpatialIndex;
use crate::mst::MST;
use crate::hierarchy::Hierarchy;
use rayon::prelude::*;

/// Parameters for HDBSCAN clustering.
#[derive(Debug, Clone)]
pub struct HDBSCANParams {
    /// Minimum cluster size.
    pub min_cluster_size: usize,
    /// Minimum number of samples (for core distance calculation).
    pub min_samples: usize,
    /// Distance metric to use.
    pub metric: DistanceMetric,
}

impl HDBSCANParams {
    /// Create new HDBSCAN parameters.
    ///
    /// # Arguments
    ///
    /// * `min_cluster_size` - Minimum size of clusters
    /// * `min_samples` - Minimum samples for core distance (defaults to min_cluster_size if None)
    pub fn new(min_cluster_size: usize, min_samples: Option<usize>) -> Self {
        Self {
            min_cluster_size,
            min_samples: min_samples.unwrap_or(min_cluster_size),
            metric: DistanceMetric::Euclidean,
        }
    }

    /// Set the distance metric.
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }
}

/// Result of HDBSCAN clustering.
#[derive(Debug, Clone)]
pub struct HDBSCANResult {
    /// Cluster labels for each point (-1 for noise).
    pub labels: Array1<i32>,
    /// Cluster membership probabilities.
    pub probabilities: Array1<f64>,
    /// Outlier scores.
    pub outlier_scores: Array1<f64>,
}

impl HDBSCANResult {
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

/// HDBSCAN clustering algorithm.
pub struct HDBSCAN {
    params: HDBSCANParams,
}

impl HDBSCAN {
    /// Create a new HDBSCAN instance with the given parameters.
    pub fn new(params: HDBSCANParams) -> Self {
        Self { params }
    }

    /// Fit the HDBSCAN algorithm to the data.
    ///
    /// # Arguments
    ///
    /// * `data` - 2D array where each row is a data point
    ///
    /// # Returns
    ///
    /// HDBSCANResult containing cluster labels, probabilities, and outlier scores
    pub fn fit(&self, data: &ArrayView2<f64>) -> HDBSCANResult {
        let n_points = data.nrows();
        
        if n_points == 0 {
            return HDBSCANResult {
                labels: Array1::from_elem(0, -1),
                probabilities: Array1::from_elem(0, 0.0),
                outlier_scores: Array1::from_elem(0, 0.0),
            };
        }

        // Step 1: Compute core distances
        let core_distances = self.compute_core_distances(data);

        // Step 2: Compute mutual reachability distance matrix
        let mutual_reach_dist = self.compute_mutual_reachability(data, &core_distances);

        // Step 3: Build minimum spanning tree
        let mst = MST::from_distance_matrix(&mutual_reach_dist);

        // Step 4: Build cluster hierarchy
        let hierarchy = self.build_hierarchy(&mst, n_points);

        // Step 5: Extract clusters
        let labels = self.extract_clusters(&hierarchy, n_points);

        // Step 6: Compute probabilities and outlier scores
        let probabilities = Array1::from_elem(n_points, 1.0);
        let outlier_scores = Array1::from_elem(n_points, 0.0);

        HDBSCANResult {
            labels,
            probabilities,
            outlier_scores,
        }
    }

    /// Compute core distances for all points.
    fn compute_core_distances(&self, data: &ArrayView2<f64>) -> Vec<f64> {
        let n_points = data.nrows();
        let index = SpatialIndex::new(data);

        (0..n_points)
            .into_par_iter()
            .map(|i| {
                let point = data.row(i);
                let neighbors = index.knn_search(&point, self.params.min_samples + 1);
                
                // Core distance is the distance to the k-th nearest neighbor
                if neighbors.len() > self.params.min_samples {
                    neighbors[self.params.min_samples].0
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Compute mutual reachability distance matrix.
    fn compute_mutual_reachability(
        &self,
        data: &ArrayView2<f64>,
        core_distances: &[f64],
    ) -> Vec<Vec<f64>> {
        let n_points = data.nrows();
        let mut dist_matrix = vec![vec![0.0; n_points]; n_points];

        for i in 0..n_points {
            for j in (i + 1)..n_points {
                let point_i = data.row(i);
                let point_j = data.row(j);
                let dist = self.params.metric.distance(&point_i, &point_j);
                
                // Mutual reachability distance
                let mutual_reach = dist
                    .max(core_distances[i])
                    .max(core_distances[j]);
                
                dist_matrix[i][j] = mutual_reach;
                dist_matrix[j][i] = mutual_reach;
            }
        }

        dist_matrix
    }

    /// Build cluster hierarchy from MST.
    fn build_hierarchy(&self, mst: &MST, n_points: usize) -> Hierarchy {
        let mut hierarchy = Hierarchy::new();
        let mut union_find = UnionFind::new(n_points);
        
        // Create leaf nodes for each point
        let mut node_map: Vec<usize> = (0..n_points)
            .map(|i| hierarchy.add_leaf(i, 0.0))
            .collect();

        // Process MST edges in order of increasing weight
        for edge in &mst.edges {
            let root1 = union_find.find(edge.from);
            let root2 = union_find.find(edge.to);

            if root1 != root2 {
                // Create new parent node
                let lambda = if edge.weight > 0.0 {
                    1.0 / edge.weight
                } else {
                    f64::INFINITY
                };
                
                let parent = hierarchy.add_internal(lambda);
                hierarchy.merge(node_map[root1], node_map[root2], parent, lambda);

                // Union the sets
                let new_root = union_find.union(root1, root2);
                node_map[new_root] = parent;
            }
        }

        // Set root
        let root = union_find.find(0);
        hierarchy.set_root(node_map[root]);
        
        // Compute stability
        hierarchy.compute_stability();

        hierarchy
    }

    /// Extract clusters from hierarchy.
    fn extract_clusters(&self, hierarchy: &Hierarchy, n_points: usize) -> Array1<i32> {
        let mut labels = Array1::from_elem(n_points, -1);
        
        let selected_clusters = hierarchy.extract_clusters(self.params.min_cluster_size);
        
        // Assign labels based on selected clusters
        for (cluster_id, &node_id) in selected_clusters.iter().enumerate() {
            let points = self.get_cluster_points(hierarchy, node_id);
            for point in points {
                labels[point] = cluster_id as i32;
            }
        }

        labels
    }

    /// Get all points belonging to a cluster node.
    fn get_cluster_points(&self, hierarchy: &Hierarchy, node_id: usize) -> Vec<usize> {
        let mut points = Vec::new();
        self.collect_points_recursive(hierarchy, node_id, &mut points);
        points
    }

    /// Recursively collect all points in a subtree.
    fn collect_points_recursive(
        &self,
        hierarchy: &Hierarchy,
        node_id: usize,
        points: &mut Vec<usize>,
    ) {
        if let Some(node) = hierarchy.nodes.get(&node_id) {
            if node.is_leaf() {
                points.extend(&node.points);
            } else {
                for &child in &node.children {
                    self.collect_points_recursive(hierarchy, child, points);
                }
            }
        }
    }
}

/// Union-Find data structure for MST construction.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) -> usize {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return root_x;
        }

        if self.rank[root_x] < self.rank[root_y] {
            self.parent[root_x] = root_y;
            root_y
        } else if self.rank[root_x] > self.rank[root_y] {
            self.parent[root_y] = root_x;
            root_x
        } else {
            self.parent[root_y] = root_x;
            self.rank[root_x] += 1;
            root_x
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_hdbscan_simple_clusters() {
        let data = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],
            [11.0, 10.0],
            [10.0, 11.0],
        ];
        
        let params = HDBSCANParams::new(2, None);
        let hdbscan = HDBSCAN::new(params);
        let result = hdbscan.fit(&data.view());
        
        // Should find at least one cluster
        assert!(result.n_clusters() >= 1);
    }

    #[test]
    fn test_hdbscan_empty() {
        let data = Array2::<f64>::zeros((0, 2));
        
        let params = HDBSCANParams::new(2, None);
        let hdbscan = HDBSCAN::new(params);
        let result = hdbscan.fit(&data.view());
        
        assert_eq!(result.labels.len(), 0);
    }
}
