//! Spatial indexing for efficient neighbor queries using KD-trees.

use kiddo::KdTree;
use ndarray::{ArrayView2, ArrayView1};

/// Spatial index wrapper for efficient nearest neighbor queries.
pub struct SpatialIndex {
    tree: KdTree<f64, 2>,
    dimensionality: usize,
}

impl SpatialIndex {
    /// Create a new spatial index from data points.
    ///
    /// # Arguments
    ///
    /// * `data` - 2D array where each row is a point
    ///
    /// # Panics
    ///
    /// Panics if the data has more than 16 dimensions (kiddo limitation).
    pub fn new(data: &ArrayView2<f64>) -> Self {
        let (_n_points, n_dims) = data.dim();
        
        if n_dims > 16 {
            panic!("KD-tree supports up to 16 dimensions, got {}", n_dims);
        }

        // For now, we'll handle 2D case. We can extend this with macros for higher dimensions.
        let mut tree = KdTree::new();
        
        for (idx, point) in data.outer_iter().enumerate() {
            if n_dims == 2 {
                tree.add(&[point[0], point[1]], idx as u64);
            } else {
                // For other dimensions, we'll need a different approach
                // This is a limitation we'll document
                panic!("Currently only 2D data is supported. Dimension: {}", n_dims);
            }
        }

        Self {
            tree,
            dimensionality: n_dims,
        }
    }

    /// Find all points within a given radius of a query point.
    ///
    /// # Arguments
    ///
    /// * `point` - Query point
    /// * `radius` - Search radius
    ///
    /// # Returns
    ///
    /// Vector of (distance, point_index) tuples
    pub fn radius_search(&self, point: &ArrayView1<f64>, radius: f64) -> Vec<(f64, usize)> {
        if self.dimensionality == 2 {
            self.tree
                .within::<kiddo::SquaredEuclidean>(&[point[0], point[1]], radius * radius)
                .iter()
                .map(|neighbor| (neighbor.distance.sqrt(), neighbor.item as usize))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Find k nearest neighbors of a query point.
    ///
    /// # Arguments
    ///
    /// * `point` - Query point
    /// * `k` - Number of neighbors to find
    ///
    /// # Returns
    ///
    /// Vector of (distance, point_index) tuples
    pub fn knn_search(&self, point: &ArrayView1<f64>, k: usize) -> Vec<(f64, usize)> {
        if self.dimensionality == 2 {
            self.tree
                .nearest_n::<kiddo::SquaredEuclidean>(&[point[0], point[1]], k)
                .iter()
                .map(|neighbor| (neighbor.distance.sqrt(), neighbor.item as usize))
                .collect()
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_spatial_index_radius_search() {
        let data = array![
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [10.0, 10.0]
        ];
        
        let index = SpatialIndex::new(&data.view());
        let query = array![0.0, 0.0];
        let neighbors = index.radius_search(&query.view(), 2.0);
        
        // Should find points at (0,0) and (1,1)
        assert_eq!(neighbors.len(), 2);
    }

    #[test]
    fn test_spatial_index_knn_search() {
        let data = array![
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [10.0, 10.0]
        ];
        
        let index = SpatialIndex::new(&data.view());
        let query = array![0.0, 0.0];
        let neighbors = index.knn_search(&query.view(), 2);
        
        assert_eq!(neighbors.len(), 2);
        assert_eq!(neighbors[0].1, 0); // Closest is point 0 itself
    }
}

