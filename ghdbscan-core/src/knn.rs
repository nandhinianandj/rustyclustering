//! K-Nearest Neighbors (KNN) implementation.

use ndarray::{Array1, ArrayView1, ArrayView2};
use crate::spatial_index::SpatialIndex;
use crate::distance::DistanceMetric;

/// Parameters for KNN.
#[derive(Debug, Clone)]
pub struct KNNParams {
    /// Number of neighbors to find.
    pub k: usize,
    /// Distance metric to use.
    pub metric: DistanceMetric,
}

impl KNNParams {
    /// Create new KNN parameters.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            metric: DistanceMetric::Euclidean,
        }
    }

    /// Set distance metric.
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }
}

/// K-Nearest Neighbors algorithm.
pub struct KNN {
    params: KNNParams,
    index: Option<SpatialIndex>,
    labels: Option<Array1<i32>>, // Optional labels for classification
    values: Option<Array1<f64>>, // Optional values for regression
}

impl KNN {
    /// Create a new KNN instance.
    pub fn new(params: KNNParams) -> Self {
        Self {
            params,
            index: None,
            labels: None,
            values: None,
        }
    }

    /// Fit the KNN model to the data.
    pub fn fit(&mut self, data: &ArrayView2<f64>) {
        self.index = Some(SpatialIndex::new(data));
    }

    /// Fit with labels for classification.
    pub fn fit_with_labels(&mut self, data: &ArrayView2<f64>, labels: &Array1<i32>) {
        if data.nrows() != labels.len() {
            panic!("Data and labels must have same length");
        }
        self.index = Some(SpatialIndex::new(data));
        self.labels = Some(labels.clone());
    }

    /// Find k-nearest neighbors for a query point.
    /// Returns (indices, distances).
    pub fn query(&self, point: &ArrayView1<f64>) -> (Vec<usize>, Vec<f64>) {
        if let Some(index) = &self.index {
            let neighbors = index.knn_search(point, self.params.k);
            let distances: Vec<f64> = neighbors.iter().map(|(d, _)| *d).collect();
            let indices: Vec<usize> = neighbors.iter().map(|(_, i)| *i).collect();
            (indices, distances)
        } else {
            panic!("KNN model must be fitted before query");
        }
    }

    /// Predict label for a query point (majority vote).
    pub fn predict(&self, point: &ArrayView1<f64>) -> i32 {
        if self.labels.is_none() {
            panic!("KNN model must be fitted with labels for prediction");
        }
        
        let (indices, _) = self.query(point);
        let labels = self.labels.as_ref().unwrap();
        
        // Count votes
        let mut counts = std::collections::HashMap::new();
        for idx in indices {
            *counts.entry(labels[idx]).or_insert(0) += 1;
        }
        
        // Return label with max votes
        counts.into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(label, _)| label)
            .unwrap_or(0) // Default if empty (shouldn't happen)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_knn_query() {
        let data = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],
        ];
        
        let mut knn = KNN::new(KNNParams::new(2));
        knn.fit(&data.view());
        
        let query_point = array![0.1, 0.1];
        let (indices, distances) = knn.query(&query_point.view());
        
        assert_eq!(indices.len(), 2);
        // Closest should be [0.0, 0.0] (index 0)
        assert_eq!(indices[0], 0);
        assert!(distances[0] < 0.2);
    }

    #[test]
    fn test_knn_classification() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.1],
        ];
        let labels = array![0, 0, 1, 1];
        
        let mut knn = KNN::new(KNNParams::new(3));
        knn.fit_with_labels(&data.view(), &labels);
        
        let query_point = array![0.05, 0.05];
        let prediction = knn.predict(&query_point.view());
        
        assert_eq!(prediction, 0);
        
        let query_point_2 = array![10.05, 10.05];
        let prediction_2 = knn.predict(&query_point_2.view());
        
        assert_eq!(prediction_2, 1);
    }
}
