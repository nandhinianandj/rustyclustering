//! Distance metrics for clustering algorithms.

use ndarray::ArrayView1;

/// Distance metric types supported by the clustering algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Euclidean distance (L2 norm)
    Euclidean,
    /// Manhattan distance (L1 norm)
    Manhattan,
    /// Cosine distance (1 - cosine similarity)
    Cosine,
}

/// Trait for computing distances between points.
pub trait Distance {
    /// Compute the distance between two points.
    fn distance(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64;
}

impl Distance for DistanceMetric {
    fn distance(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        match self {
            DistanceMetric::Euclidean => euclidean_distance(a, b),
            DistanceMetric::Manhattan => manhattan_distance(a, b),
            DistanceMetric::Cosine => cosine_distance(a, b),
        }
    }
}

/// Compute Euclidean distance between two points.
#[inline]
pub fn euclidean_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Compute Manhattan distance between two points.
#[inline]
pub fn manhattan_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum()
}

/// Compute cosine distance between two points.
#[inline]
pub fn cosine_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0; // Maximum distance for zero vectors
    }
    
    1.0 - (dot_product / (norm_a * norm_b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_relative_eq;

    #[test]
    fn test_euclidean_distance() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        assert_relative_eq!(euclidean_distance(&a.view(), &b.view()), 5.0);
    }

    #[test]
    fn test_manhattan_distance() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        assert_relative_eq!(manhattan_distance(&a.view(), &b.view()), 7.0);
    }

    #[test]
    fn test_cosine_distance() {
        let a = array![1.0, 0.0];
        let b = array![1.0, 0.0];
        assert_relative_eq!(cosine_distance(&a.view(), &b.view()), 0.0, epsilon = 1e-10);
        
        let a = array![1.0, 0.0];
        let b = array![0.0, 1.0];
        assert_relative_eq!(cosine_distance(&a.view(), &b.view()), 1.0, epsilon = 1e-10);
    }
}
