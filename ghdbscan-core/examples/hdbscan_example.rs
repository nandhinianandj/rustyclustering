//! HDBSCAN example demonstrating hierarchical clustering on 2D data.

use ghdbscan_core::{HDBSCAN, HDBSCANParams, DistanceMetric};
use ndarray::array;

fn main() {
    println!("HDBSCAN Clustering Example");
    println!("==========================\n");

    // Create sample data with clusters of varying density
    let data = array![
        // Dense cluster 1: points around (0, 0)
        [0.0, 0.0],
        [0.5, 0.0],
        [0.0, 0.5],
        [0.5, 0.5],
        [0.25, 0.25],
        // Less dense cluster 2: points around (10, 10)
        [10.0, 10.0],
        [11.0, 10.0],
        [10.0, 11.0],
        [11.0, 11.0],
        // Sparse cluster 3: points around (20, 0)
        [20.0, 0.0],
        [22.0, 0.0],
        [21.0, 1.0],
        // Noise points
        [50.0, 50.0],
        [60.0, 60.0],
    ];

    println!("Data points:");
    for (i, point) in data.outer_iter().enumerate() {
        println!("  Point {}: [{:.1}, {:.1}]", i, point[0], point[1]);
    }
    println!();

    // Configure HDBSCAN parameters
    let min_cluster_size = 3;
    let min_samples = Some(3);
    let params = HDBSCANParams::new(min_cluster_size, min_samples)
        .with_metric(DistanceMetric::Euclidean);

    println!("HDBSCAN Parameters:");
    println!("  min_cluster_size: {}", min_cluster_size);
    println!("  min_samples: {:?}", min_samples);
    println!("  metric: Euclidean\n");

    // Run HDBSCAN
    let hdbscan = HDBSCAN::new(params);
    let result = hdbscan.fit(&data.view());

    // Display results
    println!("Results:");
    println!("  Number of clusters: {}", result.n_clusters());
    println!("  Number of noise points: {}", result.n_noise());
    println!();

    println!("Cluster assignments:");
    for (i, &label) in result.labels.iter().enumerate() {
        let point = data.row(i);
        let prob = result.probabilities[i];
        let outlier_score = result.outlier_scores[i];
        
        if label == -1 {
            println!(
                "  Point {} [{:.1}, {:.1}]: Noise (outlier score: {:.3})",
                i, point[0], point[1], outlier_score
            );
        } else {
            println!(
                "  Point {} [{:.1}, {:.1}]: Cluster {} (probability: {:.3})",
                i, point[0], point[1], label, prob
            );
        }
    }
}
