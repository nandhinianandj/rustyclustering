//! DBSCAN example demonstrating clustering on 2D data.

use ghdbscan_core::{DBSCAN, DBSCANParams, DistanceMetric};
use ndarray::array;

fn main() {
    println!("DBSCAN Clustering Example");
    println!("=========================\n");

    // Create sample data with two distinct clusters
    let data = array![
        // Cluster 1: points around (0, 0)
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
        // Cluster 2: points around (10, 10)
        [10.0, 10.0],
        [11.0, 10.0],
        [10.0, 11.0],
        [11.0, 11.0],
        [10.5, 10.5],
        // Noise point
        [50.0, 50.0],
    ];

    println!("Data points:");
    for (i, point) in data.outer_iter().enumerate() {
        println!("  Point {}: [{:.1}, {:.1}]", i, point[0], point[1]);
    }
    println!();

    // Configure DBSCAN parameters
    let eps = 2.0;
    let min_samples = 3;
    let params = DBSCANParams::new(eps, min_samples)
        .with_metric(DistanceMetric::Euclidean);

    println!("DBSCAN Parameters:");
    println!("  eps: {}", eps);
    println!("  min_samples: {}", min_samples);
    println!("  metric: Euclidean\n");

    // Run DBSCAN
    let dbscan = DBSCAN::new(params);
    let result = dbscan.fit(&data.view());

    // Display results
    println!("Results:");
    println!("  Number of clusters: {}", result.n_clusters());
    println!("  Number of noise points: {}", result.n_noise());
    println!();

    println!("Cluster assignments:");
    for (i, (&label, &is_core)) in result.labels.iter().zip(result.core_points.iter()).enumerate() {
        let point = data.row(i);
        let point_type = if label == -1 {
            "Noise"
        } else if is_core {
            "Core"
        } else {
            "Border"
        };
        println!(
            "  Point {} [{:.1}, {:.1}]: Cluster {} ({})",
            i, point[0], point[1], label, point_type
        );
    }
}
