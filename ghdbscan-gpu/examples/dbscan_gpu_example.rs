//! Example demonstrating GPU-accelerated DBSCAN clustering.

use ghdbscan_gpu::{DBSCAN_GPU, DBSCANParams};
use ndarray::array;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("GPU-Accelerated DBSCAN Example");
    println!("================================\n");

    // Create sample data - two clear clusters
    let data = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
        [10.0, 10.0],
        [11.0, 10.0],
        [10.0, 11.0],
        [11.0, 11.0],
        [10.5, 10.5],
        [100.0, 100.0], // Noise point
    ];

    println!("Dataset: {} points, {} features", data.nrows(), data.ncols());
    println!("Data:\n{:?}\n", data);

    // Create DBSCAN parameters
    let params = DBSCANParams::new(2.0, 2);
    println!("Parameters:");
    println!("  eps: {}", params.eps);
    println!("  min_samples: {}\n", params.min_samples);

    // Create GPU DBSCAN instance
    println!("Initializing GPU context...");
    let dbscan = DBSCAN_GPU::new(params).await?;
    println!("GPU initialized successfully!\n");

    // Fit the model
    println!("Running DBSCAN on GPU...");
    let result = dbscan.fit(&data.view()).await?;
    println!("Clustering complete!\n");

    // Print results
    println!("Results:");
    println!("  Number of clusters: {}", result.n_clusters());
    println!("  Number of noise points: {}", result.n_noise());
    println!("\nCluster labels:");
    for (i, &label) in result.labels.iter().enumerate() {
        let point_type = if result.core_points[i] {
            "core"
        } else if label == -1 {
            "noise"
        } else {
            "border"
        };
        println!("  Point {}: cluster {} ({})", i, label, point_type);
    }

    Ok(())
}
