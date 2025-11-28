//! Example demonstrating GPU-accelerated HDBSCAN clustering.

use ghdbscan_gpu::{HDBSCAN_GPU, HDBSCANParams};
use ndarray::array;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("GPU-Accelerated HDBSCAN Example");
    println!("=================================\n");

    // Create sample data - clusters with varying densities
    let data = array![
        // Dense cluster 1
        [0.0, 0.0],
        [0.5, 0.0],
        [0.0, 0.5],
        [0.5, 0.5],
        [0.25, 0.25],
        // Less dense cluster 2
        [10.0, 10.0],
        [12.0, 10.0],
        [10.0, 12.0],
        [12.0, 12.0],
        // Noise points
        [100.0, 100.0],
        [105.0, 105.0],
    ];

    println!("Dataset: {} points, {} features", data.nrows(), data.ncols());
    println!("Data:\n{:?}\n", data);

    // Create HDBSCAN parameters
    let params = HDBSCANParams::new(3, Some(2));
    println!("Parameters:");
    println!("  min_cluster_size: {}", params.min_cluster_size);
    println!("  min_samples: {}\n", params.min_samples);

    // Create GPU HDBSCAN instance
    println!("Initializing GPU context...");
    let hdbscan = HDBSCAN_GPU::new(params).await?;
    println!("GPU initialized successfully!\n");

    // Fit the model
    println!("Running HDBSCAN on GPU...");
    let result = hdbscan.fit(&data.view()).await?;
    println!("Clustering complete!\n");

    // Print results
    println!("Results:");
    println!("  Number of clusters: {}", result.n_clusters());
    println!("  Number of noise points: {}", result.n_noise());
    println!("\nCluster labels and probabilities:");
    for (i, &label) in result.labels.iter().enumerate() {
        println!(
            "  Point {}: cluster {} (probability: {:.3}, outlier score: {:.3})",
            i, label, result.probabilities[i], result.outlier_scores[i]
        );
    }

    Ok(())
}
