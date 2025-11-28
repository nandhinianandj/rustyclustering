//! Benchmark comparing CPU and GPU DBSCAN implementations.

use ghdbscan_core::DBSCAN as DBSCAN_CPU;
use ghdbscan_gpu::{DBSCAN_GPU, DBSCANParams};
use ndarray::{Array2, Axis};
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("DBSCAN Benchmark: CPU vs GPU");
    println!("============================");

    // Generate synthetic data (5000 points, 2 clusters)
    let n_points = 5000;
    let n_features = 2;
    let mut data = Array2::<f64>::zeros((n_points, n_features));
    
    // Cluster 1: centered at (0, 0)
    for i in 0..n_points/2 {
        data[[i, 0]] = rand::random::<f64>() + 0.0;
        data[[i, 1]] = rand::random::<f64>() + 0.0;
    }
    
    // Cluster 2: centered at (5, 5)
    for i in n_points/2..n_points {
        data[[i, 0]] = rand::random::<f64>() + 5.0;
        data[[i, 1]] = rand::random::<f64>() + 5.0;
    }

    println!("Dataset shape: {:?}", data.dim());

    // Parameters
    let eps = 0.5;
    let min_samples = 5;
    let params = DBSCANParams::new(eps, min_samples);

    // CPU Run
    println!("\nRunning CPU DBSCAN...");
    let start = Instant::now();
    let cpu_params = ghdbscan_core::DBSCANParams::new(eps, min_samples);
    let cpu_dbscan = DBSCAN_CPU::new(cpu_params);
    let cpu_result = cpu_dbscan.fit(&data.view());
    let cpu_duration = start.elapsed();
    println!("CPU Time: {:.4}s", cpu_duration.as_secs_f64());
    println!("CPU Clusters: {}", cpu_result.n_clusters());

    // GPU Run
    println!("\nRunning GPU DBSCAN...");
    let start = Instant::now();
    let gpu_dbscan = DBSCAN_GPU::new(params).await?;
    let gpu_result = gpu_dbscan.fit(&data.view()).await?;
    let gpu_duration = start.elapsed();
    println!("GPU Time: {:.4}s", gpu_duration.as_secs_f64());
    println!("GPU Clusters: {}", gpu_result.n_clusters());

    // Comparison
    println!("\nSpeedup: {:.2}x", cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64());
    println!("Results match: {}", cpu_result.n_clusters() == gpu_result.n_clusters());

    Ok(())
}
