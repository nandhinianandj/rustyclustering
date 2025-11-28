//! Simple example demonstrating VaDE clustering on synthetic data.

use candle_core::{Device, DType, Tensor, IndexOp};
use ghdbscan_vae::{VaDEConfig, TrainingConfig, VaDETrainer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("VaDE Deep Clustering Example");
    println!("============================\n");
    
    // Use CPU for this example
    let device = Device::Cpu;
    
    // Generate synthetic data - 3 Gaussian clusters
    println!("Generating synthetic data...");
    let data = generate_synthetic_data(&device)?;
    println!("Data shape: {:?}", data.dims());
    println!("Number of samples: {}\n", data.dims()[0]);
    
    // Configure model
    let model_config = VaDEConfig::new(
        2,   // input_dim (2D data)
        2,   // latent_dim
        3,   // n_clusters
    );
    
    let training_config = TrainingConfig::new()
        .with_epochs(50)  // Fewer epochs for demo
        .with_batch_size(32)
        .with_learning_rate(0.0001);  // Lower LR for stability
    
    println!("Model Configuration:");
    println!("  Input dim: {}", model_config.input_dim);
    println!("  Latent dim: {}", model_config.latent_dim);
    println!("  Number of clusters: {}", model_config.n_clusters);
    println!("\nTraining Configuration:");
    println!("  Epochs: {}", training_config.epochs);
    println!("  Batch size: {}", training_config.batch_size);
    println!("  Learning rate: {}\n", training_config.learning_rate);
    
    // Create trainer
    println!("Initializing model...");
    let mut trainer = VaDETrainer::new(model_config, training_config.clone(), &device)?;
    println!("Model initialized successfully!\n");
    
    // Train
    println!("Training VaDE model...");
    let losses = trainer.train(&data, training_config.epochs)?;
    println!("\nTraining complete!");
    println!("Final loss: {:.4}\n", losses.last().unwrap());
    
    // Predict clusters
    println!("Predicting clusters...");
    let model = trainer.model();
    let labels = model.predict(&data)?;
    let probs = model.predict_proba(&data)?;
    
    println!("Clustering complete!");
    println!("\nSample predictions:");
    for i in 0..10.min(data.dims()[0]) {
        let label = labels.i(i)?.to_scalar::<u32>()?;
        let prob_vec = probs.i(i)?;
        println!("  Sample {}: Cluster {} (confidence: {:.3})", 
            i, label, prob_vec.max(0)?.to_scalar::<f32>()?);
    }
    
    // Count samples per cluster
    println!("\nCluster distribution:");
    let labels_vec: Vec<u32> = labels.to_vec1()?;
    for cluster_id in 0..3 {
        let count = labels_vec.iter().filter(|&&l| l == cluster_id).count();
        println!("  Cluster {}: {} samples", cluster_id, count);
    }
    
    println!("\n✅ Example completed successfully!");
    
    Ok(())
}

/// Generate synthetic data with 3 Gaussian clusters.
fn generate_synthetic_data(device: &Device) -> candle_core::Result<Tensor> {
    let n_samples_per_cluster = 100;
    let n_clusters = 3;
    
    // Cluster centers
    let centers = vec![
        vec![0.0f32, 0.0f32],
        vec![5.0f32, 5.0f32],
        vec![10.0f32, 0.0f32],
    ];
    
    let mut all_data = Vec::new();
    
    for (_cluster_id, center) in centers.iter().enumerate() {
        // Generate samples around this center
        for _ in 0..n_samples_per_cluster {
            // Add Gaussian noise
            let noise_x = (rand::random::<f32>() - 0.5) * 2.0;
            let noise_y = (rand::random::<f32>() - 0.5) * 2.0;
            
            all_data.push(center[0] + noise_x);
            all_data.push(center[1] + noise_y);
        }
    }
    
    // Create tensor
    let total_samples = n_samples_per_cluster * n_clusters;
    let data = Tensor::from_vec(all_data, (total_samples, 2), device)?;
    
    // Normalize to [0, 1] for BCE loss
    let min_val = data.min(0)?.min(0)?;
    let max_val = data.max(0)?.max(0)?;
    let range = max_val.sub(&min_val)?;
    
    // (data - min) / range
    data.broadcast_sub(&min_val.unsqueeze(0)?.broadcast_as(data.shape())?)?
        .broadcast_div(&range.unsqueeze(0)?.broadcast_as(data.shape())?)
}
