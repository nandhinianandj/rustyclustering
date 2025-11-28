//! Training utilities for VaDE model.

use candle_core::{Result, Tensor, Device, DType, Var};
use candle_nn::{VarBuilder, VarMap, Optimizer, AdamW, ParamsAdamW};
use crate::{VaDE, VaDEConfig, TrainingConfig};

/// Trainer for VaDE model.
pub struct VaDETrainer {
    model: VaDE,
    optimizer: AdamW,
    config: TrainingConfig,
}

impl VaDETrainer {
    /// Create a new trainer.
    pub fn new(
        model_config: VaDEConfig,
        training_config: TrainingConfig,
        device: &Device,
    ) -> Result<Self> {
        // Create variable map for model parameters
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        
        // Create model
        let model = VaDE::new(model_config, vb, device)?;
        
        // Create optimizer
        let params = ParamsAdamW {
            lr: training_config.learning_rate,
            ..Default::default()
        };
        let optimizer = AdamW::new(varmap.all_vars(), params)?;
        
        Ok(Self {
            model,
            optimizer,
            config: training_config,
        })
    }
    
    /// Train for one epoch.
    pub fn train_epoch(&mut self, data: &Tensor) -> Result<f32> {
        let n_samples = data.dims()[0];
        let batch_size = self.config.batch_size;
        let n_batches = (n_samples + batch_size - 1) / batch_size;
        
        let mut total_loss = 0.0f32;
        
        for batch_idx in 0..n_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(n_samples);
            
            // Get batch
            let batch = data.narrow(0, start, end - start)?;
            
            // Forward pass
            let loss = self.model.loss(&batch)?;
            
            // Check for NaN
            let loss_val = loss.to_scalar::<f32>()?;
            if loss_val.is_nan() || loss_val.is_infinite() {
                eprintln!("Warning: NaN or Inf loss detected, skipping batch");
                continue;
            }
            
            // Backward pass with gradient clipping
            self.optimizer.backward_step(&loss)?;
            
            // Accumulate loss
            total_loss += loss_val;
        }
        
        Ok(total_loss / n_batches as f32)
    }
    
    /// Pretrain for one epoch (reconstruction loss only).
    pub fn pretrain_epoch(&mut self, data: &Tensor) -> Result<f32> {
        let n_samples = data.dims()[0];
        let batch_size = self.config.batch_size;
        let n_batches = (n_samples + batch_size - 1) / batch_size;
        
        let mut total_loss = 0.0f32;
        
        for batch_idx in 0..n_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(n_samples);
            
            // Get batch
            let batch = data.narrow(0, start, end - start)?;
            
            // Forward pass (reconstruction only)
            let loss = self.model.pretrain_loss(&batch)?;
            
            // Check for NaN
            let loss_val = loss.to_scalar::<f32>()?;
            if loss_val.is_nan() || loss_val.is_infinite() {
                eprintln!("Warning: NaN or Inf loss detected during pretraining, skipping batch");
                continue;
            }
            
            // Backward pass
            self.optimizer.backward_step(&loss)?;
            
            // Accumulate loss
            total_loss += loss_val;
        }
        
        Ok(total_loss / n_batches as f32)
    }

    /// Run pretraining phase.
    pub fn pretrain(&mut self, data: &Tensor) -> Result<Vec<f32>> {
        let epochs = self.config.pretrain_epochs;
        if epochs == 0 {
            return Ok(Vec::new());
        }
        
        println!("Starting pretraining for {} epochs...", epochs);
        let mut losses = Vec::new();
        
        for epoch in 0..epochs {
            let loss = self.pretrain_epoch(data)?;
            losses.push(loss);
            
            if epoch % 10 == 0 {
                println!("Pretrain Epoch {}: Loss = {:.4}", epoch, loss);
            }
        }
        println!("Pretraining complete. Final loss: {:.4}", losses.last().unwrap_or(&0.0));
        
        Ok(losses)
    }

    /// Train for multiple epochs.
    pub fn train(&mut self, data: &Tensor, epochs: usize) -> Result<Vec<f32>> {
        // Run pretraining first if configured
        if self.config.pretrain_epochs > 0 {
            self.pretrain(data)?;
        }
        
        println!("Starting main training for {} epochs...", epochs);
        let mut losses = Vec::new();
        
        for epoch in 0..epochs {
            let loss = self.train_epoch(data)?;
            losses.push(loss);
            
            if epoch % 10 == 0 {
                println!("Epoch {}: Loss = {:.4}", epoch, loss);
            }
        }
        
        Ok(losses)
    }
    
    /// Get reference to the model.
    pub fn model(&self) -> &VaDE {
        &self.model
    }
}
