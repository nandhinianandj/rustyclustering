//! Configuration structures for VaDE model and training.

use serde::{Deserialize, Serialize};

/// Configuration for VaDE model architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaDEConfig {
    /// Number of input features
    pub input_dim: usize,
    
    /// Dimensionality of latent space
    pub latent_dim: usize,
    
    /// Number of clusters
    pub n_clusters: usize,
    
    /// Hidden layer sizes for encoder
    pub encoder_hidden: Vec<usize>,
    
    /// Hidden layer sizes for decoder
    pub decoder_hidden: Vec<usize>,
}

impl VaDEConfig {
    /// Create a new VaDE configuration.
    pub fn new(input_dim: usize, latent_dim: usize, n_clusters: usize) -> Self {
        Self {
            input_dim,
            latent_dim,
            n_clusters,
            // Default architecture from VaDE paper
            encoder_hidden: vec![500, 500, 2000],
            decoder_hidden: vec![2000, 500, 500],
        }
    }
    
    /// Set custom encoder hidden layers.
    pub fn with_encoder_hidden(mut self, hidden: Vec<usize>) -> Self {
        self.encoder_hidden = hidden;
        self
    }
    
    /// Set custom decoder hidden layers.
    pub fn with_decoder_hidden(mut self, hidden: Vec<usize>) -> Self {
        self.decoder_hidden = hidden;
        self
    }
}

/// Training configuration for VaDE.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of training epochs
    pub epochs: usize,
    
    /// Batch size
    pub batch_size: usize,
    
    /// Learning rate
    pub learning_rate: f64,
    
    /// Number of pretraining epochs (autoencoder only)
    pub pretrain_epochs: usize,
    
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 32,
            learning_rate: 0.0001,  // Lower LR for stability
            pretrain_epochs: 50,
            seed: 42,
        }
    }
}

impl TrainingConfig {
    /// Create a new training configuration.
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set number of epochs.
    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }
    
    /// Set batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
    
    /// Set learning rate.
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }
    
    /// Set pretraining epochs.
    pub fn with_pretrain_epochs(mut self, epochs: usize) -> Self {
        self.pretrain_epochs = epochs;
        self
    }
    
    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}
