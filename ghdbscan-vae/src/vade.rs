//! VaDE (Variational Deep Embedding) model implementation using Candle.
//!
//! Combines VAE with Gaussian Mixture Model for deep clustering.

use candle_core::{Result, Tensor, Device, DType, Var, IndexOp};
use candle_nn::VarBuilder;

use crate::{encoder::Encoder, decoder::Decoder, gmm::GaussianMixtureModel, config::VaDEConfig};

/// VaDE model for deep clustering.
#[derive(Debug)]
pub struct VaDE {
    encoder: Encoder,
    decoder: Decoder,
    gmm: GaussianMixtureModel,
    config: VaDEConfig,
}

impl VaDE {
    /// Create a new VaDE model.
    pub fn new(config: VaDEConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        let encoder = Encoder::new(
            config.input_dim,
            &config.encoder_hidden,
            config.latent_dim,
            vb.pp("encoder"),
        )?;
        
        let decoder = Decoder::new(
            config.latent_dim,
            &config.decoder_hidden,
            config.input_dim,
            vb.pp("decoder"),
        )?;
        
        let gmm = GaussianMixtureModel::new(
            config.n_clusters,
            config.latent_dim,
            device,
        )?;
        
        Ok(Self {
            encoder,
            decoder,
            gmm,
            config,
        })
    }
    
    /// Forward pass through the model.
    ///
    /// Returns: (reconstruction, z, mean, logvar)
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        // Encode
        let (z, mean, logvar) = self.encoder.encode(x)?;
        
        // Decode
        let reconstruction = self.decoder.decode(&z)?;
        
        Ok((reconstruction, z, mean, logvar))
    }
    
    /// Compute ELBO loss.
    ///
    /// ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
    pub fn loss(&self, x: &Tensor) -> Result<Tensor> {
        let (reconstruction, z, mean, logvar) = self.forward(x)?;
        
        // Reconstruction loss (binary cross-entropy)
        let recon_loss = self.reconstruction_loss(x, &reconstruction)?;
        
        // KL divergence with GMM prior
        let kl_loss = self.gmm.kl_divergence(&z, &mean, &logvar)?;
        
        // Total loss (mean over batch)
        let total_loss = (recon_loss + kl_loss)?;
        total_loss.mean_all()
    }
    
    /// Compute reconstruction loss (binary cross-entropy).
    pub fn reconstruction_loss(&self, x: &Tensor, reconstruction: &Tensor) -> Result<Tensor> {
        // BCE: -[x * log(recon) + (1-x) * log(1-recon)]
        // Add epsilon for numerical stability
        let eps = 1e-7f64;
        
        // Clamp reconstruction to avoid log(0)
        let recon_clamped = reconstruction.clamp(eps, 1.0 - eps)?;
        
        // Clamp input as well for stability
        let x_clamped = x.clamp(eps, 1.0 - eps)?;
        
        let term1 = x_clamped.mul(&recon_clamped.log()?)?;
        let one_minus_x = x_clamped.affine(-1.0, 1.0)?;
        let one_minus_recon = recon_clamped.affine(-1.0, 1.0)?;
        let term2 = one_minus_x.mul(&one_minus_recon.log()?)?;
        let sum = term1.add(&term2)?;
        let bce = sum.neg()?;
        
        bce.sum(1)
    }

    /// Compute loss for pretraining (reconstruction only).
    pub fn pretrain_loss(&self, x: &Tensor) -> Result<Tensor> {
        let (reconstruction, _, _, _) = self.forward(x)?;
        let loss = self.reconstruction_loss(x, &reconstruction)?;
        loss.mean_all()
    }
    
    /// Predict cluster assignments.
    pub fn predict(&self, x: &Tensor) -> Result<Tensor> {
        let (_, z, _, _) = self.forward(x)?;
        self.gmm.predict(&z)
    }
    
    /// Predict cluster probabilities.
    pub fn predict_proba(&self, x: &Tensor) -> Result<Tensor> {
        let (_, z, _, _) = self.forward(x)?;
        self.gmm.predict_proba(&z)
    }
    
    /// Generate samples from a specific cluster.
    pub fn generate_from_cluster(&self, cluster_id: usize, n_samples: usize, device: &Device) -> Result<Tensor> {
        // Get cluster parameters
        let means_tensor = self.gmm.means.as_tensor();
        let logvars_tensor = self.gmm.logvars.as_tensor();
        
        // Extract mean and logvar for this cluster
        let mean = means_tensor.i(cluster_id)?;
        let logvar = logvars_tensor.i(cluster_id)?;
        
        // Sample epsilon ~ N(0, 1)
        let epsilon = Tensor::randn(0.0f32, 1.0, (n_samples, self.config.latent_dim), device)?;
        
        // z = mean + std * epsilon
        let std = logvar.affine(0.5, 0.0)?.exp()?;  // 0.5 * logvar
        let mean_expanded = mean.unsqueeze(0)?.broadcast_as(epsilon.shape())?;
        let std_expanded = std.unsqueeze(0)?.broadcast_as(epsilon.shape())?;
        let z = mean_expanded.add(&std_expanded.mul(&epsilon)?)?;
        
        // Decode to data space
        self.decoder.decode(&z)
    }
    
    /// Get all trainable variables for optimization.
    pub fn vars(&self) -> Vec<&Var> {
        let mut vars = Vec::new();
        
        // Add GMM variables
        vars.extend(self.gmm.vars());
        
        // Note: Encoder and decoder variables are managed by VarBuilder
        // and will be automatically included in the optimizer
        
        vars
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vade_creation() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let config = VaDEConfig::new(784, 10, 10);
        
        let vade = VaDE::new(config, vb, &device)?;
        
        // Test forward pass
        let batch_size = 32;
        let input = Tensor::randn(0.0f32, 1.0, (batch_size, 784), &device)?;
        
        let (recon, z, mean, logvar) = vade.forward(&input)?;
        assert_eq!(recon.dims(), &[batch_size, 784]);
        assert_eq!(z.dims(), &[batch_size, 10]);
        assert_eq!(mean.dims(), &[batch_size, 10]);
        assert_eq!(logvar.dims(), &[batch_size, 10]);
        
        Ok(())
    }
    
    #[test]
    fn test_vade_loss() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let config = VaDEConfig::new(784, 10, 10);
        
        let vade = VaDE::new(config, vb, &device)?;
        
        let batch_size = 32;
        let input = Tensor::randn(0.0f32, 1.0, (batch_size, 784), &device)?;
        
        let loss = vade.loss(&input)?;
        assert_eq!(loss.dims(), &[]);  // Scalar loss
        
        Ok(())
    }
}
