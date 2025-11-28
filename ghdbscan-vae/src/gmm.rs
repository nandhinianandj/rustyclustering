//! Gaussian Mixture Model components for VaDE using Candle.
//!
//! Implements the GMM prior in latent space with learnable parameters.

use candle_core::{Result, Tensor, Device, DType, Var};
use std::f64::consts::PI;

/// Gaussian Mixture Model for VaDE latent space prior.
#[derive(Debug)]
pub struct GaussianMixtureModel {
    /// Cluster means (K x latent_dim)
    pub means: Var,
    
    /// Cluster log-variances (K x latent_dim) - diagonal covariance
    pub logvars: Var,
    
    /// Mixing coefficients logits (K,) - cluster probabilities
    pub pi_logits: Var,
    
    n_clusters: usize,
    latent_dim: usize,
}

impl GaussianMixtureModel {
    /// Create a new GMM with random initialization.
    pub fn new(n_clusters: usize, latent_dim: usize, device: &Device) -> Result<Self> {
        // Initialize means randomly
        let means = Var::from_tensor(&Tensor::randn(
            0.0f32,
            1.0,
            (n_clusters, latent_dim),
            device,
        )?)?;
        
        // Initialize log-variances to small values
        let logvars = Var::from_tensor(&Tensor::zeros((n_clusters, latent_dim), DType::F32, device)?.affine(-1.0, 0.0)?)?;
        
        // Initialize mixing coefficients uniformly
        let pi_logits = Var::from_tensor(&Tensor::zeros(n_clusters, DType::F32, device)?)?;
        
        Ok(Self {
            means,
            logvars,
            pi_logits,
            n_clusters,
            latent_dim,
        })
    }
    
    /// Compute log probability of latent points under each cluster.
    ///
    /// Returns: (batch_size, n_clusters)
    pub fn log_prob_clusters(&self, z: &Tensor) -> Result<Tensor> {
        // Expand dimensions for broadcasting
        // z: (batch_size, latent_dim) -> (batch_size, 1, latent_dim)
        let z_expanded = z.unsqueeze(1)?;
        
        // means: (n_clusters, latent_dim) -> (1, n_clusters, latent_dim)
        let means_expanded = self.means.unsqueeze(0)?;
        
        // logvars: (n_clusters, latent_dim) -> (1, n_clusters, latent_dim)
        let logvars_expanded = self.logvars.unsqueeze(0)?;
        
        // Compute squared Mahalanobis distance
        let diff = z_expanded.broadcast_sub(&means_expanded)?;
        let var = logvars_expanded.exp()?;
        let mahalanobis = diff.sqr()?.broadcast_div(&var)?.sum(2)?;  // (batch_size, n_clusters)
        
        // Log probability: -0.5 * (log(2π) + logvar + mahalanobis)
        let log_2pi = (2.0 * PI).ln() as f32;
        let log_2pi_term = (log_2pi * self.latent_dim as f32) as f64;
        
        // Sum logvars over latent dimension: (1, n_clusters, latent_dim) -> (1, n_clusters)
        let logvar_sum_single = logvars_expanded.sum(2)?;  // (1, n_clusters)
        
        // Broadcast to match batch size
        let batch_size = z.dims()[0];
        let logvar_sum = logvar_sum_single.broadcast_as((batch_size, self.n_clusters))?;  // (batch_size, n_clusters)
        
        let sum = logvar_sum.add(&mahalanobis)?;
        let sum_with_scalar = sum.affine(1.0, log_2pi_term)?;
        sum_with_scalar.affine(-0.5, 0.0)  // Return (batch_size, n_clusters)
    }
    
    /// Compute cluster assignment probabilities (soft clustering).
    ///
    /// Returns: (batch_size, n_clusters)
    pub fn predict_proba(&self, z: &Tensor) -> Result<Tensor> {
        // Get mixing coefficients (cluster priors)
        let pi = candle_nn::ops::softmax(&self.pi_logits, 0)?;
        
        // Compute log probabilities (batch_size, n_clusters)
        let log_prob = self.log_prob_clusters(z)?;
        
        // Add log mixing coefficients
        let log_pi = pi.log()?;  // (n_clusters,)
        
        // Broadcast log_pi to match log_prob shape
        let batch_size = log_prob.dims()[0];
        let log_pi_expanded = log_pi.unsqueeze(0)?.broadcast_as((batch_size, self.n_clusters))?;
        let log_weighted = log_prob.add(&log_pi_expanded)?;
        
        // Normalize to get probabilities
        candle_nn::ops::softmax(&log_weighted, 1)
    }
    
    /// Get hard cluster assignments.
    ///
    /// Returns: (batch_size,) - cluster indices
    pub fn predict(&self, z: &Tensor) -> Result<Tensor> {
        let proba = self.predict_proba(z)?;
        proba.argmax(1)
    }
    
    /// Compute KL divergence between posterior and GMM prior.
    ///
    /// This is used in the ELBO loss.
    pub fn kl_divergence(
        &self,
        z: &Tensor,
        mean: &Tensor,
        logvar: &Tensor,
    ) -> Result<Tensor> {
        // Get cluster probabilities
        let gamma = self.predict_proba(z)?;
        
        // Compute log q(z|x) - Gaussian posterior
        let log_q = {
            let diff = z.broadcast_sub(mean)?;
            let log_2pi = (2.0 * PI).ln() as f32;
            let term1_scalar = (log_2pi * self.latent_dim as f32) as f64;
            let term2 = logvar.sum(1)?;
            let term3 = diff.sqr()?.broadcast_div(&logvar.exp()?)?.sum(1)?;
            
            // Combine: (term1_scalar + term2 + term3) * -0.5
            let sum = term2.add(&term3)?;
            let sum_with_scalar = sum.affine(1.0, term1_scalar)?;
            sum_with_scalar.affine(-0.5, 0.0)?
        };
        
        // Compute log p(z) - GMM prior
        let log_p = {
            let log_prob_clusters = self.log_prob_clusters(z)?;  // (batch_size, n_clusters)
            let pi = candle_nn::ops::softmax(&self.pi_logits, 0)?;  // (n_clusters,)
            let log_pi = pi.log()?;  // (n_clusters,)
            
            // Broadcast log_pi to match log_prob_clusters shape
            let batch_size = log_prob_clusters.dims()[0];
            let log_pi_expanded = log_pi.unsqueeze(0)?.broadcast_as((batch_size, self.n_clusters))?;
            let log_weighted = log_prob_clusters.add(&log_pi_expanded)?;
            
            // Log-sum-exp for numerical stability
            let max_val = log_weighted.max(1)?;  // (batch_size,)
            let max_val_expanded = max_val.unsqueeze(1)?.broadcast_as(log_weighted.shape())?;
            let exp_sum = log_weighted.sub(&max_val_expanded)?.exp()?.sum(1)?;
            max_val.add(&exp_sum.log()?)?
        };
        
        // KL = E[log q(z|x) - log p(z)]
        log_q.sub(&log_p)
    }
    
    /// Get all trainable variables.
    pub fn vars(&self) -> Vec<&Var> {
        vec![&self.means, &self.logvars, &self.pi_logits]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gmm_creation() -> Result<()> {
        let device = Device::Cpu;
        let gmm = GaussianMixtureModel::new(10, 10, &device)?;
        
        assert_eq!(gmm.n_clusters, 10);
        assert_eq!(gmm.latent_dim, 10);
        
        Ok(())
    }
    
    #[test]
    fn test_gmm_prediction() -> Result<()> {
        let device = Device::Cpu;
        let gmm = GaussianMixtureModel::new(5, 10, &device)?;
        
        let batch_size = 32;
        let z = Tensor::randn(0.0f32, 1.0, (batch_size, 10), &device)?;
        
        let proba = gmm.predict_proba(&z)?;
        assert_eq!(proba.dims(), &[batch_size, 5]);
        
        let labels = gmm.predict(&z)?;
        assert_eq!(labels.dims(), &[batch_size]);
        
        Ok(())
    }
}
