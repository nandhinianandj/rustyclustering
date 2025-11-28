//! Encoder network for VaDE using Candle.
//!
//! Maps input data to latent space distribution parameters (mean and log-variance).

use candle_core::{Result, Tensor, Device, DType};
use candle_nn::{Linear, VarBuilder, Module, Activation};

/// Encoder network that maps input to latent distribution parameters.
#[derive(Debug)]
pub struct Encoder {
    layers: Vec<Linear>,
    fc_mean: Linear,
    fc_logvar: Linear,
}

impl Encoder {
    /// Create a new encoder.
    ///
    /// # Arguments
    /// * `input_dim` - Dimensionality of input data
    /// * `hidden_dims` - Sizes of hidden layers
    /// * `latent_dim` - Dimensionality of latent space
    /// * `vb` - Variable builder for parameter initialization
    pub fn new(
        input_dim: usize,
        hidden_dims: &[usize],
        latent_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = Vec::new();
        let mut prev_dim = input_dim;
        
        // Create hidden layers
        for (i, &hidden_dim) in hidden_dims.iter().enumerate() {
            layers.push(
                candle_nn::linear(prev_dim, hidden_dim, vb.pp(format!("encoder_layer_{}", i)))?
            );
            prev_dim = hidden_dim;
        }
        
        // Output layers for mean and log-variance
        let fc_mean = candle_nn::linear(prev_dim, latent_dim, vb.pp("fc_mean"))?;
        let fc_logvar = candle_nn::linear(prev_dim, latent_dim, vb.pp("fc_logvar"))?;
        
        Ok(Self {
            layers,
            fc_mean,
            fc_logvar,
        })
    }
    
    /// Forward pass through encoder.
    ///
    /// Returns (mean, log_variance) of latent distribution.
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut h = x.clone();
        
        // Pass through hidden layers with ReLU activation
        for layer in &self.layers {
            h = layer.forward(&h)?;
            h = h.relu()?;
        }
        
        // Compute mean and log-variance
        let mean = self.fc_mean.forward(&h)?;
        let logvar = self.fc_logvar.forward(&h)?;
        
        Ok((mean, logvar))
    }
    
    /// Encode input to latent space using reparameterization trick.
    ///
    /// z = μ + σ * ε, where ε ~ N(0, 1)
    pub fn encode(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (mean, logvar) = self.forward(x)?;
        
        // Reparameterization trick: z = mean + std * epsilon
        let std = logvar.affine(0.5, 0.0)?.exp()?;  // exp(0.5 * logvar)
        let epsilon = mean.randn_like(0.0, 1.0)?;
        let z = mean.add(&std.mul(&epsilon)?)?;
        
        Ok((z, mean, logvar))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    
    #[test]
    fn test_encoder_creation() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        
        let encoder = Encoder::new(
            784,  // MNIST input
            &[500, 500, 2000],
            10,   // latent dim
            vb,
        )?;
        
        // Test forward pass
        let batch_size = 32;
        let input = Tensor::randn(0.0f32, 1.0, (batch_size, 784), &device)?;
        
        let (mean, logvar) = encoder.forward(&input)?;
        assert_eq!(mean.dims(), &[batch_size, 10]);
        assert_eq!(logvar.dims(), &[batch_size, 10]);
        
        Ok(())
    }
}
