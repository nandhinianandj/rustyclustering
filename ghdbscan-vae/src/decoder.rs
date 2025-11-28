//! Decoder network for VaDE using Candle.
//!
//! Maps latent space representations back to data space.

use candle_core::{Result, Tensor, DType};
use candle_nn::{Linear, VarBuilder, Module};

/// Decoder network that reconstructs data from latent representations.
#[derive(Debug)]
pub struct Decoder {
    layers: Vec<Linear>,
    fc_out: Linear,
}

impl Decoder {
    /// Create a new decoder.
    ///
    /// # Arguments
    /// * `latent_dim` - Dimensionality of latent space
    /// * `hidden_dims` - Sizes of hidden layers (in reverse order from encoder)
    /// * `output_dim` - Dimensionality of output data
    /// * `vb` - Variable builder for parameter initialization
    pub fn new(
        latent_dim: usize,
        hidden_dims: &[usize],
        output_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = Vec::new();
        let mut prev_dim = latent_dim;
        
        // Create hidden layers
        for (i, &hidden_dim) in hidden_dims.iter().enumerate() {
            layers.push(
                candle_nn::linear(prev_dim, hidden_dim, vb.pp(format!("decoder_layer_{}", i)))?
            );
            prev_dim = hidden_dim;
        }
        
        // Output layer
        let fc_out = candle_nn::linear(prev_dim, output_dim, vb.pp("fc_out"))?;
        
        Ok(Self {
            layers,
            fc_out,
        })
    }
    
    /// Forward pass through decoder.
    ///
    /// Returns reconstructed data (with sigmoid activation for normalized data).
    pub fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let mut h = z.clone();
        
        // Pass through hidden layers with ReLU activation
        for layer in &self.layers {
            h = layer.forward(&h)?;
            h = h.relu()?;
        }
        
        // Output layer with sigmoid activation (for data in [0, 1])
        let output = self.fc_out.forward(&h)?;
        candle_nn::ops::sigmoid(&output)
    }
    
    /// Decode latent representation to data space.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        self.forward(z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    
    #[test]
    fn test_decoder_creation() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        
        let decoder = Decoder::new(
            10,   // latent dim
            &[2000, 500, 500],
            784,  // MNIST output
            vb,
        )?;
        
        // Test forward pass
        let batch_size = 32;
        let latent = Tensor::randn(0.0f32, 1.0, (batch_size, 10), &device)?;
        
        let output = decoder.forward(&latent)?;
        assert_eq!(output.dims(), &[batch_size, 784]);
        
        Ok(())
    }
}
