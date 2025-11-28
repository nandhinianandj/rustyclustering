//! # ghdbscan-vae
//!
//! Deep clustering using Variational Deep Embedding (VaDE).
//!
//! VaDE combines Variational Autoencoders with Gaussian Mixture Models
//! to perform unsupervised clustering in a learned latent space.

pub mod encoder;
pub mod decoder;
pub mod gmm;
pub mod vade;
pub mod config;
pub mod trainer;

pub use vade::VaDE;
pub use config::{VaDEConfig, TrainingConfig};
pub use trainer::VaDETrainer;

#[cfg(test)]
mod tests;
