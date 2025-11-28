//! # ghdbscan-gpu
//!
//! GPU-accelerated implementations of DBSCAN and HDBSCAN clustering algorithms.
//!
//! ## Features
//!
//! - **GPU-accelerated**: Leverages GPU compute for massive parallelism
//! - **Cross-platform**: Uses wgpu for broad GPU compatibility (NVIDIA, AMD, Intel)
//! - **Optional CUDA**: Maximum performance on NVIDIA GPUs with CUDA feature
//! - **Automatic fallback**: Falls back to CPU for small datasets
//!
//! ## Example
//!
//! ```rust,no_run
//! use ghdbscan_gpu::{DBSCAN_GPU, DBSCANParams};
//! use ndarray::array;
//!
//! # async fn example() {
//! let data = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0]];
//! let params = DBSCANParams::new(2.0, 2);
//! let dbscan = DBSCAN_GPU::new(params).await.unwrap();
//! let result = dbscan.fit(&data.view()).await;
//! # }
//! ```

pub mod gpu_context;
pub mod dbscan_gpu;
pub mod hdbscan_gpu;
mod utils;

pub use gpu_context::{GpuContext, GpuError};
pub use dbscan_gpu::{DBSCAN_GPU, DBSCANParams, ClusterResult};
pub use hdbscan_gpu::{HDBSCAN_GPU, HDBSCANParams, HDBSCANResult};

// Re-export common types from core
pub use ghdbscan_core::{DistanceMetric, Distance};

#[cfg(test)]
mod tests;
