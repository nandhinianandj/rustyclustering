//! # ghdbscan-core
//!
//! A high-performance implementation of DBSCAN and HDBSCAN clustering algorithms in Rust.
//!
//! ## Features
//!
//! - **DBSCAN**: Density-based spatial clustering with configurable distance metrics
//! - **HDBSCAN**: Hierarchical DBSCAN with automatic cluster extraction
//! - **Efficient**: Uses KD-trees for fast neighbor queries
//! - **Parallel**: Leverages Rayon for multi-threaded processing
//!
//! ## Example
//!
//! ```rust
//! use ghdbscan_core::{DBSCAN, DBSCANParams, DistanceMetric};
//! use ndarray::array;
//!
//! let data = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0]];
//! let params = DBSCANParams::new(2.0, 2).with_metric(DistanceMetric::Euclidean);
//! let dbscan = DBSCAN::new(params);
//! let result = dbscan.fit(&data.view());
//! ```

pub mod distance;
pub mod spatial_index;
pub mod dbscan;
pub mod hdbscan;
pub mod mst;
pub mod hierarchy;
pub mod kmeans;
pub mod knn;
pub mod hierarchical;
pub mod gmm;
pub mod spectral;

pub use distance::{DistanceMetric, Distance};
pub use dbscan::{DBSCAN, DBSCANParams, ClusterResult};
pub use hdbscan::{HDBSCAN, HDBSCANParams, HDBSCANResult};
pub use kmeans::{KMeans, KMeansParams, KMeansResult};
pub use knn::{KNN, KNNParams};
pub use hierarchical::{AgglomerativeClustering, HierarchicalParams, HierarchicalResult, Linkage};
pub use gmm::{GMM, GMMParams, GMMResult, CovarianceType};
pub use spectral::{SpectralClustering, SpectralParams, SpectralResult};

#[cfg(test)]
mod tests;
