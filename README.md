# ghdbscan

High-performance DBSCAN and HDBSCAN clustering algorithms implemented in Rust with Python bindings.

## Features

- 🚀 **Fast**: Implemented in Rust with efficient spatial indexing (KD-trees)
- 🔄 **Parallel**: Leverages multi-threading for improved performance
- 🐍 **Python-friendly**: Easy-to-use Python API via PyO3
- 📊 **Complete**: Both DBSCAN and HDBSCAN algorithms
- 🎯 **Flexible**: Multiple distance metrics (Euclidean, Manhattan, Cosine)

## Algorithms

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

DBSCAN groups together points that are closely packed together, marking points in low-density regions as outliers.

**Parameters:**
- `eps`: Maximum distance between two points to be considered neighbors
- `min_samples`: Minimum number of points required to form a dense region

### HDBSCAN (Hierarchical DBSCAN)

HDBSCAN extends DBSCAN by building a hierarchy of clusters at different density levels and extracting the most stable clusters.

**Parameters:**
- `min_cluster_size`: Minimum size of clusters
- `min_samples`: Minimum samples for core distance calculation (optional)

## Installation

### Python Package

```bash
# Install from source (requires Rust toolchain)
pip install maturin
cd ghdbscan
maturin develop --release

# Or for development
maturin develop
```

### Rust Library

Add to your `Cargo.toml`:

```toml
[dependencies]
ghdbscan-core = { path = "path/to/ghdbscan/ghdbscan-core" }
```

## Usage

### Python

```python
import numpy as np
from ghdbscan import DBSCAN, HDBSCAN

# Generate sample data
X = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])

# DBSCAN
dbscan = DBSCAN(eps=2.0, min_samples=2)
labels = dbscan.fit_predict(X)
print(f"DBSCAN labels: {labels}")

# HDBSCAN
hdbscan = HDBSCAN(min_cluster_size=2)
labels = hdbscan.fit_predict(X)
print(f"HDBSCAN labels: {labels}")
```

### Rust

```rust
use ghdbscan_core::{DBSCAN, DBSCANParams, DistanceMetric};
use ndarray::array;

fn main() {
    let data = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0]];
    
    let params = DBSCANParams::new(2.0, 2)
        .with_metric(DistanceMetric::Euclidean);
    
    let dbscan = DBSCAN::new(params);
    let result = dbscan.fit(&data.view());
    
    println!("Found {} clusters", result.n_clusters());
}
```

## Examples

### Rust Examples

```bash
# Run DBSCAN example
cargo run --example dbscan_example

# Run HDBSCAN example
cargo run --example hdbscan_example
```

### Python Examples

```bash
# DBSCAN with visualization
python examples/test_dbscan.py

# HDBSCAN with visualization
python examples/test_hdbscan.py
```

## Performance

The implementation uses:
- **KD-trees** for efficient spatial indexing and neighbor queries
- **Rayon** for parallel processing of independent computations
- **Optimized algorithms** following best practices from the literature

## Project Structure

```
ghdbscan/
├── ghdbscan-core/          # Core Rust implementation
│   ├── src/
│   │   ├── lib.rs          # Library entry point
│   │   ├── distance.rs     # Distance metrics
│   │   ├── spatial_index.rs # KD-tree wrapper
│   │   ├── dbscan.rs       # DBSCAN algorithm
│   │   ├── hdbscan.rs      # HDBSCAN algorithm
│   │   ├── mst.rs          # Minimum spanning tree
│   │   └── hierarchy.rs    # Cluster hierarchy
│   └── examples/           # Rust examples
├── ghdbscan-py/            # Python bindings
│   └── src/
│       └── lib.rs          # PyO3 bindings
├── examples/               # Python examples
│   ├── test_dbscan.py
│   └── test_hdbscan.py
└── pyproject.toml          # Python package config
```

## Testing

```bash
# Run Rust tests
cd ghdbscan-core
cargo test

# Run with output
cargo test -- --nocapture
```

## Distance Metrics

Supported distance metrics:
- **Euclidean**: Standard L2 distance
- **Manhattan**: L1 distance (city block)
- **Cosine**: 1 - cosine similarity

## Limitations

- Currently optimized for 2D data (can be extended to higher dimensions)
- KD-tree performance degrades in very high dimensions (>16)

## References

- Ester, M., et al. (1996). "A density-based algorithm for discovering clusters in large spatial databases with noise."
- Campello, R. J., et al. (2013). "Density-Based Clustering Based on Hierarchical Density Estimates."

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
