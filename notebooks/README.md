# Running the Demonstration Notebooks

This guide explains how to run the DBSCAN and HDBSCAN demonstration notebooks.

## Prerequisites

The Python package has been built and installed in a virtual environment located at `/home/roci/playspace/ghdbscan/venv`.

## Running the Notebooks

### Option 1: Using Jupyter Notebook

```bash
cd /home/roci/playspace/ghdbscan
source venv/bin/activate
jupyter notebook notebooks/
```

Then open either:
- `dbscan_demo.ipynb` - DBSCAN demonstrations
- `hdbscan_demo.ipynb` - HDBSCAN demonstrations

### Option 2: Using JupyterLab

```bash
cd /home/roci/playspace/ghdbscan
source venv/bin/activate
jupyter lab notebooks/
```

### Option 3: Command Line Execution

To execute all cells in a notebook from the command line:

```bash
cd /home/roci/playspace/ghdbscan
source venv/bin/activate
jupyter nbconvert --to notebook --execute notebooks/dbscan_demo.ipynb --output dbscan_demo_executed.ipynb
jupyter nbconvert --to notebook --execute notebooks/hdbscan_demo.ipynb --output hdbscan_demo_executed.ipynb
```

## Notebook Contents

### DBSCAN Demonstration (`dbscan_demo.ipynb`)

1. **Gaussian Blobs**: Well-separated clusters
2. **Moons Dataset**: Non-convex cluster shapes
3. **Circles Dataset**: Concentric circles
4. **Iris Dataset**: Real-world botanical data
5. **Noisy Data**: Outlier detection
6. **Parameter Sensitivity**: Effect of `eps` parameter

### HDBSCAN Demonstration (`hdbscan_demo.ipynb`)

1. **Varying Density Clusters**: Different density levels
2. **Moons Dataset**: Non-convex shapes
3. **Noisy Blobs**: Robustness to outliers
4. **Iris Dataset**: Real-world data
5. **DBSCAN Comparison**: Side-by-side comparison
6. **Parameter Sensitivity**: Effect of `min_cluster_size`

## Expected Results

Both notebooks will generate:
- Visualizations comparing ground truth vs clustering results
- Performance metrics (number of clusters, noise points, execution time)
- Parameter sensitivity analyses
- Comparative visualizations

## Troubleshooting

If you encounter import errors:
```bash
# Rebuild the package
cd /home/roci/playspace/ghdbscan
source venv/bin/activate
cd ghdbscan-py
maturin develop
cd ..
```

If matplotlib plots don't show:
- Make sure you're running in a graphical environment
- Or use `%matplotlib inline` in Jupyter notebooks (already included)
