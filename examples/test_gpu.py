"""
Test script for GPU-accelerated DBSCAN and HDBSCAN.

This script tests the GPU implementations and compares performance with CPU versions.
"""

import numpy as np
import time
from ghdbscan import DBSCAN, HDBSCAN, DBSCAN_GPU, HDBSCAN_GPU

def test_gpu_availability():
    """Test if GPU is available."""
    print("=" * 60)
    print("GPU Availability Test")
    print("=" * 60)
    
    dbscan_available = DBSCAN_GPU.is_available()
    hdbscan_available = HDBSCAN_GPU.is_available()
    
    print(f"DBSCAN_GPU available: {dbscan_available}")
    print(f"HDBSCAN_GPU available: {hdbscan_available}")
    print()
    
    return dbscan_available and hdbscan_available

def test_dbscan_small():
    """Test DBSCAN on small dataset."""
    print("=" * 60)
    print("DBSCAN Small Dataset Test")
    print("=" * 60)
    
    # Create small dataset
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(20, 2) + [0, 0],
        np.random.randn(20, 2) + [10, 10],
        np.random.randn(5, 2) + [50, 50],  # Noise
    ])
    
    print(f"Dataset shape: {X.shape}")
    
    # CPU version
    print("\nCPU DBSCAN:")
    start = time.time()
    dbscan_cpu = DBSCAN(eps=2.0, min_samples=3)
    labels_cpu = dbscan_cpu.fit_predict(X)
    cpu_time = time.time() - start
    print(f"  Time: {cpu_time:.4f}s")
    print(f"  Clusters found: {len(set(labels_cpu)) - (1 if -1 in labels_cpu else 0)}")
    print(f"  Noise points: {sum(labels_cpu == -1)}")
    
    # GPU version
    print("\nGPU DBSCAN:")
    start = time.time()
    dbscan_gpu = DBSCAN_GPU(eps=2.0, min_samples=3)
    labels_gpu = dbscan_gpu.fit_predict(X)
    gpu_time = time.time() - start
    print(f"  Time: {gpu_time:.4f}s")
    print(f"  Clusters found: {len(set(labels_gpu)) - (1 if -1 in labels_gpu else 0)}")
    print(f"  Noise points: {sum(labels_gpu == -1)}")
    
    # Compare results
    print(f"\nResults match: {np.array_equal(labels_cpu, labels_gpu)}")
    print()

def test_dbscan_large():
    """Test DBSCAN on large dataset."""
    print("=" * 60)
    print("DBSCAN Large Dataset Test (Performance)")
    print("=" * 60)
    
    # Create large dataset
    np.random.seed(42)
    n_points = 5000
    X = np.vstack([
        np.random.randn(n_points // 2, 2) + [0, 0],
        np.random.randn(n_points // 2, 2) + [10, 10],
    ])
    
    print(f"Dataset shape: {X.shape}")
    
    # CPU version
    print("\nCPU DBSCAN:")
    start = time.time()
    dbscan_cpu = DBSCAN(eps=1.5, min_samples=5)
    labels_cpu = dbscan_cpu.fit_predict(X)
    cpu_time = time.time() - start
    print(f"  Time: {cpu_time:.4f}s")
    print(f"  Clusters found: {len(set(labels_cpu)) - (1 if -1 in labels_cpu else 0)}")
    
    # GPU version
    print("\nGPU DBSCAN:")
    start = time.time()
    dbscan_gpu = DBSCAN_GPU(eps=1.5, min_samples=5)
    labels_gpu = dbscan_gpu.fit_predict(X)
    gpu_time = time.time() - start
    print(f"  Time: {gpu_time:.4f}s")
    print(f"  Clusters found: {len(set(labels_gpu)) - (1 if -1 in labels_gpu else 0)}")
    
    # Performance comparison
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Results match: {np.array_equal(labels_cpu, labels_gpu)}")
    print()

def test_hdbscan_small():
    """Test HDBSCAN on small dataset."""
    print("=" * 60)
    print("HDBSCAN Small Dataset Test")
    print("=" * 60)
    
    # Create small dataset with varying densities
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(15, 2) * 0.5 + [0, 0],  # Dense cluster
        np.random.randn(15, 2) * 1.5 + [8, 8],  # Sparse cluster
        np.random.randn(5, 2) + [50, 50],       # Noise
    ])
    
    print(f"Dataset shape: {X.shape}")
    
    # CPU version
    print("\nCPU HDBSCAN:")
    start = time.time()
    hdbscan_cpu = HDBSCAN(min_cluster_size=5)
    labels_cpu = hdbscan_cpu.fit_predict(X)
    cpu_time = time.time() - start
    print(f"  Time: {cpu_time:.4f}s")
    print(f"  Clusters found: {len(set(labels_cpu)) - (1 if -1 in labels_cpu else 0)}")
    print(f"  Noise points: {sum(labels_cpu == -1)}")
    
    # GPU version
    print("\nGPU HDBSCAN:")
    start = time.time()
    hdbscan_gpu = HDBSCAN_GPU(min_cluster_size=5)
    labels_gpu = hdbscan_gpu.fit_predict(X)
    gpu_time = time.time() - start
    print(f"  Time: {gpu_time:.4f}s")
    print(f"  Clusters found: {len(set(labels_gpu)) - (1 if -1 in labels_gpu else 0)}")
    print(f"  Noise points: {sum(labels_gpu == -1)}")
    
    print()

def test_hdbscan_large():
    """Test HDBSCAN on large dataset."""
    print("=" * 60)
    print("HDBSCAN Large Dataset Test (Performance)")
    print("=" * 60)
    
    # Create large dataset
    np.random.seed(42)
    n_points = 3000
    X = np.vstack([
        np.random.randn(n_points // 3, 2) * 0.8 + [0, 0],
        np.random.randn(n_points // 3, 2) * 1.2 + [10, 10],
        np.random.randn(n_points // 3, 2) * 1.5 + [20, 0],
    ])
    
    print(f"Dataset shape: {X.shape}")
    
    # CPU version
    print("\nCPU HDBSCAN:")
    start = time.time()
    hdbscan_cpu = HDBSCAN(min_cluster_size=50)
    labels_cpu = hdbscan_cpu.fit_predict(X)
    cpu_time = time.time() - start
    print(f"  Time: {cpu_time:.4f}s")
    print(f"  Clusters found: {len(set(labels_cpu)) - (1 if -1 in labels_cpu else 0)}")
    
    # GPU version
    print("\nGPU HDBSCAN:")
    start = time.time()
    hdbscan_gpu = HDBSCAN_GPU(min_cluster_size=50)
    labels_gpu = hdbscan_gpu.fit_predict(X)
    gpu_time = time.time() - start
    print(f"  Time: {gpu_time:.4f}s")
    print(f"  Clusters found: {len(set(labels_gpu)) - (1 if -1 in labels_gpu else 0)}")
    
    # Performance comparison
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    print(f"\nSpeedup: {speedup:.2f}x")
    print()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GPU-Accelerated Clustering Test Suite")
    print("=" * 60 + "\n")
    
    # Check GPU availability
    gpu_available = test_gpu_availability()
    
    if not gpu_available:
        print("⚠️  GPU not available. Tests will still run but may fall back to CPU.")
        print()
    
    # Run tests
    try:
        test_dbscan_small()
        test_dbscan_large()
        test_hdbscan_small()
        test_hdbscan_large()
        
        print("=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
