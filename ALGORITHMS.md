# DBSCAN and HDBSCAN Algorithms

This document provides a detailed explanation of the DBSCAN and HDBSCAN clustering algorithms implemented in this library.

## DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

### Overview

DBSCAN is a density-based clustering algorithm that groups together points that are closely packed together, while marking points in low-density regions as outliers (noise).

### Key Concepts

#### 1. Core Points
A point `p` is a **core point** if at least `min_samples` points are within distance `eps` of it (including `p` itself).

#### 2. Border Points
A point `q` is a **border point** if it is within distance `eps` of a core point but has fewer than `min_samples` points within `eps` of itself.

#### 3. Noise Points
Points that are neither core nor border points are classified as **noise**.

### Algorithm Steps

1. **Neighbor Discovery**: For each point, find all points within distance `eps`
2. **Core Point Identification**: Mark points with at least `min_samples` neighbors as core points
3. **Cluster Formation**: 
   - Start with an unvisited core point
   - Create a new cluster
   - Add all density-reachable points to the cluster (BFS/DFS)
   - Repeat until all core points are visited
4. **Border Point Assignment**: Assign border points to nearby clusters
5. **Noise Labeling**: Mark remaining points as noise (-1)

### Parameters

- **eps (ε)**: The maximum distance between two samples for one to be considered in the neighborhood of the other
  - Too small: Most points become noise
  - Too large: All points merge into one cluster
  
- **min_samples**: The minimum number of samples in a neighborhood for a point to be considered a core point
  - Larger values: More robust to noise, but may miss smaller clusters
  - Smaller values: More sensitive, may create spurious clusters

### Complexity

- **Time**: O(n log n) with spatial indexing (KD-tree), O(n²) without
- **Space**: O(n)

---

## HDBSCAN (Hierarchical Density-Based Spatial Clustering)

### Overview

HDBSCAN extends DBSCAN by:
1. Building a hierarchy of clusters at different density levels
2. Extracting the most stable clusters from the hierarchy
3. Providing cluster membership probabilities

### Key Concepts

#### 1. Core Distance
The core distance of a point `p` is the distance to its k-th nearest neighbor, where k = `min_samples`.

```
core_distance(p) = distance(p, kNN(p, min_samples))
```

#### 2. Mutual Reachability Distance
The mutual reachability distance between points `a` and `b` is:

```
d_mreach(a, b) = max(core_distance(a), core_distance(b), d(a, b))
```

This transforms the space to emphasize density.

#### 3. Minimum Spanning Tree (MST)
An MST is constructed using mutual reachability distances. This captures the cluster hierarchy.

#### 4. Cluster Hierarchy
The MST is converted into a dendrogram by processing edges in order of increasing weight:
- Each point starts as its own cluster
- Clusters are merged as we traverse the MST
- This creates a hierarchy of clusters at different density levels

#### 5. Cluster Stability
For each cluster in the hierarchy, we compute a stability score:

```
stability(C) = Σ (λ_death - λ_birth) for all points in C
```

where λ = 1/distance (larger λ = denser)

#### 6. Cluster Extraction
We select clusters that maximize overall stability:
- Start from leaves of the hierarchy
- For each node, compare its stability to the sum of its children's stabilities
- Select the node if it's more stable, otherwise select children

### Algorithm Steps

1. **Compute Core Distances**: Find k-th nearest neighbor distance for each point
2. **Build Mutual Reachability Graph**: Compute mutual reachability distances
3. **Construct MST**: Use Prim's algorithm on the mutual reachability graph
4. **Build Cluster Hierarchy**: Convert MST to dendrogram
5. **Compute Stability**: Calculate stability for each cluster
6. **Extract Clusters**: Select most stable clusters
7. **Assign Labels**: Label points based on selected clusters

### Parameters

- **min_cluster_size**: The minimum number of samples in a cluster
  - Larger values: Fewer, larger clusters
  - Smaller values: More, smaller clusters
  
- **min_samples**: The number of samples in a neighborhood for core distance
  - Controls how conservative the clustering is
  - Defaults to `min_cluster_size` if not specified

### Advantages over DBSCAN

1. **No eps parameter**: Automatically finds clusters at varying densities
2. **Hierarchical**: Provides a hierarchy of clusters
3. **Soft clustering**: Provides membership probabilities
4. **Outlier scores**: Quantifies how much a point is an outlier

### Complexity

- **Time**: O(n² log n) for distance matrix, O(n² log n) for MST
- **Space**: O(n²) for distance matrix

### Optimizations

Our implementation includes:
- **Parallel core distance computation**: Using Rayon
- **Efficient MST construction**: Prim's algorithm with priority queue
- **Sparse distance matrix**: Only compute needed distances

---

## Comparison

| Feature | DBSCAN | HDBSCAN |
|---------|--------|---------|
| Parameters | eps, min_samples | min_cluster_size, min_samples |
| Varying density | ❌ | ✅ |
| Hierarchical | ❌ | ✅ |
| Probabilities | ❌ | ✅ |
| Speed | Faster | Slower |
| Memory | Lower | Higher |

## When to Use Which

### Use DBSCAN when:
- Clusters have similar density
- You need fast results
- Memory is limited
- You can tune eps parameter

### Use HDBSCAN when:
- Clusters have varying densities
- You want hierarchical structure
- You need membership probabilities
- You want to avoid tuning eps

## Mathematical Foundations

### Density-Reachability

A point `q` is **directly density-reachable** from `p` if:
1. `p` is a core point
2. `q` is in the eps-neighborhood of `p`

A point `q` is **density-reachable** from `p` if there exists a chain of points `p₁, ..., pₙ` where:
- `p₁ = p` and `pₙ = q`
- Each `pᵢ₊₁` is directly density-reachable from `pᵢ`

### Density-Connectivity

Points `p` and `q` are **density-connected** if there exists a point `o` such that both `p` and `q` are density-reachable from `o`.

A cluster is a maximal set of density-connected points.

## References

1. Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In Kdd (Vol. 96, No. 34, pp. 226-231).

2. Campello, R. J., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. In Pacific-Asia conference on knowledge discovery and data mining (pp. 160-172). Springer, Berlin, Heidelberg.

3. McInnes, L., Healy, J., & Astels, S. (2017). hdbscan: Hierarchical density based clustering. Journal of Open Source Software, 2(11), 205.
