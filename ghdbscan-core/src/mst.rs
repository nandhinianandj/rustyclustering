//! Minimum spanning tree construction and utilities.

use ordered_float::OrderedFloat;
use std::collections::BinaryHeap;
use std::cmp::Reverse;

/// An edge in the minimum spanning tree.
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    /// Source vertex index.
    pub from: usize,
    /// Destination vertex index.
    pub to: usize,
    /// Edge weight (distance).
    pub weight: f64,
}

impl Edge {
    pub fn new(from: usize, to: usize, weight: f64) -> Self {
        Self { from, to, weight }
    }
}

/// Minimum spanning tree representation.
#[derive(Debug, Clone)]
pub struct MST {
    /// Edges in the MST, sorted by weight.
    pub edges: Vec<Edge>,
    /// Number of vertices.
    pub n_vertices: usize,
}

impl MST {
    /// Construct a minimum spanning tree using Prim's algorithm.
    ///
    /// # Arguments
    ///
    /// * `distances` - Distance matrix (n x n)
    ///
    /// # Returns
    ///
    /// Minimum spanning tree
    pub fn from_distance_matrix(distances: &[Vec<f64>]) -> Self {
        let n = distances.len();
        if n == 0 {
            return Self {
                edges: Vec::new(),
                n_vertices: 0,
            };
        }

        let mut edges = Vec::with_capacity(n - 1);
        let mut in_mst = vec![false; n];
        let mut min_heap: BinaryHeap<Reverse<(OrderedFloat<f64>, usize, usize)>> = BinaryHeap::new();

        // Start from vertex 0
        in_mst[0] = true;
        for j in 1..n {
            if distances[0][j] < f64::INFINITY {
                min_heap.push(Reverse((OrderedFloat(distances[0][j]), 0, j)));
            }
        }

        // Prim's algorithm
        while let Some(Reverse((OrderedFloat(weight), from, to))) = min_heap.pop() {
            if in_mst[to] {
                continue;
            }

            // Add edge to MST
            edges.push(Edge::new(from, to, weight));
            in_mst[to] = true;

            // Add new edges from the newly added vertex
            for j in 0..n {
                if !in_mst[j] && distances[to][j] < f64::INFINITY {
                    min_heap.push(Reverse((OrderedFloat(distances[to][j]), to, j)));
                }
            }
        }

        // Sort edges by weight for HDBSCAN
        edges.sort_by_key(|e| OrderedFloat(e.weight));

        Self {
            edges,
            n_vertices: n,
        }
    }

    /// Get the maximum edge weight in the MST.
    pub fn max_edge_weight(&self) -> f64 {
        self.edges
            .iter()
            .map(|e| e.weight)
            .max_by_key(|&w| OrderedFloat(w))
            .unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mst_simple() {
        // Simple triangle graph
        let distances = vec![
            vec![0.0, 1.0, 3.0],
            vec![1.0, 0.0, 2.0],
            vec![3.0, 2.0, 0.0],
        ];

        let mst = MST::from_distance_matrix(&distances);
        
        assert_eq!(mst.edges.len(), 2);
        assert_eq!(mst.n_vertices, 3);
        
        // MST should contain edges with weights 1.0 and 2.0
        let total_weight: f64 = mst.edges.iter().map(|e| e.weight).sum();
        assert!((total_weight - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mst_empty() {
        let distances: Vec<Vec<f64>> = vec![];
        let mst = MST::from_distance_matrix(&distances);
        
        assert_eq!(mst.edges.len(), 0);
        assert_eq!(mst.n_vertices, 0);
    }
}
