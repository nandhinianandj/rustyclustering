//! Cluster hierarchy management for HDBSCAN.

use std::collections::HashMap;

/// A node in the cluster hierarchy tree.
#[derive(Debug, Clone)]
pub struct ClusterNode {
    /// Node identifier.
    pub id: usize,
    /// Parent node ID (None for root).
    pub parent: Option<usize>,
    /// Child node IDs.
    pub children: Vec<usize>,
    /// Lambda value (1/distance) at which this cluster was formed.
    pub lambda_birth: f64,
    /// Lambda value at which this cluster split (if applicable).
    pub lambda_death: Option<f64>,
    /// Points in this cluster (leaf nodes only).
    pub points: Vec<usize>,
    /// Cluster stability score.
    pub stability: f64,
}

impl ClusterNode {
    /// Create a new leaf node.
    pub fn new_leaf(id: usize, point: usize, lambda_birth: f64) -> Self {
        Self {
            id,
            parent: None,
            children: Vec::new(),
            lambda_birth,
            lambda_death: None,
            points: vec![point],
            stability: 0.0,
        }
    }

    /// Create a new internal node.
    pub fn new_internal(id: usize, lambda_birth: f64) -> Self {
        Self {
            id,
            parent: None,
            children: Vec::new(),
            lambda_birth,
            lambda_death: None,
            points: Vec::new(),
            stability: 0.0,
        }
    }

    /// Check if this is a leaf node.
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

/// Cluster hierarchy (dendrogram) for HDBSCAN.
#[derive(Debug, Clone)]
pub struct Hierarchy {
    /// All nodes in the hierarchy.
    pub nodes: HashMap<usize, ClusterNode>,
    /// Root node ID.
    pub root: Option<usize>,
    /// Next available node ID.
    next_id: usize,
}

impl Hierarchy {
    /// Create a new empty hierarchy.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            root: None,
            next_id: 0,
        }
    }

    /// Add a leaf node to the hierarchy.
    pub fn add_leaf(&mut self, point: usize, lambda_birth: f64) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        
        let node = ClusterNode::new_leaf(id, point, lambda_birth);
        self.nodes.insert(id, node);
        id
    }

    /// Add an internal node to the hierarchy.
    pub fn add_internal(&mut self, lambda_birth: f64) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        
        let node = ClusterNode::new_internal(id, lambda_birth);
        self.nodes.insert(id, node);
        id
    }

    /// Merge two nodes into a parent node.
    pub fn merge(&mut self, child1: usize, child2: usize, parent: usize, lambda: f64) {
        // Set death lambda for children
        if let Some(node) = self.nodes.get_mut(&child1) {
            node.lambda_death = Some(lambda);
            node.parent = Some(parent);
        }
        if let Some(node) = self.nodes.get_mut(&child2) {
            node.lambda_death = Some(lambda);
            node.parent = Some(parent);
        }

        // Add children to parent
        if let Some(parent_node) = self.nodes.get_mut(&parent) {
            parent_node.children.push(child1);
            parent_node.children.push(child2);
        }
    }

    /// Set the root of the hierarchy.
    pub fn set_root(&mut self, root: usize) {
        self.root = Some(root);
    }

    /// Compute stability for all nodes.
    pub fn compute_stability(&mut self) {
        for node_id in self.nodes.keys().copied().collect::<Vec<_>>() {
            let stability = self.compute_node_stability(node_id);
            if let Some(node) = self.nodes.get_mut(&node_id) {
                node.stability = stability;
            }
        }
    }

    /// Compute stability for a single node.
    fn compute_node_stability(&self, node_id: usize) -> f64 {
        let node = match self.nodes.get(&node_id) {
            Some(n) => n,
            None => return 0.0,
        };

        let lambda_birth = node.lambda_birth;
        let lambda_death = node.lambda_death.unwrap_or(f64::INFINITY);
        
        // For leaf nodes, stability is based on the point's lifetime
        if node.is_leaf() {
            return (lambda_death - lambda_birth).max(0.0);
        }

        // For internal nodes, sum over all descendant points
        let n_points = self.count_descendant_points(node_id);
        (lambda_death - lambda_birth) * n_points as f64
    }

    /// Count the number of descendant points for a node.
    fn count_descendant_points(&self, node_id: usize) -> usize {
        let node = match self.nodes.get(&node_id) {
            Some(n) => n,
            None => return 0,
        };

        if node.is_leaf() {
            return node.points.len();
        }

        node.children
            .iter()
            .map(|&child| self.count_descendant_points(child))
            .sum()
    }

    /// Extract stable clusters from the hierarchy.
    pub fn extract_clusters(&self, min_cluster_size: usize) -> Vec<usize> {
        let mut selected_clusters = Vec::new();
        
        if let Some(root) = self.root {
            self.select_clusters_recursive(root, min_cluster_size, &mut selected_clusters);
        }
        
        selected_clusters
    }

    /// Recursively select the most stable clusters.
    fn select_clusters_recursive(
        &self,
        node_id: usize,
        min_cluster_size: usize,
        selected: &mut Vec<usize>,
    ) -> f64 {
        let node = match self.nodes.get(&node_id) {
            Some(n) => n,
            None => return 0.0,
        };

        // Check if cluster is large enough
        let n_points = self.count_descendant_points(node_id);
        if n_points < min_cluster_size {
            return 0.0;
        }

        // If leaf or no children, this is a candidate cluster
        if node.children.is_empty() {
            selected.push(node_id);
            return node.stability;
        }

        // Compare stability of this cluster vs. sum of children
        let children_stability: f64 = node
            .children
            .iter()
            .map(|&child| self.select_clusters_recursive(child, min_cluster_size, &mut Vec::new()))
            .sum();

        if node.stability > children_stability {
            // Select this cluster
            selected.push(node_id);
            node.stability
        } else {
            // Select children instead
            for &child in &node.children {
                self.select_clusters_recursive(child, min_cluster_size, selected);
            }
            children_stability
        }
    }
}

impl Default for Hierarchy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchy_creation() {
        let mut hierarchy = Hierarchy::new();
        
        let leaf1 = hierarchy.add_leaf(0, 0.0);
        let leaf2 = hierarchy.add_leaf(1, 0.0);
        let parent = hierarchy.add_internal(0.5);
        
        hierarchy.merge(leaf1, leaf2, parent, 0.5);
        hierarchy.set_root(parent);
        
        assert_eq!(hierarchy.nodes.len(), 3);
        assert_eq!(hierarchy.root, Some(parent));
    }

    #[test]
    fn test_stability_computation() {
        let mut hierarchy = Hierarchy::new();
        
        let leaf1 = hierarchy.add_leaf(0, 0.0);
        let leaf2 = hierarchy.add_leaf(1, 0.0);
        let parent = hierarchy.add_internal(0.5);
        
        hierarchy.merge(leaf1, leaf2, parent, 1.0);
        hierarchy.set_root(parent);
        
        hierarchy.compute_stability();
        
        // Check that stability was computed
        assert!(hierarchy.nodes.get(&parent).unwrap().stability > 0.0);
    }
}
