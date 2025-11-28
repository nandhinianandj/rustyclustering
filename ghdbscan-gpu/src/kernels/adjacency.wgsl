// Fused Adjacency Kernel for DBSCAN
// Computes pairwise distances and populates a bit-packed adjacency matrix in one pass.
// Each thread computes 32 neighbor relationships (one u32 word) to optimize memory access.

struct Params {
    n_points: u32,
    n_features: u32,
    eps: f32,
}

@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read_write> adjacency: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Each thread handles one row (point i) and a chunk of 32 columns (points j to j+31)
    // Actually, to keep it simple and parallel:
    // Let's have each thread handle one (i, j_block) pair where j_block is a block of 32 points.
    
    // Global ID x maps to the linear index of the (i, j_block) pair
    // We need to decode this into i and j_block
    
    let n_points = params.n_points;
    let n_blocks = (n_points + 31u) / 32u;
    
    let idx = global_id.x;
    let total_blocks = n_points * n_blocks;
    
    if (idx >= total_blocks) {
        return;
    }
    
    let i = idx / n_blocks;
    let j_block = idx % n_blocks;
    let j_start = j_block * 32u;
    
    var bitmask: u32 = 0u;
    
    // Iterate over the 32 points in this block
    for (var bit: u32 = 0u; bit < 32u; bit = bit + 1u) {
        let j = j_start + bit;
        
        if (j >= n_points) {
            continue;
        }
        
        // Compute distance between point i and point j
        var dist_sq: f32 = 0.0;
        for (var k: u32 = 0u; k < params.n_features; k = k + 1u) {
            let val_i = data[i * params.n_features + k];
            let val_j = data[j * params.n_features + k];
            let diff = val_i - val_j;
            dist_sq = dist_sq + diff * diff;
        }
        
        // Check if neighbor (avoid sqrt for performance)
        if (dist_sq <= params.eps * params.eps) {
            bitmask = bitmask | (1u << bit);
        }
    }
    
    // Write the bitmask to the adjacency matrix
    adjacency[idx] = bitmask;
}
