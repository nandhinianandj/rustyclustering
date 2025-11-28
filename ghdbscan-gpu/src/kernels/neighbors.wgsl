// Neighbor finding kernel for DBSCAN
// Finds all neighbors within epsilon distance for each point

struct Params {
    n_points: u32,
    eps: f32,
}

@group(0) @binding(0) var<storage, read> distances: array<f32>;
@group(0) @binding(1) var<storage, read_write> neighbor_counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> neighbors: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

// Maximum neighbors per point (will be configurable)
const MAX_NEIGHBORS: u32 = 256u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    
    if (i >= params.n_points) {
        return;
    }
    
    var count: u32 = 0u;
    let base_idx = i * MAX_NEIGHBORS;
    
    // Find all neighbors within eps
    for (var j: u32 = 0u; j < params.n_points; j = j + 1u) {
        let dist_idx = i * params.n_points + j;
        let dist = distances[dist_idx];
        
        if (dist <= params.eps) {
            if (count < MAX_NEIGHBORS) {
                neighbors[base_idx + count] = j;
                count = count + 1u;
            }
        }
    }
    
    neighbor_counts[i] = count;
}
