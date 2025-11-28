// Core distance computation for HDBSCAN
// Computes k-th nearest neighbor distance for each point

struct Params {
    n_points: u32,
    min_samples: u32,
}

@group(0) @binding(0) var<storage, read> distances: array<f32>;
@group(0) @binding(1) var<storage, read_write> core_distances: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// Simple selection to find k-th smallest element
// For production, would use a more efficient algorithm
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    
    if (i >= params.n_points) {
        return;
    }
    
    // Collect distances to all other points
    var dists: array<f32, 1024>;
    var count: u32 = 0u;
    
    for (var j: u32 = 0u; j < params.n_points; j = j + 1u) {
        if (i != j && count < 1024u) {
            let dist_idx = i * params.n_points + j;
            dists[count] = distances[dist_idx];
            count = count + 1u;
        }
    }
    
    // Partial sort to find k-th element (selection algorithm)
    // This is a simple bubble-sort-like approach for finding k-th smallest
    for (var pass: u32 = 0u; pass < params.min_samples; pass = pass + 1u) {
        for (var j: u32 = pass + 1u; j < count; j = j + 1u) {
            if (dists[j] < dists[pass]) {
                let temp = dists[pass];
                dists[pass] = dists[j];
                dists[j] = temp;
            }
        }
    }
    
    // The k-th nearest neighbor distance (0-indexed, so min_samples-1)
    if (params.min_samples > 0u && params.min_samples <= count) {
        core_distances[i] = dists[params.min_samples - 1u];
    } else {
        core_distances[i] = 0.0;
    }
}
