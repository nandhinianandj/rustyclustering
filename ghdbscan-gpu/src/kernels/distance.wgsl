// Euclidean distance computation kernel
// Computes pairwise distances between points

struct Params {
    n_points: u32,
    n_features: u32,
}

@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read_write> distances: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;
    
    if (i >= params.n_points || j >= params.n_points) {
        return;
    }
    
    // Only compute upper triangle (symmetric matrix)
    if (i > j) {
        return;
    }
    
    var dist: f32 = 0.0;
    
    // Compute Euclidean distance
    for (var k: u32 = 0u; k < params.n_features; k = k + 1u) {
        let idx_i = i * params.n_features + k;
        let idx_j = j * params.n_features + k;
        let diff = data[idx_i] - data[idx_j];
        dist = dist + diff * diff;
    }
    
    dist = sqrt(dist);
    
    // Store in both positions for symmetric access
    let idx = i * params.n_points + j;
    distances[idx] = dist;
    
    if (i != j) {
        let idx_sym = j * params.n_points + i;
        distances[idx_sym] = dist;
    }
}
