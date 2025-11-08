// Compute shader for calculating distances between cells in GRNs

struct GRNData {
    // GPU Buffers for inputs and outputs
    // Previous states hidden nodes are included in input buffer
    in_state: array<f32>,
    h_state: array<atomic<f32>>,
    out_state: array<f32>,
    // GPU Buffers for weight matrices
    w_inh_h: array<f32>,
    w_h_out: array<f32>,
}

@group(0) @binding(0)
var<storage, read> grns: array<GRNData>;

@group(0) @binding(1)
var<storage, read> input_size: u32;

@group(0) @binding(2)
var<storage, read> hidden_size: u32;

@group(0) @binding(3)
var<storage, read> output_size: u32;


@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let idy = global_id.y;
    let idz = global_id.z;
    
    let grn = grns[idx];
    // Input * weight
    let value = grn.in_state[idy] * grn.w_inh_h[idy * input_size + idz];

    atomicStore(grn.h_state[idz], value);
}

