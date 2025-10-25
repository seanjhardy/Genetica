// WebGPU Compute Shader for 3x3 kernel multiplication
// Multiplies x and y position coordinates by stored constants

struct Constants {
    multiplier_x: f32,
    multiplier_y: f32,
}

struct Result {
    value: f32,
}

@group(0) @binding(0) var<uniform> constants: Constants;
@group(0) @binding(1) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Use global_id.x as the linear index since we're dispatching (9, 1, 1)
    let index = global_id.x;
    
    // Bounds check to prevent out-of-bounds access
    if (index >= 9u) {
        return;
    }
    
    // Calculate x and y positions for 3x3 grid
    let x = f32(index % 3u);
    let y = f32(index / 3u);
    
    // Multiply positions by the constants
    let result_value = x * constants.multiplier_x + y * constants.multiplier_y;
    
    // Store the result
    results[index] = Result(result_value);
}
