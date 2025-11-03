// Compute shader for Verlet integration

struct Point {
    pos: vec2<f32>,
    prev_pos: vec2<f32>,
    velocity: vec2<f32>,
}

// Uniforms struct must match Rust struct layout exactly (including padding)
// In WGSL uniform buffers, vec3<f32> takes 16 bytes for alignment (padded to vec4)
struct Uniforms {
    delta_time: f32, // 4 bytes at offset 0
    _padding1: f32, // 4 bytes at offset 4 - pad to align vec2 (not needed, but for consistency)
    _padding2: f32, // 4 bytes at offset 8
    _padding3: f32, // 4 bytes at offset 12 (totals 16 bytes to align vec2)
    camera_pos: vec2<f32>, // 8 bytes at offset 16
    zoom: f32, // 4 bytes at offset 24
    point_radius: f32, // 4 bytes at offset 28
    bounds: vec4<f32>, // 16 bytes at offset 32
    view_size: vec2<f32>, // 8 bytes at offset 48
    _padding5: vec2<f32>, // 8 bytes at offset 56 to reach 64 bytes
    _final_padding: vec4<f32>, // 16 bytes at offset 64 to reach 80 bytes
}

@group(0) @binding(0)
var<storage, read_write> points: array<Point>;

@group(0) @binding(1)
var<uniform> uniforms: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if index >= arrayLength(&points) {
        return;
    }
    
    var point = points[index];
    
    // Verlet integration
    let dt = uniforms.delta_time;
    let velocity = point.pos - point.prev_pos;
    
    // Simple gravity and damping
    let gravity = vec2<f32>(0.0, 98.0);
    let damping = 0.99;
    
    let new_pos = point.pos + velocity * damping + gravity * dt * dt;
    
    point.prev_pos = point.pos;
    point.pos = new_pos;
    point.velocity = velocity / dt;
    
    // Boundary constraints
    let radius = uniforms.point_radius;
    let min_x = uniforms.bounds.x + radius;
    let max_x = uniforms.bounds.z - radius;
    let min_y = uniforms.bounds.y + radius;
    let max_y = uniforms.bounds.w - radius;
    
    if point.pos.x < min_x {
        point.prev_pos.x = point.pos.x;
        point.pos.x = min_x;
    } else if point.pos.x > max_x {
        point.prev_pos.x = point.pos.x;
        point.pos.x = max_x;
    }
    
    if point.pos.y < min_y {
        point.prev_pos.y = point.pos.y;
        point.pos.y = min_y;
    } else if point.pos.y > max_y {
        point.prev_pos.y = point.pos.y;
        point.pos.y = max_y;
    }
    
    points[index] = point;
}


