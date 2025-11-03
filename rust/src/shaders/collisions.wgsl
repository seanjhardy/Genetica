// Compute shader for collision detection and response

struct Point {
    pos: vec2<f32>,
    prev_pos: vec2<f32>,
    velocity: vec2<f32>,
}

struct Uniforms {
    delta_time: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
    camera_pos: vec2<f32>,
    zoom: f32,
    point_radius: f32,
    bounds: vec4<f32>,
    view_size: vec2<f32>,
    _padding5: vec2<f32>,
    _final_padding: vec4<f32>,
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
    
    var point_a = points[index];
    let radius = uniforms.point_radius;
    let min_distance = radius * 2.0;
    
    // Check collisions with all other points
    // Using parallel reduction approach - each point checks against all others
    var force = vec2<f32>(0.0, 0.0);
    
    for (var i: u32 = 0u; i < arrayLength(&points); i++) {
        if i == index {
            continue;
        }
        
        let point_b = points[i];
        let delta = point_a.pos - point_b.pos;
        let distance = length(delta);
        
        if distance < min_distance && distance > 0.001 {
            let overlap = min_distance - distance;
            let normal = normalize(delta);
            let separation_force = overlap * 1000.0; // Separation force strength
            
            force += normal * separation_force;
        }
    }
    
    // Apply collision forces
    if length(force) > 0.0 {
        let dt = 0.016; // Approximate delta time
        point_a.pos += force * dt * dt;
    }
    
    points[index] = point_a;
}

