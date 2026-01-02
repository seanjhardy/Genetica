@include src/gpu/wgsl/types.wgsl;
@include src/gpu/wgsl/constants.wgsl;
@include src/gpu/wgsl/utils/color.wgsl;

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> points: array<VerletPoint>;

const FRICTION: f32 = 0.99;
const VELOCITY_EPSILON: f32 = 0.001;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let dt = uniforms.sim_params.x;
    
    if idx >= arrayLength(&points) {
        return;
    }
    
    var point: VerletPoint = points[idx];
    let pos: vec2<f32> = point.pos;
    let radius: f32 = point.radius;
    var vel: vec2<f32> = point.pos - point.prev_pos;
    
    // Apply friction
    vel = vel * FRICTION;
    
    var next_pos: vec2<f32> = point.pos + vel * dt + point.accel * (dt * dt);
    
    let min_bound = uniforms.bounds.xy + vec2<f32>(radius, radius);
    let max_bound = uniforms.bounds.zw - vec2<f32>(radius, radius);
    
    if (next_pos.x < min_bound.x) {
        vel.x = abs(vel.x);
        next_pos.x = min_bound.x + vel.x;
    } else if (next_pos.x > max_bound.x) {
        next_pos.x = max_bound.x;
        vel.x = -abs(vel.x);
        next_pos.x = max_bound.x + vel.x;
    }
    
    if (next_pos.y < min_bound.y) {
        vel.y = abs(vel.y);
        next_pos.y = min_bound.y + vel.y;
    } else if (next_pos.y > max_bound.y) {
        next_pos.y = max_bound.y;
        vel.y = -abs(vel.y);
        next_pos.y = max_bound.y + vel.y;
    }
    
    point.prev_pos = point.pos;
    point.pos = next_pos;
    point.accel = vec2<f32>(0.0, 0.0);
    
    points[idx] = point; 
}