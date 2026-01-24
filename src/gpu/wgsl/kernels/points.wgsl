@include src/gpu/wgsl/types.wgsl;
@include src/gpu/wgsl/constants.wgsl;
@include src/gpu/wgsl/utils/color.wgsl;

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> points: array<Point>;
@group(0) @binding(2) var<storage, read_write> link_corrections: array<atomic<i32>>;

const FRICTION: f32 = 0.99;
const VELOCITY_EPSILON: f32 = 0.001;

fn correction_base(point_idx: u32) -> u32 {
    return point_idx * 3u;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let dt = uniforms.sim_params.x;
    
    if idx >= arrayLength(&points) {
        return;
    }

    var point: Point = points[idx];
    let base = correction_base(idx);
    var correction = vec2<f32>(0.0, 0.0);
    var angle_correction = 0.0;
    if base + 2u < arrayLength(&link_corrections) {
        correction = vec2<f32>(
            f32(atomicExchange(&link_corrections[base], 0)),
            f32(atomicExchange(&link_corrections[base + 1u], 0))
        ) / LINK_CORRECTION_SCALE_POS;
        angle_correction = f32(atomicExchange(&link_corrections[base + 2u], 0)) / LINK_CORRECTION_SCALE_ANGLE;
    }

    if (point.flags & POINT_FLAG_ACTIVE) == 0u {
        return;
    }
    
    point.pos = point.pos + correction;
    point.prev_pos = point.prev_pos;// + correction;
    point.angle = point.angle + angle_correction;

    let radius: f32 = point.radius;

    // Clamp to exact bounds from uniforms (accounting for radius)
    let min_bound = uniforms.bounds.xy + vec2<f32>(radius, radius);
    let max_bound = uniforms.bounds.zw - vec2<f32>(radius, radius);

    // Clamp position to bounds - this should keep ALL points within bounds at all times
    point.pos.x = clamp(point.pos.x, min_bound.x, max_bound.x);
    point.pos.y = clamp(point.pos.y, min_bound.y, max_bound.y);

    let pos: vec2<f32> = point.pos;
    var vel: vec2<f32> = point.pos - point.prev_pos;

    // Apply friction
    vel = vel * FRICTION;

    var next_pos: vec2<f32> = point.pos + vel * dt + point.accel * (dt * dt);

    // Clamp the calculated next position to bounds (using same uniform bounds)
    next_pos.x = clamp(next_pos.x, min_bound.x, max_bound.x);
    next_pos.y = clamp(next_pos.y, min_bound.y, max_bound.y);

    point.prev_pos = point.pos;
    point.pos = next_pos;
    point.accel = vec2<f32>(0.0, 0.0); // No acceleration
    
    points[idx] = point; 
}
