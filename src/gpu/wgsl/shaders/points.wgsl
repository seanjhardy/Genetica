@include src/gpu/wgsl/types.wgsl;
@include src/gpu/wgsl/constants.wgsl;
@include src/gpu/wgsl/utils/color.wgsl;

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> points: array<Point>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(0.0);
    out.uv = vec2<f32>(0.0);

    if instance_index >= arrayLength(&points) {
        return out;
    }
    let point = points[instance_index];
    if (point.flags & POINT_FLAG_ACTIVE) == 0u {
        return out;
    }

    let pos = point.pos;

    var uv = vec2<f32>(0.0);
    switch vertex_index {
        case 0u: { uv = vec2<f32>(-1.0, -1.0); }
        case 1u: { uv = vec2<f32>(1.0, -1.0); }
        case 2u: { uv = vec2<f32>(-1.0, 1.0); }
        case 3u: { uv = vec2<f32>(1.0, 1.0); }
        case 4u: { uv = vec2<f32>(1.0, -1.0); }
        default: { uv = vec2<f32>(-1.0, 1.0); }
    }

    let world_pos = pos + uv * point.radius;

    let view_w = uniforms.sim_params.z;
    let view_h = uniforms.sim_params.w;
    let zoom = uniforms.sim_params.y;
    let cam_x = uniforms.camera.x;
    let cam_y = uniforms.camera.y;

    let clip_x = ((world_pos.x - cam_x) / (view_w / zoom)) * 2.0;
    let clip_y = -((world_pos.y - cam_y) / (view_h / zoom)) * 2.0;

    out.clip_position = vec4<f32>(clip_x, clip_y, 0.0, 1.0);
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let dist = length(input.uv);
    if (dist > 1.0) {
        discard;
    }
    let t = smoothstep(1.0, 0.6, dist);
    let color = mix(vec3<f32>(0.2, 0.8, 1.0), vec3<f32>(0.05, 0.2, 0.4), t);
    return vec4<f32>(color, 1.0);
}

