@include src/gpu/wgsl/types.wgsl;
@include src/gpu/wgsl/constants.wgsl;
@include src/gpu/wgsl/utils/color.wgsl;

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var<storage, read> points: array<VerletPoint>;

@group(0) @binding(2)
var<storage, read> cells: array<Cell>;

@group(0) @binding(3)
var<storage, read> links: array<Link>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

fn compute_clip_position(world_pos: vec2<f32>) -> vec4<f32> {
    let relative = world_pos - uniforms.camera.xy;
    let view_size_x = uniforms.sim_params.z / uniforms.sim_params.y;
    let view_size_y = uniforms.sim_params.w / uniforms.sim_params.y;
    let clip_x = (relative.x / view_size_x) * 2.0;
    let clip_y = -(relative.y / view_size_y) * 2.0;
    return vec4<f32>(clip_x, clip_y, 0.0, 1.0);
}

fn empty_vertex() -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    return out;
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var output: VertexOutput;
    if instance_index >= arrayLength(&links) {
        return empty_vertex();
    }

    let link = links[instance_index];
    if (link.flags & LINK_FLAG_ACTIVE) == 0u {
        return empty_vertex();
    }

    if link.a_cell >= arrayLength(&cells) || link.b_cell >= arrayLength(&cells) {
        return empty_vertex();
    }

    let cell_a = cells[link.a_cell];
    let cell_b = cells[link.b_cell];
    if (cell_a.flags & CELL_FLAG_ACTIVE) == 0u || (cell_b.flags & CELL_FLAG_ACTIVE) == 0u {
        return empty_vertex();
    }

    if cell_a.generation != link.a_generation || cell_b.generation != link.b_generation {
        return empty_vertex();
    }

    if cell_a.point_idx >= arrayLength(&points) || cell_b.point_idx >= arrayLength(&points) {
        return empty_vertex();
    }

    let point_a = points[cell_a.point_idx];
    let point_b = points[cell_b.point_idx];

    let angle_a = point_a.angle + link.angle_from_a;
    let angle_b = point_b.angle + link.angle_from_b;
    let attach_a = point_a.pos + vec2<f32>(cos(angle_a), sin(angle_a)) * point_a.radius;
    let attach_b = point_b.pos + vec2<f32>(cos(angle_b), sin(angle_b)) * point_a.radius;

    let delta = attach_b - attach_a;
    let dist = length(delta);
    if dist <= 0.0001 {
        return empty_vertex();
    }

    let angle_to_b = atan2(delta.y, delta.x) + M_PI/2.0;
    let offset = vec2<f32>(cos(angle_to_b), sin(angle_to_b)) * 0.1;

    var world_pos: vec2<f32>;
    switch vertex_index {
        case 0u: { world_pos = attach_a + offset; }
        case 1u: { world_pos = attach_a - offset; }
        case 2u: { world_pos = attach_b + offset; }
        default: { world_pos = attach_b - offset; }
    }

    output.clip_position = compute_clip_position(world_pos);
    output.color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    return output;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if in.color.a <= 0.0 {
        discard;
    }
    return saturate(brighten(in.color, 3), 1.0);
}
