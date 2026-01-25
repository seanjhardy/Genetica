import package::types::{Uniforms, Point, Cell, Link};
import package::constants::{LINK_FLAG_ACTIVE, CELL_FLAG_ACTIVE};
import package::utils::color::{alpha, brighten};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var<storage, read> points: array<Point>;

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
    let attach_a = point_a.pos + vec2<f32>(cos(angle_a), sin(angle_a)) * point_a.radius * 0.7;
    let attach_b = point_b.pos + vec2<f32>(cos(angle_b), sin(angle_b)) * point_b.radius * 0.7;

    let delta_a = attach_a - point_a.pos;
    let delta_b = attach_b - point_b.pos;
    let dist_a = length(delta_a);
    let dist_b = length(delta_b);
    if dist_a <= 0.0001 || dist_b <= 0.0001 {
        return empty_vertex();
    }

    let dir_a = delta_a / dist_a;
    let dir_b = delta_b / dist_b;
    let offset_a = vec2<f32>(-dir_a.y, dir_a.x) * (0.05 * point_a.radius);
    let offset_b = vec2<f32>(-dir_b.y, dir_b.x) * (0.05 * point_b.radius);

    var world_pos: vec2<f32>;
    switch vertex_index {
        // Segment A center -> hook
        case 0u: { world_pos = point_a.pos + offset_a; }
        case 1u: { world_pos = point_a.pos - offset_a; }
        case 2u: { world_pos = attach_a + offset_a; }
        case 3u: { world_pos = attach_a - offset_a; }
        // Degenerate bridge between strips
        case 4u: { world_pos = attach_a - offset_a; }
        case 5u: { world_pos = point_b.pos + offset_b; }
        case 6u: { world_pos = point_b.pos + offset_b; }
        // Segment B center -> hook
        case 7u: { world_pos = point_b.pos - offset_b; }
        case 8u: { world_pos = attach_b + offset_b; }
        default: { world_pos = attach_b - offset_b; }
    }

    output.clip_position = compute_clip_position(world_pos);
    output.color = cell_a.color;
    return output;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if in.color.a <= 0.0 {
        discard;
    }
    return alpha(brighten(in.color, 2.0), 0.2);
    //return vec4<f32>(1.0, 0.0, 0.0, 0.0);
}
