// Render shader for drawing links between cells as trapezoids
const MAX_GRN_RECEPTOR_INPUTS: u32 = 16u;
const MAX_GRN_REGULATORY_UNITS: u32 = 16u;
const MAX_GRN_STATE_SIZE: u32 = MAX_GRN_RECEPTOR_INPUTS + MAX_GRN_REGULATORY_UNITS;

struct Cell {
    pos: vec2<f32>,
    prev_pos: vec2<f32>,
    random_force: vec2<f32>,
    radius: f32,
    energy: f32,
    cell_wall_thickness: f32,
    is_alive: u32,
    lifeform_slot: u32,
    metadata: u32,
    color: vec4<f32>,
}

struct Link {
    a: u32,
    b: u32,
    flags: u32,
    generation_a: u32,
    rest_length: f32,
    stiffness: f32,
    energy_transfer_rate: f32,
    generation_b: u32,
}

struct Uniforms {
    sim_params: vec4<f32>,
    cell_count: vec4<f32>,
    camera: vec4<f32>,
    bounds: vec4<f32>,
    nutrient: vec4<u32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var<storage, read> cells: array<Cell>;

@group(0) @binding(2)
var<storage, read> links: array<Link>;

const LINK_FLAG_ALIVE: u32 = 1u;

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
    return VertexOutput(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0));
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    if instance_index >= arrayLength(&links) {
        return empty_vertex();
    }

    let link = links[instance_index];
    if (link.flags & LINK_FLAG_ALIVE) == 0u {
        return empty_vertex();
    }

    if link.a >= arrayLength(&cells) || link.b >= arrayLength(&cells) {
        return empty_vertex();
    }

    let cell_a = cells[link.a];
    let cell_b = cells[link.b];
    if cell_a.is_alive == 0u || cell_b.is_alive == 0u {
        return empty_vertex();
    }

    let delta = cell_b.pos - cell_a.pos;
    let dist = length(delta);
    if dist <= 0.0001 {
        return empty_vertex();
    }

    let dir = delta / dist;
    let perp = vec2<f32>(-dir.y, dir.x);
    let offset_a = perp * cell_a.radius;
    let offset_b = perp * cell_b.radius;

    var world_pos: vec2<f32>;
    var color: vec4<f32>;

    switch vertex_index {
        case 0u: {
            world_pos = cell_a.pos + offset_a;
            color = cell_a.color;
        }
        case 1u: {
            world_pos = cell_a.pos - offset_a;
            color = cell_a.color;
        }
        case 2u: {
            world_pos = cell_b.pos + offset_b;
            color = cell_b.color;
        }
        default: {
            world_pos = cell_b.pos - offset_b;
            color = cell_b.color;
        }
    }

    var output: VertexOutput;
    output.clip_position = compute_clip_position(world_pos);
    output.color = color;
    return output;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}


