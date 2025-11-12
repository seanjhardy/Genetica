// Compute shader for link constraints and lifecycle (placeholder implementation)
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
}

struct Link {
    a: u32,
    b: u32,
    flags: u32,
    _padding0: u32,
    rest_length: f32,
    stiffness: f32,
    energy_transfer_rate: f32,
    _padding1: f32,
}

struct Uniforms {
    sim_params: vec4<f32>,
    cell_count: vec4<f32>,
    camera: vec4<f32>,
    bounds: vec4<f32>,
    nutrient: vec4<u32>,
}

struct CellFreeList {
    count: atomic<u32>,
    indices: array<u32>,
}

struct Counter {
    value: atomic<u32>,
}

struct LifeformFlagArray {
    values: array<atomic<u32>>,
}

struct DivisionRequest {
    parent_lifeform_slot: u32,
    cell_index: u32,
    pos: vec2<f32>,
    radius: f32,
    energy: f32,
}

struct NutrientGrid {
    values: array<atomic<u32>>,
}

struct CellEvent {
    kind: u32,
    parent_cell_index: u32,
    parent_lifeform_slot: u32,
    flags: u32,
    position: vec2<f32>,
    radius: f32,
    energy: f32,
}

struct LinkEvent {
    kind: u32,
    link_index: u32,
    cell_a: u32,
    cell_b: u32,
    rest_length: f32,
    stiffness: f32,
    energy_transfer_rate: f32,
    _padding: f32,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var<storage, read_write> cells: array<Cell>;

@group(0) @binding(2)
var<storage, read_write> cell_free_list: CellFreeList;

@group(0) @binding(3)
var<storage, read_write> alive_counter: Counter;

@group(0) @binding(4)
var<storage, read_write> spawn_count: Counter;

@group(0) @binding(5)
var<storage, read> spawn_requests: array<Cell>;

@group(0) @binding(6)
var<storage, read_write> lifeform_active: LifeformFlagArray;

@group(0) @binding(7)
var<storage, read_write> division_request_count: Counter;

@group(0) @binding(8)
var<storage, read_write> division_requests: array<DivisionRequest>;

@group(0) @binding(9)
var<storage, read_write> nutrient_grid: NutrientGrid;

@group(0) @binding(10)
var<storage, read_write> links: array<Link>;

@group(0) @binding(11)
var<storage, read_write> link_free_count: Counter;

@group(0) @binding(12)
var<storage, read_write> link_free_list: array<u32>;

@group(0) @binding(13)
var<storage, read_write> link_event_count: Counter;

@group(0) @binding(14)
var<storage, read_write> link_events: array<LinkEvent>;

@group(0) @binding(15)
var<storage, read_write> cell_event_count: Counter;

@group(0) @binding(16)
var<storage, read_write> cell_events: array<CellEvent>;

const LINK_FLAG_ALIVE: u32 = 1u;
const LINK_FLAG_ADHESIVE: u32 = 1u << 1u;

fn release_link(index: u32) {
    links[index].flags = 0u;
    let free_index = atomicAdd(&link_free_count.value, 1u);
    if free_index < arrayLength(&link_free_list) {
        link_free_list[free_index] = index;
    } else {
        atomicSub(&link_free_count.value, 1u);
    }
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if index >= arrayLength(&links) {
        return;
    }

    let link = links[index];
    if (link.flags & LINK_FLAG_ALIVE) == 0u {
        return;
    }

    if link.a >= arrayLength(&cells) || link.b >= arrayLength(&cells) {
        release_link(index);
        return;
    }

    var cell_a = cells[link.a];
    var cell_b = cells[link.b];

    if cell_a.is_alive == 0u || cell_b.is_alive == 0u {
        release_link(index);
        return;
    }

    let delta = cell_b.pos - cell_a.pos;
    let dist_sq = dot(delta, delta);
    if dist_sq == 0.0 {
        return;
    }

    let dist = sqrt(dist_sq);
    let rest_length = link.rest_length;
    if rest_length == 0.0 {
        return;
    }

    let stiffness = link.stiffness;
    let diff = dist - rest_length;
    let correction = (diff / dist) * 0.5 * stiffness;
    let adjustment = delta * correction;

    cell_a.prev_pos = cell_a.pos;
    cell_b.prev_pos = cell_b.pos;

    cell_a.pos += adjustment;
    cell_b.pos -= adjustment;

    cells[link.a] = cell_a;
    cells[link.b] = cell_b;
    cells[link.a].energy += 0.2;
    cells[link.b].energy += 0.2;
}

