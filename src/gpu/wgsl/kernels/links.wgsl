@include src/gpu/wgsl/constants.wgsl;
@include src/gpu/wgsl/types.wgsl;

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var<storage, read_write> points: array<VerletPoint>;

@group(0) @binding(2)
var<storage, read_write> physics_free_list: FreeList;

@group(0) @binding(3)
var<storage, read_write> physics_counter: atomic<u32>;

@group(0) @binding(4)
var<storage, read_write> cells: array<Cell>;

@group(0) @binding(5)
var<storage, read_write> cell_free_list: FreeList;

@group(0) @binding(6)
var<storage, read_write> cell_counter: atomic<u32>;

@group(0) @binding(7)
var<storage, read_write> nutrient_grid: NutrientGrid;

@group(0) @binding(8)
var<storage, read_write> links: array<Link>;

@group(0) @binding(9)
var<storage, read_write> link_free_list: FreeList;


fn norm_angle(angle: f32) -> f32 {
    return atan2(sin(angle), cos(angle));
}

@compute @workgroup_size(1024)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if index >= arrayLength(&links) {
        return;
    }

    let link = links[index];
    if (link.flags & LINK_FLAG_ACTIVE) == 0u {
        return;
    }

    if link.a_cell >= arrayLength(&cells) || link.b_cell >= arrayLength(&cells) {
        return;
    }

    if link.a_cell == link.b_cell {
        return;
    }

    let cell_a = cells[link.a_cell];
    let cell_b = cells[link.b_cell];

    if link.a_generation != cell_a.generation || link.b_generation != cell_b.generation {
        return;
    }

    if (cell_a.flags & CELL_FLAG_ACTIVE) == 0u || (cell_b.flags & CELL_FLAG_ACTIVE) == 0u {
        return;
    }

    if cell_a.point_idx >= arrayLength(&points) || cell_b.point_idx >= arrayLength(&points) {
        return;
    }

    var point_a = points[cell_a.point_idx];
    var point_b = points[cell_b.point_idx];

    if (point_a.flags & POINT_FLAG_ACTIVE) == 0u || (point_b.flags & POINT_FLAG_ACTIVE) == 0u {
        return;
    }

    let delta = point_b.pos - point_a.pos;
    let dist = length(delta);
    if dist < 0.0001 {
        return;
    }

    let angle_to_b = atan2(delta.y, delta.x);
    let angle_to_a = angle_to_b + M_PI;
    let max_rotation = 0.5 * M_PI;

    let base_a = point_a.angle + link.angle_from_a;
    let base_b = point_b.angle + link.angle_from_b;
    let stiffness = clamp(link.stiffness, 0.0, 1.0);

    let delta_a = norm_angle(angle_to_b - base_a) * (stiffness * 0.1);
    let delta_b = norm_angle(angle_to_a - base_b) * (stiffness * 0.1);

    point_a.angle = point_a.angle + delta_a;
    point_b.angle = point_b.angle + delta_b;

    let new_angle_a = point_a.angle + link.angle_from_a;
    let new_angle_b = point_b.angle + link.angle_from_b;
    let point_pos_a = point_a.pos + vec2<f32>(cos(new_angle_a), sin(new_angle_a)) * point_a.radius;
    let point_pos_b = point_b.pos + vec2<f32>(cos(new_angle_b), sin(new_angle_b)) * point_b.radius;

    // Calculate desired correction to bring attachment points together
    let target_offset = point_pos_b - point_pos_a;
    var correction = target_offset * 0.1;

    // Apply correction as velocity change (better for Verlet integration)
    point_a.pos = point_a.pos + correction;
    point_a.prev_pos = point_a.prev_pos + correction;
    point_b.pos = point_b.pos - correction;
    point_b.prev_pos = point_b.prev_pos - correction;

    points[cell_a.point_idx] = point_a;
    points[cell_b.point_idx] = point_b;
}
