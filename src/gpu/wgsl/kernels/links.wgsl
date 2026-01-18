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

@group(0) @binding(13)
var<storage, read_write> link_corrections: array<atomic<i32>>;


fn norm_angle(angle: f32) -> f32 {
    return atan2(sin(angle), cos(angle));
}

fn correction_base(point_idx: u32) -> u32 {
    return point_idx * 3u;
}

fn apply_correction(base: u32, delta: vec2<f32>, delta_angle: f32) {
    if base + 2u >= arrayLength(&link_corrections) {
        return;
    }
    let dx = i32(delta.x * LINK_CORRECTION_SCALE_POS);
    let dy = i32(delta.y * LINK_CORRECTION_SCALE_POS);
    let da = i32(delta_angle * LINK_CORRECTION_SCALE_ANGLE);
    atomicAdd(&link_corrections[base], dx);
    atomicAdd(&link_corrections[base + 1u], dy);
    atomicAdd(&link_corrections[base + 2u], da);
}

@compute @workgroup_size(1024)
fn accumulate(@builtin(global_invocation_id) global_id: vec3<u32>) {
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
    let base_a = point_a.angle + link.angle_from_a;
    let base_b = point_b.angle + link.angle_from_b;
    let stiffness = clamp(link.stiffness, 0.0, 1.0);

    let delta_a = norm_angle(angle_to_b - base_a) * (stiffness * 0.1);
    let delta_b = norm_angle(angle_to_a - base_b) * (stiffness * 0.1);

    let new_angle_a = point_a.angle + delta_a + link.angle_from_a;
    let new_angle_b = point_b.angle + delta_b + link.angle_from_b;
    let point_pos_a = point_a.pos + vec2<f32>(cos(new_angle_a), sin(new_angle_a)) * point_a.radius * 0.7;
    let point_pos_b = point_b.pos + vec2<f32>(cos(new_angle_b), sin(new_angle_b)) * point_b.radius * 0.7;

    let correction = (point_pos_b - point_pos_a) * 0.1;
    let half_correction = correction * 0.5;

    let base_a_idx = correction_base(cell_a.point_idx);
    let base_b_idx = correction_base(cell_b.point_idx);

    apply_correction(base_a_idx, half_correction, delta_a);
    apply_correction(base_b_idx, -half_correction, delta_b);
}

@compute @workgroup_size(1024)
fn apply(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let point_idx = global_id.x;
    if point_idx >= arrayLength(&points) {
        return;
    }

    let base = correction_base(point_idx);
    if base + 2u >= arrayLength(&link_corrections) {
        return;
    }

    let dx = f32(atomicExchange(&link_corrections[base], 0));
    let dy = f32(atomicExchange(&link_corrections[base + 1u], 0));
    let da = f32(atomicExchange(&link_corrections[base + 2u], 0));

    var point = points[point_idx];
    if (point.flags & POINT_FLAG_ACTIVE) == 0u {
        return;
    }

    let delta = vec2<f32>(
        dx / LINK_CORRECTION_SCALE_POS,
        dy / LINK_CORRECTION_SCALE_POS
    );
    point.pos = point.pos + delta;
    point.prev_pos = point.prev_pos + delta;
    point.angle = point.angle + (da / LINK_CORRECTION_SCALE_ANGLE);

    points[point_idx] = point;
}
