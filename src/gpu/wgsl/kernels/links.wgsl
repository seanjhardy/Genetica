@include src/gpu/wgsl/constants.wgsl;
@include src/gpu/wgsl/types.wgsl;

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var<storage, read_write> points: array<Point>;

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

const HOOK_RADIUS_SCALE: f32 = 0.7;
const MIN_LINK_ANGLE: f32 = M_PI * 0.95;

fn rotate_vec(v: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec2<f32>(v.x * c - v.y * s, v.x * s + v.y * c);
}

fn get_attachment_point(point: Point, angle_on_point: f32) -> vec2<f32> {
    let angle = point.angle + angle_on_point;
    return point.pos + vec2<f32>(cos(angle), sin(angle)) * point.radius * HOOK_RADIUS_SCALE;
}

fn project_center_to_hook(center: vec2<f32>, hook: vec2<f32>, radius: f32) -> vec2<f32> {
    let to_hook = hook - center;
    let len = length(to_hook);
    return hook - (to_hook / len) * (radius * HOOK_RADIUS_SCALE);
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

    let base_a = point_a.angle + link.angle_from_a;
    let base_b = point_b.angle + link.angle_from_b;
    let stiffness = clamp(link.stiffness, 0.0, 1.0);

    // Constraint 1: Hook positions coincide (distance = 0).
    let hook_offset_a = hook_offset(base_a, point_a.radius);
    let hook_offset_b = hook_offset(base_b, point_b.radius);
    let hook_a = point_a.pos + hook_offset_a;
    let hook_b = point_b.pos + hook_offset_b;
    let hook_ab = (hook_a + hook_b) * 0.5;

    // Constraint 2: Move centers so hook_ab lies at the correct radius.
    let center_a = project_center_to_hook(point_a.pos, hook_ab, point_a.radius);
    let center_b = project_center_to_hook(point_b.pos, hook_ab, point_b.radius);

    // Hook after center adjustments.
    let hook_ab2 = (center_a + center_b) * 0.5;

    // Constraint 3: Distance between centers (stiffness-scaled).
    let ab = center_b - center_a;
    let d = length(ab);
    if d < 0.0001 {
        return;
    }
    let desired = point_a.radius + point_b.radius;
    let diff = d - desired;
    let corr = (ab / d) * (diff * 0.5 * stiffness);
    let center_a_stiff = center_a + corr;
    let center_b_stiff = center_b - corr;

    // Constraint 4: Enforce minimum angle around hook_ab2.
    let vec_a = center_a_stiff - hook_ab2;
    let vec_b = center_b_stiff - hook_ab2;
    let dir_a = normalize(vec_a);
    let dir_b = normalize(vec_b);
    let cos_angle = clamp(dot(dir_a, dir_b), -1.0, 1.0);
    let angle = acos(cos_angle);
    var rot_a = 0.0;
    var rot_b = 0.0;
    if angle < MIN_LINK_ANGLE {
        let delta = (MIN_LINK_ANGLE - angle) * 0.5;
        let cross_z = dir_a.x * dir_b.y - dir_a.y * dir_b.x;
        if cross_z >= 0.0 {
            rot_a = -delta;
            rot_b = delta;
        } else {
            rot_a = delta;
            rot_b = -delta;
        }
    }
    let final_center_a = hook_ab2 + rotate_vec(vec_a, rot_a);
    let final_center_b = hook_ab2 + rotate_vec(vec_b, rot_b);

    // Step 5/6: rotate angles by hook rotation around each center.
    let old_dir_a = atan2(hook_ab.y - point_a.pos.y, hook_ab.x - point_a.pos.x);
    let new_dir_a = atan2(hook_ab2.y - final_center_a.y, hook_ab2.x - final_center_a.x);
    let old_dir_b = atan2(hook_ab.y - point_b.pos.y, hook_ab.x - point_b.pos.x);
    let new_dir_b = atan2(hook_ab2.y - final_center_b.y, hook_ab2.x - final_center_b.x);
    let delta_a = norm_angle(new_dir_a - old_dir_a);
    let delta_b = norm_angle(new_dir_b - old_dir_b);
    let delta_pos_a = (final_center_a - point_a.pos) * 0.5;
    let delta_pos_b = (final_center_b - point_b.pos) * 0.5;

    let base_a_idx = correction_base(cell_a.point_idx);
    let base_b_idx = correction_base(cell_b.point_idx);

    apply_correction(base_a_idx, delta_pos_a, delta_a);
    apply_correction(base_b_idx, delta_pos_b, delta_b);
}
