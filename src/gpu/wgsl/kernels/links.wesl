import package::types::{Uniforms, Point, Cell, NutrientGrid, Link, FreeList};
import package::constants::{
    M_PI,
    LINK_CORRECTION_SCALE_POS,
    LINK_CORRECTION_SCALE_ANGLE,
    LINK_FLAG_ACTIVE,
    CELL_FLAG_ACTIVE,
    POINT_FLAG_ACTIVE,
};
import package::utils::verlet::{distance_constraint, rotate_points_to_angle, min_angle_constraint, signed_angle};

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


const CELL_DISTANCE_SCALE: f32 = 0.7;
const MIN_LINK_ANGLE: f32 = M_PI * 0.95;

fn get_attachment_point(point: Point, angle_on_point: f32) -> Point {
    let angle = point.angle + angle_on_point;
    let pos = point.pos + vec2<f32>(cos(angle), sin(angle)) * point.radius * CELL_DISTANCE_SCALE;
    return Point(pos, pos, vec2<f32>(0.0, 0.0), angle, point.radius, point.flags);
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

    let stiffness = clamp(link.stiffness, 0.0, 1.0);
    let a_mass = point_a.radius * point_a.radius;
    let b_mass = point_b.radius * point_b.radius;

    var a_new = point_a;
    var b_new = point_b;

    var attachment_a = get_attachment_point(a_new, link.angle_from_a);
    var attachment_b = get_attachment_point(b_new, link.angle_from_b);

    // Constraint 1: Distance constraint between attachment points
    let correction = distance_constraint(attachment_a, attachment_b, 0.0, 1.0);
    attachment_a.pos += correction.xy;
    attachment_b.pos += correction.zw;

    // Constraint 2: Distance constraint between cell center and attachment point
    /*let correction2 = distance_constraint(a_new, attachment_a, a_new.radius, 1.0);
    a_new.pos += correction2.xy;
    attachment_a.pos += correction2.zw; 

    let correction3 = distance_constraint(b_new, attachment_b, b_new.radius, 1.0);
    b_new.pos += correction3.xy;
    attachment_b.pos += correction3.zw;*/
    // Compute cell pos a and b directly from the attachment point by working backwards from their angles
    let angle_to_attachment_a = a_new.angle + link.angle_from_a;
    a_new.pos = attachment_a.pos - vec2<f32>(cos(angle_to_attachment_a), sin(angle_to_attachment_a)) * a_new.radius * CELL_DISTANCE_SCALE;
    let angle_to_attachment_b = b_new.angle + link.angle_from_b;
    b_new.pos = attachment_b.pos - vec2<f32>(cos(angle_to_attachment_b), sin(angle_to_attachment_b)) * b_new.radius * CELL_DISTANCE_SCALE;

    let mass_ratio = b_mass / (a_mass + b_mass);
    let attachment_ab = attachment_a.pos + (attachment_b.pos - attachment_a.pos) * mass_ratio;

    // Constraint 3: Constraint angles between attachment points and cell center
    let correction4 = rotate_points_to_angle(a_new, b_new, attachment_ab, stiffness, M_PI);
    a_new.pos = correction4.xy;
    b_new.pos = correction4.zw;

    // Constraint 3: Min angle must be above 90*
    /*let correction4 = min_angle_constraint(a_new, b_new, attachment_ab, MIN_LINK_ANGLE);
    a_new.pos += correction4.xy;
    b_new.pos += correction4.zw;*/

    // Calculate change in position
    let delta_pos_a = a_new.pos - point_a.pos;
    let delta_pos_b = b_new.pos - point_b.pos;

    // Calculate change in angle
    let old_angle_a = point_a.angle + link.angle_from_a;
    let old_angle_b = point_b.angle + link.angle_from_b;

    // Calculate new attachment point angles in world space
    let new_attachment_a = get_attachment_point(a_new, link.angle_from_a);
    let new_attachment_b = get_attachment_point(b_new, link.angle_from_b);
    
    // Get direction vectors from center to attachment points
    let old_dir_a = vec2<f32>(cos(old_angle_a), sin(old_angle_a));
    let new_dir_a = normalize(new_attachment_a.pos - a_new.pos);
    let old_dir_b = vec2<f32>(cos(old_angle_b), sin(old_angle_b));
    let new_dir_b = normalize(new_attachment_b.pos - b_new.pos);
    
    // Calculate delta angles (automatically in range [-pi, pi] from signed_angle)
    let delta_angle_a = signed_angle(old_dir_a, new_dir_a);
    let delta_angle_b = signed_angle(old_dir_b, new_dir_b);

    let base_a_idx = cell_a.point_idx * 3u;
    let base_b_idx = cell_b.point_idx * 3u;

    apply_correction(base_a_idx, delta_pos_a, delta_angle_a);
    apply_correction(base_b_idx, delta_pos_b, delta_angle_b);
}
