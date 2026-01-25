import package::types::{Point, Cell};
import package::constants::{CELL_CAPACITY, CELL_FLAG_ACTIVE, POINT_FLAG_ACTIVE};

struct PickParams {
    mouse_pos: vec2<f32>,
    _pad: vec2<f32>,
}

struct PickResult {
    best_dist: atomic<u32>,
    cell_index: atomic<u32>,
}

@group(0) @binding(0)
var<uniform> pick: PickParams;

@group(0) @binding(1)
var<storage, read> points: array<Point>;

@group(0) @binding(2)
var<storage, read> cells: array<Cell>;

@group(0) @binding(3)
var<storage, read_write> result: PickResult;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let cell_idx = id.x;
    if cell_idx >= CELL_CAPACITY {
        return;
    }

    let cell = cells[cell_idx];
    if (cell.flags & CELL_FLAG_ACTIVE) == 0u {
        return;
    }

    let point = points[cell.point_idx];
    if (point.flags & POINT_FLAG_ACTIVE) == 0u {
        return;
    }

    let dx = point.pos.x - pick.mouse_pos.x;
    let dy = point.pos.y - pick.mouse_pos.y;
    let dist_sq = dx * dx + dy * dy;
    let radius = point.radius;

    if dist_sq > radius * radius {
        return;
    }

    let dist_bits = bitcast<u32>(dist_sq);
    let prev = atomicMin(&result.best_dist, dist_bits);
    if dist_bits <= prev {
        atomicStore(&result.cell_index, cell_idx);
    }
}
