fn generate_random_position(seed: f32) -> vec2<f32> {
    let bounds_left = uniforms.bounds.x;
    let bounds_top = uniforms.bounds.y;
    let bounds_right = uniforms.bounds.z;
    let bounds_bottom = uniforms.bounds.w;

    let rand_x = rand_01(seed);
    let rand_y = rand_01(seed + 777.0);

    let padding = 10.0;
    let pos_x = bounds_left + padding + rand_x * (bounds_right - bounds_left - 2.0 * padding);
    let pos_y = bounds_top + padding + rand_y * (bounds_bottom - bounds_top - 2.0 * padding);

    return vec2<f32>(pos_x, pos_y);
}

fn spawn_cell_at_slot(
    point_slot: u32,
    cell_slot: u32,
    position: vec2<f32>,
    velocity: vec2<f32>,
    radius: f32,
    color: vec4<f32>,
    angle: f32,
    seed: f32,
    has_parent: bool,
    parent: Cell,
    lifeform_id: u32,
    generation: u32,
    energy: f32
) {
    var point: VerletPoint;
    point.pos = position;
    point.prev_pos = position - velocity * uniforms.sim_params.x;
    point.accel = vec2<f32>(0.0, 0.0);
    point.angle = angle;
    point.radius = radius;
    point.flags = POINT_FLAG_ACTIVE;

    var cell: Cell;
    if has_parent {
        // Create a fresh cell instead of copying parent to avoid data corruption
        cell.cell_wall_thickness = parent.cell_wall_thickness;
        cell.noise_permutations = array<f32, CELL_WALL_SAMPLES>(); // Initialize empty
        cell.noise_texture_offset = parent.noise_texture_offset;
        cell.color = parent.color; // Inherit parent's color
    } else {
        cell.cell_wall_thickness = 0.05;
        cell.noise_permutations = array<f32, CELL_WALL_SAMPLES>();
        let margin = 75.0;
        let max_pos = 400.0 - margin;
        cell.noise_texture_offset = vec2<f32>(
            margin + rand_01(seed + 1000.0) * (max_pos - margin),
            margin + rand_01(seed + 2000.0) * (max_pos - margin)
        );
        cell.color = vec4<f32>(0.0);
    }

    cell.point_idx = point_slot;
    cell.lifeform_id = lifeform_id;
    cell.generation = generation;
    cell.energy = energy;

    cell.color = color;
    cell.flags = CELL_FLAG_ACTIVE;
    cell.link_count = 0u;
    for (var i: u32 = 0u; i < MAX_CELL_LINKS; i = i + 1u) {
        cell.link_indices[i] = 0u;
    }

    points[point_slot] = point;
    cells[cell_slot] = cell;
}

fn add_link_to_cell(cell_index: u32, link_index: u32) {
    if cell_index >= arrayLength(&cells) {
        return;
    }

    var cell = cells[cell_index];
    if cell.link_count >= MAX_CELL_LINKS {
        return;
    }

    cell.link_indices[cell.link_count] = link_index;
    cell.link_count = cell.link_count + 1u;
    cells[cell_index] = cell;
}

fn create_link(
    a_cell: u32,
    a_generation: u32,
    b_cell: u32,
    b_generation: u32,
    angle_from_a: f32,
    angle_from_b: f32,
    stiffness: f32
) -> Link {
    var link: Link;
    link.a_cell = a_cell;
    link.a_generation = a_generation;
    link.b_cell = b_cell;
    link.b_generation = b_generation;
    link.angle_from_a = angle_from_a;
    link.angle_from_b = angle_from_b;
    link.stiffness = stiffness;
    link.flags = LINK_FLAG_ACTIVE;
    link._pad0 = 0u;
    link._pad1 = 0u;
    link._pad2 = 0u;
    link._pad3 = 0u;
    return link;
}
