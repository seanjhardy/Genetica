@include src/gpu/wgsl/types.wgsl;
@include src/gpu/wgsl/constants.wgsl;

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var<storage, read> cells: array<Cell>;

@group(0) @binding(2)
var<storage, read> cell_free_list: CellFreeList;

@group(0) @binding(6)
var<storage, read> links: array<Link>;

@group(0) @binding(9)
var<storage, read> cell_bucket_heads_readonly: array<i32>;

@group(0) @binding(11)
var<storage, read> cell_hash_next: array<i32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) cell_index: f32,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) cell_wall_thickness: f32,
    @location(4) radius: f32,
    @location(5) world_pos: vec2<f32>,
}

@vertex
fn vs_main(@builtin(instance_index) instance_index: u32, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let cell_idx = instance_index;
    let quad_vertex = vertex_index;

    let cell = cells[cell_idx];
    if cell.is_alive == 0u {
        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.cell_index = 0.0;
        out.uv = vec2<f32>(0.0);
        out.color = vec4<f32>(0.0);
        out.cell_wall_thickness = 0.0;
        out.radius = 0.0;
        out.world_pos = vec2<f32>(0.0);
        return out;
    }

    let world_pos = cell.pos;
    let relative_x = world_pos.x - uniforms.camera.x;
    let relative_y = world_pos.y - uniforms.camera.y;

    let view_size_x = uniforms.sim_params.z / uniforms.sim_params.y;
    let view_size_y = uniforms.sim_params.w / uniforms.sim_params.y;

    let clip_x = (relative_x / view_size_x) * 2.0;
    let clip_y = -(relative_y / view_size_y) * 2.0;

    if abs(clip_x) > 10.0 || abs(clip_y) > 10.0 {
        out.clip_position = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        return out;
    }

    let cell_radius_world = cell.radius;
    let cell_size_clip_x = (cell_radius_world / view_size_x) * 2.0;
    let cell_size_clip_y = (cell_radius_world / view_size_y) * 2.0;

    var offset: vec2<f32>;
    var uv_offset: vec2<f32>;

    switch quad_vertex {
        case 0u {
            offset = vec2<f32>(-1.0, -1.0) * vec2<f32>(cell_size_clip_x, cell_size_clip_y);
            uv_offset = vec2<f32>(0.0, 1.0);
        }
        case 1u {
            offset = vec2<f32>(1.0, -1.0) * vec2<f32>(cell_size_clip_x, cell_size_clip_y);
            uv_offset = vec2<f32>(1.0, 1.0);
        }
        case 2u {
            offset = vec2<f32>(-1.0, 1.0) * vec2<f32>(cell_size_clip_x, cell_size_clip_y);
            uv_offset = vec2<f32>(0.0, 0.0);
        }
        default {
            offset = vec2<f32>(1.0, 1.0) * vec2<f32>(cell_size_clip_x, cell_size_clip_y);
            uv_offset = vec2<f32>(1.0, 0.0);
        }
    }

    out.clip_position = vec4<f32>(clip_x + offset.x, clip_y + offset.y, 0.0, 1.0);
    out.cell_index = f32(cell_idx);
    out.uv = uv_offset;
    out.color = cell.color;
    out.cell_wall_thickness = cell.cell_wall_thickness;
    out.radius = cell.radius;
    out.world_pos = world_pos;
    return out;
}

// Non-atomic version of hash function
fn hash_cell_position_readonly(pos: vec2<f32>) -> u32 {
    let bucket_count = arrayLength(&cell_bucket_heads_readonly);
    if bucket_count == 0u {
        return 0u;
    }
    let grid = vec2<i32>(floor(pos / HASH_CELL_SIZE));
    let hashed = (grid.x * 73856093) ^ (grid.y * 19349663);
    let mask = bucket_count - 1u;
    return u32(hashed) & mask;
}

// Calculate adjusted radius based on nearby neighbors
// Takes a point on the circumference and checks if it should be adjusted
fn calculate_directional_radius(
    point_on_circumference: vec2<f32>,
    cell_center: vec2<f32>,
    current_radius: f32,
    search_radius: f32,
    cell_index: u32
) -> f32 {
    var min_adjusted_radius = current_radius;
    
    // Direction from cell center to point on circumference
    let pixel_offset = point_on_circumference - cell_center;
    let pixel_dist = length(pixel_offset);

    if pixel_dist == 0.0 {
        return current_radius;
    }

    let pixel_dir = pixel_offset / pixel_dist;

    let bucket_count = arrayLength(&cell_bucket_heads_readonly);
    if bucket_count == 0u {
        return current_radius;
    }

    let cell_capacity = arrayLength(&cells);
    let next_length = arrayLength(&cell_hash_next);
    
    // Search nearby grid cells using spatial hash
    // Search based on cell center to ensure we find all neighbors that could affect this pixel
    // Account for both cell radius and search radius
    let max_search_dist = current_radius + search_radius;
    let search_grid_size = max(2u, u32(ceil(max_search_dist / HASH_CELL_SIZE)));

    var dx: i32 = -i32(search_grid_size);
    loop {
        if dx > i32(search_grid_size) {
            break;
        }

        var dy: i32 = -i32(search_grid_size);
        loop {
            if dy > i32(search_grid_size) {
                break;
            }

            // Search grid cells around the cell center (not pixel position)
            // This ensures we find all neighbors regardless of pixel position
            let grid_pos = cell_center + vec2<f32>(f32(dx), f32(dy)) * HASH_CELL_SIZE;
            let neighbor_hash = hash_cell_position_readonly(grid_pos);

            if neighbor_hash >= bucket_count {
                dy = dy + 1;
                continue;
            }

            var head = cell_bucket_heads_readonly[neighbor_hash];
            loop {
                if head == -1 {
                    break;
                }

                var neighbor_index = u32(head);
                var next_head: i32 = -1;

                if neighbor_index < cell_capacity {
                    let neighbor = cells[neighbor_index];
                    if neighbor.is_alive != 0u {
                        // Vector from current cell center to neighbor center
                        let to_neighbor = neighbor.pos - cell_center;
                        let to_neighbor_dist = length(to_neighbor);
                        
                        // Skip if neighbor is too far away to affect any pixel of this cell
                        // Check if neighbor could potentially overlap with this cell
                        let max_relevant_dist = current_radius + neighbor.radius + search_radius;
                        if to_neighbor_dist > max_relevant_dist || to_neighbor_dist < 0.001 {
                            var next_head: i32 = -1;
                            if neighbor_index < next_length {
                                next_head = cell_hash_next[neighbor_index];
                            }
                            head = next_head;
                            continue;
                        }
                        
                        // Check if circles overlap
                        let overlap_dist = current_radius + neighbor.radius;
                        if to_neighbor_dist < overlap_dist {
                            let neighbor_dir = to_neighbor / to_neighbor_dist;
                            
                            // Calculate midpoint boundary distance from cell center
                            // Midpoint is where distance to both edges is equal:
                            // r1 - d1 = r2 - d2, where d1 + d2 = to_neighbor_dist
                            // Solving: d1 = (to_neighbor_dist - neighbor.radius + current_radius) / 2.0
                            let midpoint_dist_from_center = (to_neighbor_dist - neighbor.radius + current_radius) * 0.5;
                            
                            // Only adjust if midpoint is inside the current circle (cells overlap)
                            if midpoint_dist_from_center < current_radius && midpoint_dist_from_center > 0.0 {
                                // Algorithm: Calculate where the ray from center through P intersects the midpoint plane
                                // The midpoint plane is perpendicular to the line A->B at distance midpoint_dist_from_center
                                
                                // Point P on circumference: at distance current_radius in direction pixel_dir
                                let P = point_on_circumference;
                                
                                // Check if P is inside neighbor's circle (would be in overlap region)
                                let P_to_neighbor = P - neighbor.pos;
                                let P_to_neighbor_dist = length(P_to_neighbor);

                                if P_to_neighbor_dist < neighbor.radius {
                                    // P is inside neighbor's circle - need to project onto midpoint plane
                                    // The midpoint plane is perpendicular to neighbor_dir at distance midpoint_dist_from_center
                                    // Project P onto the line from cell center to neighbor
                                    let P_to_cell = P - cell_center;
                                    let proj_dist_along_line = dot(P_to_cell, neighbor_dir);
                                    
                                    // If projection is past midpoint, calculate intersection with plane
                                    if proj_dist_along_line > midpoint_dist_from_center {
                                        // The plane equation: dot(point, neighbor_dir) = midpoint_dist_from_center
                                        // Ray: cell_center + t * pixel_dir * current_radius
                                        // Intersection: dot(cell_center + t * pixel_dir * current_radius, neighbor_dir) = midpoint_dist_from_center
                                        // Solving: dot(cell_center, neighbor_dir) + t * dot(pixel_dir * current_radius, neighbor_dir) = midpoint_dist_from_center
                                        // t = (midpoint_dist_from_center - dot(cell_center, neighbor_dir)) / dot(pixel_dir * current_radius, neighbor_dir)
                                        // But cell_center is the origin relative to itself, so dot(cell_center, neighbor_dir) = 0
                                        // t = midpoint_dist_from_center / (dot(pixel_dir, neighbor_dir) * current_radius)

                                        let alignment = dot(pixel_dir, neighbor_dir);
                                        if alignment > 0.0 {
                                            // Calculate where ray intersects the midpoint plane
                                            // Plane: dot(point - cell_center, neighbor_dir) = midpoint_dist_from_center
                                            // Ray: cell_center + t * pixel_dir * current_radius
                                            // Solving: dot(t * pixel_dir * current_radius, neighbor_dir) = midpoint_dist_from_center
                                            // t = midpoint_dist_from_center / (alignment * current_radius)
                                            // Distance from center to intersection = t * current_radius = midpoint_dist_from_center / alignment
                                            let adjusted_radius = midpoint_dist_from_center / alignment;
                                            // Clamp to current radius (can't be larger than original)
                                            let clamped_radius = min(adjusted_radius, current_radius);
                                            min_adjusted_radius = min(min_adjusted_radius, clamped_radius);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                
                // Move to next cell in hash chain
                if neighbor_index < next_length {
                    next_head = cell_hash_next[neighbor_index];
                }
                head = next_head;
            }

            dy = dy + 1;
        }

        dx = dx + 1;
    }

    return min_adjusted_radius;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let cell_idx = u32(in.cell_index);
    let cell = cells[cell_idx];

    if cell.is_alive == 0u {
        discard;
    }
    
    // Calculate world position of this pixel
    let center = vec2<f32>(0.5, 0.5);
    let uv_offset = in.uv - center; // [-0.5, 0.5]
    let dist = length(uv_offset);
    
    // Calculate the point on the circumference in the direction of this pixel
    // Normalize the UV offset to get direction, then project to circumference at original radius
    var pixel_dir_world: vec2<f32>;
    if dist > 0.0 {
        let uv_dir = uv_offset / dist;
        // Convert UV direction to world space direction
        let world_dir = uv_dir * in.radius * 2.0;
        pixel_dir_world = normalize(world_dir);
    } else {
        pixel_dir_world = vec2<f32>(1.0, 0.0);
    }
    
    // Point on circumference at original radius in this direction
    let point_on_circumference = in.world_pos + pixel_dir_world * in.radius;
    
    // Calculate adjusted radius based on neighbors
    // Increase search radius to ensure we find all potentially overlapping neighbors
    let search_radius = in.radius * 3.5; // Search wider to catch all neighbors
    let adjusted_radius = calculate_directional_radius(
        point_on_circumference,
        in.world_pos,
        in.radius,
        search_radius,
        cell_idx
    );
    
    // Render circle with adjusted radius
    let radius_normalized = 0.5 * (adjusted_radius / in.radius);

    if dist > radius_normalized {
        discard;
    }

    var final_color = in.color;

    // Border: draw at min(radius, midpoint to neighbour) - border_thickness
    // The adjusted_radius already accounts for midpoints, so border is at adjusted_radius - border_thickness
    let border_radius = adjusted_radius - in.cell_wall_thickness;
    let border_radius_normalized = 0.5 * (border_radius / in.radius);
    
    // Darken the border
    if dist > border_radius_normalized {
        final_color = final_color * 0.2;
    }

    // Add nucleus
    if dist < 0.2 / in.radius {
        final_color = final_color * 0.2;
    }

    return final_color;
}
