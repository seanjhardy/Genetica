@include src/gpu/wgsl/types.wgsl;
@include src/gpu/wgsl/constants.wgsl;
@include src/gpu/wgsl/utils/color.wgsl;

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
    @location(4) radius: f32, // Base radius (for cell body) or base organelle radius (for organelles)
    @location(5) world_pos: vec2<f32>,
    @location(6) organelle_index: f32, // 0 = cell body, 1-5 = organelles
    @location(7) organelle_world_pos: vec2<f32>, // World position of organelle center (or cell center if cell body)
    @location(8) max_radius: f32, // Maximum radius including perturbation (for quad sizing)
}

@vertex
fn vs_main(@builtin(instance_index) instance_index: u32, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Decode instance_index: each cell has 6 instances (0 = cell body, 1-5 = organelles)
    // instance_index = cell_index * 6 + organelle_index
    const INSTANCES_PER_CELL: u32 = 6u;
    let cell_idx = instance_index / INSTANCES_PER_CELL;
    let organelle_idx = instance_index % INSTANCES_PER_CELL;
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
        out.organelle_index = -1.0;
        out.organelle_world_pos = vec2<f32>(0.0);
        out.max_radius = 0.0;
        return out;
    }

    let cell_center = cell.pos;
    let cell_radius_world = cell.radius;
    
    // Determine if this is a cell body or organelle
    let is_organelle = organelle_idx > 0u;
    
    // Calculate world position and radius based on instance type
    var world_pos: vec2<f32>;
    var radius_world: f32;
    var organelle_world_pos: vec2<f32>;
    
    var base_radius_world: f32;
    var max_radius_world: f32;
    
    if is_organelle {
        // Get organelle position relative to cell center
        let org_idx = organelle_idx - 1u; // Convert to 0-4 range
        
        // Check if organelle data is valid (non-zero)
        let org_x_raw = cell.organelles[org_idx * 2u];
        let org_y_raw = cell.organelles[org_idx * 2u + 1u];
        
        // If organelle position is zero/empty, skip rendering this organelle
        if org_x_raw == 0.0 && org_y_raw == 0.0 {
            // Cull empty organelle
            out.clip_position = vec4<f32>(0.0, 0.0, -1.0, 1.0);
            out.cell_index = f32(cell_idx);
            out.uv = vec2<f32>(0.0);
            out.color = cell.color;
            out.cell_wall_thickness = cell.cell_wall_thickness;
            out.radius = 0.0;
            out.world_pos = cell_center;
            out.organelle_index = f32(organelle_idx);
            out.organelle_world_pos = cell_center;
            out.max_radius = 0.0;
            return out;
        }
        
        // Calculate organelle position relative to cell center (in normalized coordinates)
        let org_rel_pos = vec2<f32>(org_x_raw, org_y_raw);
        
        // Apply momentum offset: organelles lag behind cell movement
        // Calculate cell velocity (momentum)
        let cell_velocity = cell.pos - cell.prev_pos;
        // Apply momentum offset to organelle position (opposite direction of movement)
        // The organelle should lag behind, so offset in opposite direction of velocity
        let momentum_offset = -cell_velocity * 0.3; // 30% of velocity as offset
        let org_rel_pos_with_momentum = org_rel_pos + momentum_offset / cell_radius_world;
        
        // Clamp organelle position to stay within cell boundary
        // Ensure organelle center stays at least its radius away from cell edge
        let org_rel_dist = length(org_rel_pos_with_momentum);
        var max_org_radius: f32 = 0.0;
        if org_idx == 0u {
            max_org_radius = 0.6; // Large blob
        } else if org_idx < 4u {
            max_org_radius = 0.2; // Small blobs
        } else {
            max_org_radius = 0.3; // Nucleus
        }
        // Maximum distance from center = 1.0 - max_org_radius (to keep organelle inside)
        let max_allowed_dist = 1.0 - max_org_radius;
        
        var clamped_org_rel_pos = org_rel_pos_with_momentum;
        if org_rel_dist > max_allowed_dist {
            // Clamp to maximum allowed distance from center
            clamped_org_rel_pos = normalize(org_rel_pos_with_momentum) * max_allowed_dist;
        }
        
        // Convert to world space
        let org_x = clamped_org_rel_pos.x * cell_radius_world;
        let org_y = clamped_org_rel_pos.y * cell_radius_world;
        organelle_world_pos = cell_center + vec2<f32>(org_x, org_y);
        world_pos = organelle_world_pos;
        
        // Determine organelle base radius based on type
        if org_idx == 0u {
            // Large dark blob: 30% perturbation max
            base_radius_world = cell_radius_world * 0.6;
            max_radius_world = base_radius_world * 1.3; // Account for max perturbation
        } else if org_idx < 4u {
            // Small white blobs: 30% perturbation max
            base_radius_world = cell_radius_world * (0.1 + 0.05 * f32(org_idx));
            max_radius_world = base_radius_world * 1.3; // Account for max perturbation
        } else {
            // Nucleus: 25% perturbation max
            base_radius_world = cell_radius_world * 0.3;
            max_radius_world = base_radius_world * 1.25; // Account for max perturbation
        }
        radius_world = max_radius_world; // Use max radius for quad sizing
    } else {
        // Cell body
        world_pos = cell_center;
        organelle_world_pos = cell_center;
        base_radius_world = cell_radius_world;
        max_radius_world = cell_radius_world;
        radius_world = cell_radius_world;
    }
    
    let relative_x = world_pos.x - uniforms.camera.x;
    let relative_y = world_pos.y - uniforms.camera.y;

    let view_size_x = uniforms.sim_params.z / uniforms.sim_params.y;
    let view_size_y = uniforms.sim_params.w / uniforms.sim_params.y;

    let clip_x = (relative_x / view_size_x) * 2.0;
    let clip_y = -(relative_y / view_size_y) * 2.0;

    if abs(clip_x) > 10.0 || abs(clip_y) > 10.0 {
        out.clip_position = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        out.cell_index = f32(cell_idx);
        out.uv = vec2<f32>(0.0);
        out.color = cell.color;
        out.cell_wall_thickness = cell.cell_wall_thickness;
        out.radius = base_radius_world;
        out.world_pos = world_pos;
        out.organelle_index = f32(organelle_idx);
        out.organelle_world_pos = organelle_world_pos;
        out.max_radius = max_radius_world;
        return out;
    }

    // LOD check: skip organelles when cell size in clip space is too small
    // Use a very lenient threshold - temporarily disabled for debugging
    // TODO: Re-enable LOD check once organelles are confirmed visible
    let cell_size_clip = max((cell_radius_world / view_size_x) * 2.0, (cell_radius_world / view_size_y) * 2.0);
    const MIN_ORGANELLE_LOD: f32 = 0.0001; // Minimum cell size in clip space to show organelles (lower = more visible)
    
    // Temporarily disable LOD check to debug organelle visibility
    // if is_organelle && cell_size_clip < MIN_ORGANELLE_LOD {
    //     // Cull organelle by placing it off-screen
    //     out.clip_position = vec4<f32>(0.0, 0.0, -1.0, 1.0);
    //     out.cell_index = f32(cell_idx);
    //     out.uv = vec2<f32>(0.0);
    //     out.color = cell.color;
    //     out.cell_wall_thickness = cell.cell_wall_thickness;
    //     out.radius = base_radius_world;
    //     out.world_pos = world_pos;
    //     out.organelle_index = f32(organelle_idx);
    //     out.organelle_world_pos = organelle_world_pos;
    //     out.max_radius = max_radius_world;
    //     return out;
    // }

    let size_clip_x = (radius_world / view_size_x) * 2.0;
    let size_clip_y = (radius_world / view_size_y) * 2.0;

    var offset: vec2<f32>;
    var uv_offset: vec2<f32>;

    switch quad_vertex {
        case 0u {
            offset = vec2<f32>(-1.0, -1.0) * vec2<f32>(size_clip_x, size_clip_y);
            uv_offset = vec2<f32>(0.0, 1.0);
        }
        case 1u {
            offset = vec2<f32>(1.0, -1.0) * vec2<f32>(size_clip_x, size_clip_y);
            uv_offset = vec2<f32>(1.0, 1.0);
        }
        case 2u {
            offset = vec2<f32>(-1.0, 1.0) * vec2<f32>(size_clip_x, size_clip_y);
            uv_offset = vec2<f32>(0.0, 0.0);
        }
        default {
            offset = vec2<f32>(1.0, 1.0) * vec2<f32>(size_clip_x, size_clip_y);
            uv_offset = vec2<f32>(1.0, 0.0);
        }
    }

    // No z-offset needed since we render organelles in a separate pass after cell bodies
    out.clip_position = vec4<f32>(clip_x + offset.x, clip_y + offset.y, 0.0, 1.0);
    out.cell_index = f32(cell_idx);
    out.uv = uv_offset;
    out.color = cell.color;
    out.cell_wall_thickness = cell.cell_wall_thickness;
    out.radius = base_radius_world; // Store base radius for fragment shader calculations
    out.world_pos = world_pos;
    out.organelle_index = f32(organelle_idx);
    out.organelle_world_pos = organelle_world_pos;
    out.max_radius = max_radius_world; // Store max radius for reference
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
// current_radius is the perturbed radius, we'll clamp it to midpoint boundaries
fn calculate_directional_radius(
    point_on_circumference: vec2<f32>,
    cell_center: vec2<f32>,
    current_radius: f32,
    search_radius: f32,
    cell_index: u32
) -> f32 {
    var min_adjusted_radius = current_radius;
    
    // Get base radius from cell data (needed for midpoint calculation)
    let cell = cells[cell_index];
    let base_radius = cell.radius;
    
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
    // Account for both cell radius and search radius (use base_radius for search)
    let max_search_dist = base_radius + search_radius;
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
                        // Check if neighbor could potentially overlap with this cell (use base_radius)
                        let max_relevant_dist = base_radius + neighbor.radius + search_radius;
                        if to_neighbor_dist > max_relevant_dist || to_neighbor_dist < 0.001 {
                            var next_head: i32 = -1;
                            if neighbor_index < next_length {
                                next_head = cell_hash_next[neighbor_index];
                            }
                            head = next_head;
                            continue;
                        }
                        
                        // Check if circles overlap (using base radii)
                        let overlap_dist = base_radius + neighbor.radius;
                        if to_neighbor_dist < overlap_dist {
                            let neighbor_dir = to_neighbor / to_neighbor_dist;
                            
                            // Calculate midpoint boundary distance from cell center
                            // Midpoint is where distance to both edges is equal:
                            // r1 - d1 = r2 - d2, where d1 + d2 = to_neighbor_dist
                            // Solving: d1 = (to_neighbor_dist - neighbor.radius + base_radius) / 2.0
                            // Use base_radius for midpoint calculation (not perturbed radius)
                            let midpoint_dist_from_center = (to_neighbor_dist - neighbor.radius + base_radius) * 0.5;
                            
                            // Check if the pixel direction aligns with the neighbor direction
                            // If so, we need to clamp the perturbed radius to the midpoint boundary
                            let alignment = dot(pixel_dir, neighbor_dir);
                            
                            // Only adjust if:
                            // 1. The direction aligns (pointing toward neighbor)
                            // 2. The midpoint is inside the current perturbed circle (so we need to clamp)
                            if alignment > 0.0 && midpoint_dist_from_center < current_radius && midpoint_dist_from_center > 0.0 {
                                // Calculate where the ray in pixel_dir intersects the midpoint plane
                                // Ray: cell_center + t * pixel_dir * current_radius
                                // Plane: dot(point - cell_center, neighbor_dir) = midpoint_dist_from_center
                                // Solving: dot(t * pixel_dir * current_radius, neighbor_dir) = midpoint_dist_from_center
                                // t * dot(pixel_dir, neighbor_dir) * current_radius = midpoint_dist_from_center
                                // t = midpoint_dist_from_center / (alignment * current_radius)
                                // Distance from center to intersection = t * current_radius = midpoint_dist_from_center / alignment
                                let clamped_radius = midpoint_dist_from_center / alignment;
                                
                                // Only clamp if this would reduce the radius (point is past midpoint)
                                // This creates a flat edge at the midpoint boundary
                                if clamped_radius < current_radius {
                                    min_adjusted_radius = min(min_adjusted_radius, clamped_radius);
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

// Perlin noise function using permutation table for smooth cell wall perturbation
fn cell_noise(permutations: array<u32, CELL_WALL_SAMPLES>, angle: f32) -> f32 {
    // Normalize angle to [0, 2π] range
    let normalized_angle = angle - floor(angle / (2.0 * M_PI)) * (2.0 * M_PI);
    
    // Use a lower frequency for smoother variation
    // Map angle to fewer samples to create smoother curves
    let samples_per_circle = f32(CELL_WALL_SAMPLES) * 0.4; // Use 40% of samples for smoother curves
    let angle_scaled = normalized_angle * samples_per_circle / (2.0 * M_PI);
    
    // Get integer and fractional parts
    let i = floor(angle_scaled);
    let f = angle_scaled - i;
    
    // Get permutation values for interpolation points (wrap around)
    let i0 = u32(i) % CELL_WALL_SAMPLES;
    let i1 = (i0 + 1u) % CELL_WALL_SAMPLES;
    
    // Get hash values from permutations
    let hash0 = permutations[i0];
    let hash1 = permutations[i1];
    
    // Use hash to select gradient direction (in 1D, gradient is just -1 or 1)
    // Use hash bits to determine gradient sign and magnitude
    // This creates proper random variation instead of a cosine wave
    var grad0_sign: f32;
    if (hash0 & 1u) == 1u {
        grad0_sign = 1.0;
    } else {
        grad0_sign = -1.0;
    }
    
    var grad1_sign: f32;
    if (hash1 & 1u) == 1u {
        grad1_sign = 1.0;
    } else {
        grad1_sign = -1.0;
    }
    
    // Use more hash bits to add variation to gradient magnitude
    // This breaks up the pure cosine pattern
    let grad0_mag = 0.5 + (f32(hash0 & 0xFFu) / 255.0) * 0.5; // Range [0.5, 1.0]
    let grad1_mag = 0.5 + (f32(hash1 & 0xFFu) / 255.0) * 0.5; // Range [0.5, 1.0]
    
    let grad0 = grad0_sign * grad0_mag;
    let grad1 = grad1_sign * grad1_mag;
    
    // For 1D Perlin noise, compute the contribution from each grid point
    // At grid point 0: contribution = grad0 * distance_from_0 = grad0 * f
    // At grid point 1: contribution = grad1 * distance_from_1 = grad1 * (f - 1.0)
    let v0 = grad0 * f;
    let v1 = grad1 * (f - 1.0);
    
    // Use smoothstep interpolation (3rd order) for C1 continuity
    // This is smoother than linear but less smooth than cosine, giving more natural variation
    let t = f * f * (3.0 - 2.0 * f);
    
    // Interpolate smoothly
    let result = mix(v0, v1, t);
    
    // The result should now be in a good range, but we need to ensure it reaches [-1, 1]
    // The maximum occurs when gradients are opposite and f=0.5
    // At f=0.5: v0 = grad0*0.5, v1 = grad1*(-0.5), result ≈ 0.5*(grad0 - grad1)
    // When grad0=1, grad1=-1: result ≈ 0.5*(1 - (-1)) = 1.0 (good!)
    // But with varying magnitudes, we might need slight scaling
    // Actually, with the current setup, the range should be approximately [-1, 1]
    return clamp(result, -1.0, 1.0);
}

// Helper function to get organelle perturbation based on 3 sample points
fn get_organelle_perturbation(
    permutations: array<u32, CELL_WALL_SAMPLES>,
    organelle_pos: vec2<f32>,
    organelle_radius: f32
) -> f32 {
    // Sample noise at 3 points around the organelle (low frequency)
    let angle0 = atan2(organelle_pos.y, organelle_pos.x);
    let angle1 = angle0 + 2.094; // 120 degrees
    let angle2 = angle0 + 4.189; // 240 degrees
    
    let noise0 = cell_noise(permutations, angle0);
    let noise1 = cell_noise(permutations, angle1);
    let noise2 = cell_noise(permutations, angle2);
    
    // Average the 3 samples for smooth, low-frequency perturbation
    return (noise0 + noise1 + noise2) / 3.0;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let cell_idx = u32(in.cell_index);
    let cell = cells[cell_idx];

    if cell.is_alive == 0u {
        discard;
    }
    
    let organelle_idx = u32(in.organelle_index);
    let is_organelle = organelle_idx > 0u;
    
    // Calculate world position of this pixel
    let center = vec2<f32>(0.5, 0.5);
    let uv_offset = in.uv - center; // [-0.5, 0.5]
    let dist = length(uv_offset);
    
    if is_organelle {
        // Render organelle as a simple circle - no perturbation calculations for performance
        let org_idx = organelle_idx - 1u; // Convert to 0-4 range
        
        // Simple circle check using base radius (no perturbation)
        // in.radius is the base organelle radius, in.max_radius is the quad size
        let radius_normalized = 0.5 * (in.radius / in.max_radius) + in.radius * 0.05 * cell_noise(cell.noise_permutations, atan2(uv_offset.y, uv_offset.x));
        if dist > radius_normalized {
            discard;
        }
        
        // Check if organelle pixel is outside the cell boundary
        // Convert UV offset to world space relative to organelle center
        let uv_world_offset = uv_offset * in.max_radius * 2.0; // Scale from [-0.5, 0.5] to world space
        let organelle_pixel_world_pos = in.organelle_world_pos + uv_world_offset;
        let pixel_offset_from_cell = organelle_pixel_world_pos - cell.pos;
        let dist_from_cell_center = length(pixel_offset_from_cell);
        
        // Calculate the cell's actual boundary at this angle (accounting for noise and neighbor clipping)
        // This ensures organelles are clipped to the cell's actual shape, including neighbor interactions
        if dist_from_cell_center > 0.001 {
            let pixel_dir_from_cell = pixel_offset_from_cell / dist_from_cell_center;
            
            // Calculate cell boundary with noise perturbation
            let angle_from_cell = atan2(pixel_offset_from_cell.y, pixel_offset_from_cell.x);
            let noise_value = cell_noise(cell.noise_permutations, angle_from_cell);
            let perturbation_amount = cell.radius * 0.1; // 10% of radius
            var cell_boundary_radius = cell.radius + noise_value * perturbation_amount;
            
            // Apply neighbor clipping to get actual cell boundary
            let perturbed_point = cell.pos + pixel_dir_from_cell * cell_boundary_radius;
            let search_radius = cell.radius * 3.5;
            cell_boundary_radius = calculate_directional_radius(
                perturbed_point,
                cell.pos,
                cell_boundary_radius,
                search_radius,
                cell_idx
            );
            
            // Discard if organelle pixel is outside the actual cell boundary
            if dist_from_cell_center > cell_boundary_radius {
                discard;
            }
        }
        
        // Apply organelle-specific coloring
        var color = in.color;
        if org_idx == 0u {
            // Shadow
            color = brighten(saturate(color + vec4<f32>(0, 0, 0.9, 0), 2.0), 0.1);
            return alpha(color, 0.4);
        } else if org_idx < 4u {
            // Organelles
            color = brighten(saturate(color + vec4<f32>(0, 0, 0.1, 0), 1.3), 2.5);
            return alpha(color, 0.2);
        }  else {
            // Nucleus
            color = brighten(color, 1.7);
            if dist < radius_normalized * 0.5 {
                color = brighten(color, 1.8); // Additional brightening for center
            }
            return alpha(color, 0.5);
        } 
    } else {
        // Render cell body (existing logic)
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
        
        // LOD: Calculate cell size in clip space to determine detail level
        let view_size_x = uniforms.sim_params.z / uniforms.sim_params.y;
        let view_size_y = uniforms.sim_params.w / uniforms.sim_params.y;
        let cell_size_clip = max((in.radius / view_size_x) * 2.0, (in.radius / view_size_y) * 2.0);
        
        // When cells are very large on screen (zoomed in), skip expensive calculations
        // This prevents performance issues when zooming in
        const MAX_DETAIL_LOD: f32 = 0.1; // If cell size in clip space > 0.1, use simplified rendering
        
        var adjusted_radius: f32;
        // Full detail rendering for normal/zoomed out view
        // FIRST: Apply perlin noise perturbation to cell wall
        let angle = atan2(uv_offset.y, uv_offset.x);
        let noise_value = cell_noise(cell.noise_permutations, angle);
        let perturbation_amount = in.radius * 0.1; // 10% of radius
        adjusted_radius = in.radius + noise_value * perturbation_amount;
        
        // SECOND: Clamp to midpoint boundary if near neighbors
        // This creates flat edges where cells are close together
        let perturbed_point = in.world_pos + pixel_dir_world * adjusted_radius;
        let search_radius = in.radius * 3.5;
        adjusted_radius = calculate_directional_radius(
            perturbed_point,
            in.world_pos,
            adjusted_radius,
            search_radius,
            cell_idx
        );
        
        // Render circle with adjusted radius
        let radius_normalized = 0.5 * (adjusted_radius / in.radius);

        if dist > radius_normalized {
            discard;
        }

        var color = in.color;

        // Border: draw at min(radius, midpoint to neighbour) - border_thickness
        let border_radius = adjusted_radius - in.cell_wall_thickness;
        let border_radius_normalized = 0.5 * (border_radius / in.radius);
        
        // Darken the border
        if dist > border_radius_normalized {
            color = saturate(brighten(color, 3), 1.0);
        } else {
            color = saturate(brighten(color, 0.8), 0.9);
        }

        return color;
    }
}
