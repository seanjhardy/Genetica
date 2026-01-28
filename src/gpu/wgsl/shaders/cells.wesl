import package::types::{Uniforms, Point, Cell, Link};
import package::constants::{
    POINT_FLAG_ACTIVE,
    CELL_FLAG_ACTIVE,
    MAX_CELL_LINKS,
    LINK_FLAG_ACTIVE,
    CELL_WALL_SAMPLES,
    M_PI,
};
import package::utils::color::{brighten};
import package::utils::distortion::{fisheye_distortion};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var<storage, read> points: array<Point>;

@group(0) @binding(2)
var<storage, read> cells: array<Cell>;

@group(0) @binding(3) var perlin_noise_texture: texture_2d<f32>;
@group(0) @binding(4) var perlin_noise_sampler: sampler;
@group(0) @binding(5)
var<storage, read> links: array<Link>;


struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) cell_index: f32,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) cell_wall_thickness: f32,
    @location(4) radius: f32, // Base radius (for cell body)
    @location(5) world_pos: vec2<f32>,
    @location(6) max_radius: f32, // Maximum radius including perturbation (for quad sizing)
    @location(7) cell_angle: f32, // Cell rotation angle for UV rotation
    @location(8) link_plane0: vec4<f32>,
    @location(9) link_plane1: vec4<f32>,
    @location(10) link_plane2: vec4<f32>,
    @location(11) link_plane3: vec4<f32>,
    @location(12) min_plane_distance: f32,
}

const MAX_LINK_PLANES: u32 = 4u;

@vertex
fn vs_main(@builtin(instance_index) instance_index: u32, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    let cell_idx = instance_index;
    let quad_vertex = vertex_index;

    let cell = cells[cell_idx];

    let point = points[cell.point_idx];

    if (point.flags & POINT_FLAG_ACTIVE) == 0u || (cell.flags & CELL_FLAG_ACTIVE) == 0u {
        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.cell_index = 0.0;
        out.uv = vec2<f32>(0.0);
        out.color = vec4<f32>(0.0);
        out.cell_wall_thickness = 0.0;
        out.radius = 0.0;
        out.world_pos = vec2<f32>(0.0);
        out.max_radius = 0.0;
        out.cell_angle = 0.0;
        out.link_plane0 = vec4<f32>(0.0);
        out.link_plane1 = vec4<f32>(0.0);
        out.link_plane2 = vec4<f32>(0.0);
        out.link_plane3 = vec4<f32>(0.0);
        out.min_plane_distance = 1e9;
        return out;
    }

    let cell_center = point.pos;
    let cell_radius_world = point.radius;
    
    
    // Calculate world position and radius based on instance type
    var world_pos: vec2<f32> = cell_center;
    var radius_world: f32 = cell_radius_world;
    
    var base_radius_world: f32 = cell_radius_world;
    var max_radius_world: f32 = cell_radius_world;
    
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
        out.max_radius = max_radius_world;
        out.cell_angle = point.angle;
        out.link_plane0 = vec4<f32>(0.0);
        out.link_plane1 = vec4<f32>(0.0);
        out.link_plane2 = vec4<f32>(0.0);
        out.link_plane3 = vec4<f32>(0.0);
        out.min_plane_distance = 1e9;
        return out;
    }

    var link_planes: array<vec4<f32>, 4>;
    var link_plane_dists: array<f32, 4>;
    for (var i: u32 = 0u; i < MAX_LINK_PLANES; i = i + 1u) {
        link_planes[i] = vec4<f32>(0.0);
        link_plane_dists[i] = 1e9;
    }

    // Skip link calculations if cell is too small
    if cell_radius_world < 20 {
        let link_count = min(cell.link_count, MAX_CELL_LINKS);
        for (var i: u32 = 0u; i < link_count; i = i + 1u) {
            let link_idx = cell.link_indices[i];
            if link_idx >= arrayLength(&links) {
                continue;
            }

            let link = links[link_idx];
            if (link.flags & LINK_FLAG_ACTIVE) == 0u {
                continue;
            }

            var neighbor_idx: u32 = 0u;
            var neighbor_generation: u32 = 0u;
            if link.a_cell == cell_idx {
                neighbor_idx = link.b_cell;
                neighbor_generation = link.b_generation;
            } else if link.b_cell == cell_idx {
                neighbor_idx = link.a_cell;
                neighbor_generation = link.a_generation;
            } else {
                continue;
            }

            if neighbor_idx >= arrayLength(&cells) {
                continue;
            }

            let neighbor_cell = cells[neighbor_idx];
            if neighbor_cell.generation != neighbor_generation {
                continue;
            }
            if (neighbor_cell.flags & CELL_FLAG_ACTIVE) == 0u {
                continue;
            }
            if neighbor_cell.point_idx >= arrayLength(&points) {
                continue;
            }

            let neighbor_point = points[neighbor_cell.point_idx];
            if (neighbor_point.flags & POINT_FLAG_ACTIVE) == 0u {
                continue;
            }

            let to_neighbor = neighbor_point.pos - cell_center;
            let dist = length(to_neighbor);
            if dist <= 0.0001 {
                continue;
            }

            let half_dist = 0.5 * dist;
            let plane = vec4<f32>(to_neighbor / dist, half_dist, 1.0);

            var max_index: u32 = 0u;
            var max_dist: f32 = link_plane_dists[0];
            for (var j: u32 = 1u; j < MAX_LINK_PLANES; j = j + 1u) {
                if link_plane_dists[j] > max_dist {
                    max_dist = link_plane_dists[j];
                    max_index = j;
                }
            }

            if half_dist < max_dist {
                link_planes[max_index] = plane;
                link_plane_dists[max_index] = half_dist;
            }
        }
    }

    // LOD check: skip organelles when cell size in clip space is too small
    // Use a very lenient threshold - temporarily disabled for debugging
    // TODO: Re-enable LOD check once organelles are confirmed visible
    let cell_size_clip = max((cell_radius_world / view_size_x) * 2.0, (cell_radius_world / view_size_y) * 2.0);

    // Quad sized to the cell diameter so UVs map 0-1 across the cell.
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

    out.clip_position = vec4<f32>(clip_x + offset.x, clip_y + offset.y, 0.0, 1.0);
    out.cell_index = f32(cell_idx);
    out.uv = uv_offset;
    out.color = cell.color;
    out.cell_wall_thickness = cell.cell_wall_thickness;
    out.radius = base_radius_world; // Store base radius for fragment shader calculations
    out.world_pos = world_pos;
    out.max_radius = max_radius_world; // Store max radius for reference
    out.cell_angle = point.angle;
    out.link_plane0 = link_planes[0];
    out.link_plane1 = link_planes[1];
    out.link_plane2 = link_planes[2];
    out.link_plane3 = link_planes[3];
    var min_plane_distance: f32 = 1e9;
    for (var i: u32 = 0u; i < MAX_LINK_PLANES; i = i + 1u) {
        if link_plane_dists[i] < min_plane_distance {
            min_plane_distance = link_plane_dists[i];
        }
    }
    out.min_plane_distance = min_plane_distance;
    return out;
}

fn perlin_sample(pos: vec2<f32>) -> vec4<f32> {
    // Sample directly using UV coordinates (pos is already in appropriate range)
    let perlin_noise_sample = textureSample(perlin_noise_texture, perlin_noise_sampler, pos);
    return perlin_noise_sample;
}

// Simple linear interpolation between the 20 sample points
fn cell_noise(permutations: array<f32, CELL_WALL_SAMPLES>, angle: f32) -> f32 {
    // Normalize angle to [0, 2π] range
    let normalized_angle = angle - floor(angle / (2.0 * M_PI)) * (2.0 * M_PI);

    // Convert angle to fraction [0, 1) around the circle
    let angle_fraction = normalized_angle / (2.0 * M_PI);

    // Map to sample positions [0, 20)
    let sample_position = angle_fraction * f32(CELL_WALL_SAMPLES);

    // Get integer and fractional parts
    let i = floor(sample_position);
    let fraction = sample_position - i;

    // Get adjacent sample indices (wrap around)
    let sample0_idx = u32(i) % CELL_WALL_SAMPLES;
    let sample1_idx = (sample0_idx + 1u) % CELL_WALL_SAMPLES;

    // Get the sample values (simple modulo for randomness)
    let sample0 = permutations[sample0_idx];
    let sample1 = permutations[sample1_idx];

    // Linear interpolation between the two samples
    return mix(sample0, sample1, fraction);
}

fn clamp_radius_with_plane(plane: vec4<f32>, pixel_dir: vec2<f32>, current_radius: f32) -> f32 {
    if plane.w == 0.0 {
        return current_radius;
    }

    let denom = dot(plane.xy, pixel_dir);
    if denom <= 0.0001 {
        return current_radius;
    }

    let t = plane.z / denom;
    if t > 0.0 && t < current_radius {
        return t;
    }

    return current_radius;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let cell_idx = u32(in.cell_index);
    let cell = cells[cell_idx];

    if (cell.flags & CELL_FLAG_ACTIVE) == 0u {
        discard;
    }
    let selected_cell = uniforms.selection.x;
    let is_selected = selected_cell != 0xffffffffu && cell_idx == selected_cell;

    let center = vec2<f32>(0.5, 0.5);
    let uv_offset = (in.uv - center) * 2.0; // [-0.5, 0.5]
    let dist = length(uv_offset);
    let fragment_world_pos = in.world_pos + uv_offset * in.max_radius;
    let bounds = uniforms.bounds;
    if fragment_world_pos.x < bounds.x
        || fragment_world_pos.x > bounds.z
        || fragment_world_pos.y < bounds.y
        || fragment_world_pos.y > bounds.w {
        discard;
    }

    // Calculate the point on the circumference in the direction of this pixel
    // Use non-rotated UV for neighbor calculations
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

    // LOD: Skip noise texture sampling for cells smaller than 20 pixels on screen
    let cell_diameter_clip = max((in.radius / view_size_x) * 2.0, (in.radius / view_size_y) * 2.0);
    let cell_pixels = cell_diameter_clip * (uniforms.sim_params.z / 2.0);


    // Full detail rendering for normal/zoomed out view
    // FIRST: Apply perlin noise perturbation to cell wall
    // Use non-rotated angle for noise sampling (cell wall doesn't rotate)
    let angle = atan2(uv_offset.y, uv_offset.x);
    let noise_value = 0.0f;//cell_noise(cell.noise_permutations, angle);
    let perturbation_amount = in.radius * 0.1; // 10% of radius
    adjusted_radius = in.radius + noise_value * perturbation_amount;
    
    // SECOND: Clamp to midpoint boundaries for linked neighbors to flatten shared walls.
    if adjusted_radius > in.min_plane_distance && cell_pixels >= 20.0 {
        var planes: array<vec4<f32>, 4>;
        planes[0] = in.link_plane0;
        planes[1] = in.link_plane1;
        planes[2] = in.link_plane2;
        planes[3] = in.link_plane3;
        for (var i: u32 = 0u; i < MAX_LINK_PLANES; i = i + 1u) {
            adjusted_radius = clamp_radius_with_plane(planes[i], pixel_dir_world, adjusted_radius);
        }
    }
    
    // Base cell color from compute_cell_color
    var cell_color = in.color;

    // Render circle with adjusted radius (normalized to quad size)
    // Use non-rotated distance for circle check
    let radius_normalized = adjusted_radius / in.radius;

    if dist > radius_normalized {
        discard;
    }

    if dist > radius_normalized - cell.cell_wall_thickness {
        if is_selected {
            let highlight = vec3<f32>(0.55, 0.8, 1.0);
            cell_color = vec4<f32>(mix(cell_color.rgb, highlight, 0.85), cell_color.a);
        } else {
            cell_color = brighten(cell_color, 1.5);
        }
    }

    // Add directional indicator: brighten pixels within 5 degrees of cell's rotation angle
    // Calculate the angle of the current pixel relative to cell center
    let pixel_angle = atan2(uv_offset.y, uv_offset.x);
    
    // Calculate angular difference between pixel angle and cell angle
    var angle_diff = pixel_angle - in.cell_angle;
    
    // Normalize angle difference to [-PI, PI] range
    if angle_diff > M_PI {
        angle_diff -= 2.0 * M_PI;
    } else if angle_diff < -M_PI {
        angle_diff += 2.0 * M_PI;
    }
    
    // 5 degrees in radians
    let direction_threshold = 5.0 * M_PI / 180.0;
    
    // If within 5 degrees and not at center, brighten to show direction
    if abs(angle_diff) < direction_threshold && dist > 0.1 && dist < radius_normalized {
        cell_color = brighten(cell_color, 1.3);
    }

    if cell_pixels >= 20.0 {
        // Sample perlin noise texture using UV coordinates with fisheye distortion
        // Rotate UVs by cell angle to make texture follow cell rotation
        let cos_angle = cos(in.cell_angle);
        let sin_angle = sin(in.cell_angle);
        let rotation_matrix = mat2x2<f32>(
            cos_angle, -sin_angle,
            sin_angle, cos_angle
        );
        let rotated_uv = rotation_matrix * uv_offset;
        let local_uv = rotated_uv * 0.05;

        // Apply fisheye distortion normalized to cell radius
        // Points at cell edge (dist = radius_normalized) get maximum distortion
        let normalized_dist = dist / radius_normalized;
        let distortion_uv = rotated_uv / radius_normalized; // Normalize so cell edge = ±1.0
        let distorted_uv = fisheye_distortion(distortion_uv, 1.5);
        let final_local_uv = distorted_uv * 0.05; // Scale back to original local_uv range

        let cell_uv_offset = cell.noise_texture_offset / 400.0; // Convert world offset to UV offset (400x400 texture)
        let texture_sample_uv = cell_uv_offset + final_local_uv;

        let bg_sample = perlin_sample(texture_sample_uv);

        // Sample each channel separately for more varied texturing
        let noise_r = bg_sample[0]; // Red channel
        let noise_g = bg_sample[1]; // Green channel
        let noise_b = bg_sample[2]; // Blue channel

        // Apply brightness increases based on different thresholds for each channel

        var brightness_boost = -0.5;
        if noise_r > 0.5 {
            brightness_boost += 0.2; // Red channel contribution - lower threshold for visibility
        }
        if noise_g > 0.5 {
            brightness_boost += 0.4; // Green channel contribution
        }
        if noise_b > 0.6 {
            brightness_boost += 0.8; // Blue channel contribution
        }

        brightness_boost *= (1.0 - dist);

        cell_color = brighten(cell_color, 1.0 + brightness_boost);
    }

    let blue_tint = vec3<f32>(0.95, 0.98, 1.05);
    cell_color = vec4<f32>(cell_color.rgb * blue_tint, cell_color.a);

    let normalized_dist = clamp(dist / max(radius_normalized, 1.0e-4), 0.0, 1.0);
    let edge_opacity = smoothstep(0.1, 1.0, normalized_dist);
    var out_a = mix(0.4, 0.9, edge_opacity);
    if dist > radius_normalized - cell.cell_wall_thickness {
        out_a = 1.0;
    }

    return vec4<f32>(cell_color.rgb, out_a * cell_color.a);
}
