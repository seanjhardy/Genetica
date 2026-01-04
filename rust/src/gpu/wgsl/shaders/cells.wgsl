@include src/gpu/wgsl/types.wgsl;
@include src/gpu/wgsl/constants.wgsl;
@include src/gpu/wgsl/utils/color.wgsl;

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var<storage, read> points: array<VerletPoint>;

@group(0) @binding(2)
var<storage, read> cells: array<Cell>;

@group(0) @binding(5) var perlin_noise_texture: texture_2d<f32>;
@group(0) @binding(7) var perlin_noise_sampler: sampler;


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
}

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
        return out;
    }

    // LOD check: skip organelles when cell size in clip space is too small
    // Use a very lenient threshold - temporarily disabled for debugging
    // TODO: Re-enable LOD check once organelles are confirmed visible
    let cell_size_clip = max((cell_radius_world / view_size_x) * 2.0, (cell_radius_world / view_size_y) * 2.0);

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
    return out;
}

fn perlin_sample(pos: vec2<f32>) -> f32 {
    let noise_x_mod = pos.x - floor(pos.x / 200.0) * 200.0;
    let noise_y_mod = pos.y - floor(pos.y / 200.0) * 200.0;
    let noise_uv = vec2<f32>(noise_x_mod / 200.0, noise_y_mod / 200.0);
    let perlin_noise_sample = textureSample(perlin_noise_texture, perlin_noise_sampler, noise_uv).r;
    
    return perlin_noise_sample;
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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let cell_idx = u32(in.cell_index);
    let cell = cells[cell_idx];

    if (cell.flags & CELL_FLAG_ACTIVE) == 0u {
        discard;
    }

    let center = vec2<f32>(0.5, 0.5);
    let uv_offset = in.uv - center; // [-0.5, 0.5]
    let dist = length(uv_offset);

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
    // Full detail rendering for normal/zoomed out view
    // FIRST: Apply perlin noise perturbation to cell wall
    // Use non-rotated angle for noise sampling (cell wall doesn't rotate)
    let angle = atan2(uv_offset.y, uv_offset.x);
    let noise_value = cell_noise(cell.noise_permutations, angle);
    let perturbation_amount = in.radius * 0.3; // 10% of radius
    adjusted_radius = in.radius + noise_value * perturbation_amount;
    
    // SECOND: Clamp to midpoint boundary if near neighbors
    // This creates flat edges where cells are close together
    let perturbed_point = in.world_pos + pixel_dir_world * adjusted_radius;
    let search_radius = in.radius * 3.5;
    /*adjusted_radius = calculate_directional_radius(
        perturbed_point,
        in.world_pos,
        adjusted_radius,
        search_radius,
        cell_idx
    );*/
    
    // Render circle with adjusted radius
    // Use non-rotated distance for circle check
    let radius_normalized = 0.5 * (adjusted_radius / in.radius);

    if dist > radius_normalized {
        discard;
    }

    var color = in.color;

    // Border: draw at min(radius, midpoint to neighbour) - border_thickness
    let border_radius = adjusted_radius - in.cell_wall_thickness;
    let border_radius_normalized = 0.5 * (border_radius / in.radius);
    
    // Darken the border
    // Use non-rotated distance for border check
    if dist > border_radius_normalized {
        color = saturate(brighten(color, 3), 1.0);
    } else {
        color = saturate(brighten(color, 0.8), 0.9);
    }

    // Sample perlin noise texture using cell's random offset + pixel offset
    // Calculate the pixel's offset from cell center in world space
    let pixel_world_offset = uv_offset * in.max_radius * 2.0; // Scale UV to world space
    let texture_sample_pos = cell.noise_texture_offset + pixel_world_offset;
    
    /*let bg_sample = perlin_sample(texture_sample_pos);

    // Apply simple thresholded white tint
    if bg_sample > 0.5 {
        // Apply fixed white tint when noise is above threshold
        color = brighten(color, 1.5);
    }*/

    return color;
}
