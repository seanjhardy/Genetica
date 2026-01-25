@group(0) @binding(0) var viewport_texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Uniforms {
    screen_width: f32,
    screen_height: f32,
    time: f32,              // Time for animated effects
    _padding: f32,          // Alignment padding
}

@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(@location(0) pos: vec2<f32>, @location(1) uv: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;
    // Convert pixel coordinates to normalized device coordinates (-1 to 1)
    // Y is flipped because screen coordinates have (0,0) at top-left, but NDC has (0,0) at center
    let x = (pos.x / uniforms.screen_width) * 2.0 - 1.0;
    let y = 1.0 - (pos.y / uniforms.screen_height) * 2.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = uv;
    return out;
}

struct FragmentInput {
    @location(0) uv: vec2<f32>,
}

// Hash function for pseudo-random noise
fn hash3(p: vec3<f32>) -> f32 {
    var p3 = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Generate random noise in 2x2 pixel blocks that flickers each frame
fn random_noise(uv: vec2<f32>, time: f32, screen_width: f32, screen_height: f32) -> f32 {
    // Convert UV to pixel coordinates
    let pixel_pos = vec2<f32>(
        floor(uv.x * screen_width),
        floor(uv.y * screen_height)
    );
    
    // Group pixels into 2x2 blocks by dividing and flooring
    let block_pos = vec2<f32>(
        floor(pixel_pos.x / 10.0),
        floor(pixel_pos.y / 10.0)
    );
    
    // Combine block position with time to get unique random value per 2x2 block per frame
    // Use large prime numbers to ensure good distribution
    let seed = vec3<f32>(
        block_pos.x * 12.9898 + block_pos.y * 78.233,
        block_pos.x * 37.7193 + block_pos.y * 91.1837,
        time * 1000.0
    );

    return hash3(seed);
}

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    let center = vec2<f32>(0.5, 0.5);
    let uv_centered = input.uv - center;
    let dist_from_center = length(uv_centered);

       // === FISH EYE EFFECT ===
    let fish_eye_strength = 0.3;
    let fish_eye_radius = fish_eye_strength * dist_from_center * dist_from_center;
    let fish_eye_uv = input.uv + uv_centered * fish_eye_radius;

    // === CHROMATIC ABERRATION ===
    // Color fringing at edges like optical lens imperfection
    let aberration_strength = 0.01;
    let aberration = aberration_strength * dist_from_center * dist_from_center;

    let red_sample_point = fish_eye_uv + uv_centered * aberration;
    let green_sample_point = fish_eye_uv;
    let blue_sample_point = fish_eye_uv - uv_centered * aberration;

    // Use black border color for out-of-bounds samples (like real fisheye lens edges)
    // This prevents smearing artifacts from clamping
    let border_color = vec3<f32>(0.0, 0.0, 0.0);

    let red_in_bounds = (red_sample_point.x >= 0.0 && red_sample_point.x <= 1.0) && (red_sample_point.y >= 0.0 && red_sample_point.y <= 1.0);
    let green_in_bounds = (green_sample_point.x >= 0.0 && green_sample_point.x <= 1.0) && (green_sample_point.y >= 0.0 && green_sample_point.y <= 1.0);
    let blue_in_bounds = (blue_sample_point.x >= 0.0 && blue_sample_point.x <= 1.0) && (blue_sample_point.y >= 0.0 && blue_sample_point.y <= 1.0);

    let r = select(
        border_color.r,
        textureSample(viewport_texture, texture_sampler, red_sample_point).r,
        red_in_bounds
    );
    let g = select(
        border_color.g,
        textureSample(viewport_texture, texture_sampler, green_sample_point).g,
        green_in_bounds
    );
    let b = select(
        border_color.b,
        textureSample(viewport_texture, texture_sampler, blue_sample_point).b,
        blue_in_bounds
    );

    var color = vec4<f32>(r, g, b, 1.0);

    // === VIGNETTE ===
    // Darken edges like looking through microscope lens
    let vignette_strength = 0.8;
    let vignette_size = 0.5;
    let vignette = smoothstep(vignette_size, vignette_size * 0.3, dist_from_center);
    let vignette_factor = mix(1.0 - vignette_strength, 1.0, vignette);

    color = vec4<f32>(color.rgb * vignette_factor, color.a);

    // === NOISE ===
    // Add random flickering noise in 2x2 pixel blocks that changes each frame
    let noise_value = random_noise(input.uv, uniforms.time, uniforms.screen_width, uniforms.screen_height);
    
    // Increase noise strength towards edges (like film grain or sensor noise)
    let edge_noise_factor = smoothstep(0.3, 0.8, dist_from_center);
    let noise_strength = 0.1; // Base noise strength
    let edge_noise_strength = 0.8; // Additional noise at edges
    
    // Combine base noise with edge noise
    let total_noise_strength = noise_strength + edge_noise_factor * edge_noise_strength;
    
    // Darken color based on noise (subtract noise, making darker pixels darker)
    let noise_darkening = noise_value * total_noise_strength;
    color = vec4<f32>(color.rgb * (1.0 - noise_darkening), color.a);
    
    // Apply edge fade to prevent smearing artifacts
    color = vec4<f32>(color.rgb, color.a * 1.0);

    return color;
}