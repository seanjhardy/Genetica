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

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    let center = vec2<f32>(0.5, 0.5);
    let uv_centered = input.uv - center;
    let dist_from_center = length(uv_centered);
    
    // === DEPTH OF FIELD ===
    // Blur edges while keeping center sharp (simulates microscope focal plane)
    let focus_range = 0.35;  // Area that stays sharp
    let blur_strength = smoothstep(focus_range, 0.7, dist_from_center);
    
    // Calculate blur based on distance from center
    /*let blur_radius = blur_strength * 0.003;  // Max blur radius in UV space
    
    // Multi-sample blur (simple box blur with 5 samples for performance)
    var blurred_color = vec4<f32>(0.0);
    if (blur_strength > 0.01) {
        // Use gaussian blur for better quality
        blurred_color += textureSample(viewport_texture, texture_sampler, input.uv);
        blurred_color += textureSample(viewport_texture, texture_sampler, input.uv + vec2<f32>(blur_radius, 0.0));
        blurred_color += textureSample(viewport_texture, texture_sampler, input.uv - vec2<f32>(blur_radius, 0.0));
        blurred_color += textureSample(viewport_texture, texture_sampler, input.uv + vec2<f32>(0.0, blur_radius));
        blurred_color += textureSample(viewport_texture, texture_sampler, input.uv - vec2<f32>(0.0, blur_radius));
        blurred_color /= 5.0;
    } else {
        blurred_color = textureSample(viewport_texture, texture_sampler, input.uv);
    }*/
    
    // === CHROMATIC ABERRATION ===
    // Color fringing at edges like optical lens imperfection
    let aberration_strength = 0.05;  // Increased for more noticeable effect
    let aberration = aberration_strength * dist_from_center * dist_from_center;
    
    // Sample each color channel with slight offset (more offset at edges)
    // Apply to the blurred result for combined effect
    let r = textureSample(viewport_texture, texture_sampler, clamp(input.uv + uv_centered * aberration, vec2<f32>(0.0), vec2<f32>(1.0))).r;
    let g = textureSample(viewport_texture, texture_sampler, input.uv).g;
    let b = textureSample(viewport_texture, texture_sampler, clamp(input.uv - uv_centered * aberration, vec2<f32>(0.0), vec2<f32>(1.0))).b;
    
    var color = vec4<f32>(r, g, b, 1.0);
    
    // === VIGNETTE ===
    // Darken edges like looking through microscope lens
    let vignette_strength = 0.6;
    let vignette_size = 0.7; 
    let vignette = smoothstep(vignette_size, vignette_size * 0.3, dist_from_center);
    let vignette_factor = mix(1.0 - vignette_strength, 1.0, vignette);
    
    color = vec4<f32>(color.rgb * vignette_factor, color.a);
    
    return color;
}

