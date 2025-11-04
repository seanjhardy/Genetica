@group(0) @binding(0) var atlas_texture: texture_2d<f32>;
@group(0) @binding(1) var atlas_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
}

struct Uniforms {
    screen_width: f32,
    screen_height: f32,
}

@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(@location(0) pos: vec2<f32>, @location(1) uv: vec2<f32>, @location(2) color: vec4<f32>) -> VertexOutput {
    var out: VertexOutput;
    // Convert pixel coordinates to normalized device coordinates (-1 to 1)
    // Y is flipped because screen coordinates have (0,0) at top-left, but NDC has (0,0) at center
    let x = (pos.x / uniforms.screen_width) * 2.0 - 1.0;
    let y = 1.0 - (pos.y / uniforms.screen_height) * 2.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = uv;
    out.color = color;
    return out;
}

struct FragmentInput {
    @location(0) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
}

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    // Texture is R8Unorm (single channel), so all RGB channels have the same value
    let tex_alpha = textureSample(atlas_texture, atlas_sampler, input.uv).r;
    // Use the vertex color and multiply by texture alpha
    return vec4<f32>(input.color.rgb * tex_alpha, input.color.a * tex_alpha);
}

