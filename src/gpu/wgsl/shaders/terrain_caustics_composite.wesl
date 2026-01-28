// Terrain caustics composite (terrain + caustics projected onto height)

@group(0) @binding(0)
var base_texture: texture_2d<f32>;

@group(0) @binding(1)
var caustics_texture: texture_2d<f32>;

@group(0) @binding(2)
var height_texture: texture_2d<f32>;

@group(0) @binding(3)
var tex_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

fn brighten(color: vec4<f32>, amount: f32) -> vec4<f32> {
    let r = clamp(color.r * amount, 0.0, 1.0);
    let g = clamp(color.g * amount, 0.0, 1.0);
    let b = clamp(color.b * amount, 0.0, 1.0);
    return vec4<f32>(r, g, b, color.a);
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    var pos: vec2<f32>;
    switch vertex_index {
        case 0u: { pos = vec2<f32>(-1.0, -1.0); }
        case 1u: { pos = vec2<f32>(1.0, -1.0); }
        case 2u: { pos = vec2<f32>(-1.0, 1.0); }
        default: { pos = vec2<f32>(1.0, 1.0); }
    }
    out.clip_position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = vec2<f32>(
        (pos.x + 1.0) * 0.5,
        1.0 - (pos.y + 1.0) * 0.5
    );
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let base = textureSample(base_texture, tex_sampler, in.uv);
    let intensity = textureSample(caustics_texture, tex_sampler, in.uv).r;
    let height = textureSample(height_texture, tex_sampler, in.uv).r;
    let height_factor = height * 0.6;
    let scaled_intensity = intensity * height_factor;

    let brighten_amount = 1.0 + 10.0 * scaled_intensity;
    let brightened = brighten(base, brighten_amount);
    return vec4<f32>(brightened.rgb, base.a);
}
