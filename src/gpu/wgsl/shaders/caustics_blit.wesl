@group(0) @binding(0)
var caustics_texture: texture_2d<f32>;

@group(0) @binding(1)
var caustics_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
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
    return textureSample(caustics_texture, caustics_sampler, in.uv);
}
