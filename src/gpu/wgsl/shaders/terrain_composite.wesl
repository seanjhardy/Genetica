import package::utils::color::{brighten};

// Terrain composite (height + rock noise + shadow -> final color)

const DEBUG_HEIGHT: bool = false;

@group(0) @binding(0)
var height_texture: texture_2d<f32>;

@group(0) @binding(1)
var rock_noise_texture: texture_2d<f32>;

@group(0) @binding(2)
var shadow_texture: texture_2d<f32>;

@group(0) @binding(3)
var tex_sampler: sampler;

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
    let height = textureSample(height_texture, tex_sampler, in.uv).r;
    let rock_noise = textureSample(rock_noise_texture, tex_sampler, in.uv);
    let dims = vec2<f32>(textureDimensions(shadow_texture));
    let texel = 1.0 / max(dims, vec2<f32>(1.0, 1.0));

    if (DEBUG_HEIGHT) {
        return vec4<f32>(height, height, height, 1.0);
    }

    // Rock color
    let rock_brown = vec4<f32>(89, 89, 89, 255.0) / 255.0;
    let rock_gray = vec4<f32>(71, 40, 25, 255.0) / 255.0;
    let rock_vegetation = vec4<f32>(15, 97, 31, 255.0) / 255.0;

    let rock_hue = mix(rock_brown, rock_gray, rock_noise.r);
    let rock_plus_vegetation = mix(rock_hue, rock_vegetation, rock_noise.g);
    let rock_color = brighten(rock_hue, 0.5 + rock_noise.b * 0.5);

    // Tint the colour based on depth
    let deep = vec4<f32>(1, 3, 18, 255.0) / 1000.0;
    let shallow = vec4<f32>(1, 3, 18, 255.0) / 20.0;

    let water_color = mix(deep, shallow, height);

    var color = mix(rock_color, water_color, 1.0 - pow(height, 4) * 0.9);

    let shadow = textureSample(shadow_texture, tex_sampler, in.uv).r;
    color = vec4<f32>(color.rgb * shadow, 1.0);

    return color;
}
