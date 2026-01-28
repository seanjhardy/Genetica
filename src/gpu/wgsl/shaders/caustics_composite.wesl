@group(0) @binding(0)
var base_texture: texture_2d<f32>;

@group(0) @binding(1)
var caustics_texture: texture_2d<f32>;

@group(0) @binding(2)
var height_texture: texture_2d<f32>;

@group(0) @binding(3)
var tex_sampler: sampler;

struct CausticsCompositeUniform {
    camera_pos_zoom_thickness: vec4<f32>,
    view_size_grid: vec4<f32>,
    grid_threshold_padding: vec4<f32>,
    border_color: vec4<f32>,
    bounds: vec4<f32>,
    padding: vec4<f32>,
    time_params: vec4<f32>,
}

@group(0) @binding(4)
var<uniform> uniform_data: CausticsCompositeUniform;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) screen_uv: vec2<f32>,
    @location(1) world_pos: vec2<f32>,
}

fn sample_height(uv: vec2<f32>) -> f32 {
    let clamped = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
    return clamp(textureSample(height_texture, tex_sampler, clamped).r, 0.0, 1.0);
}

fn parallax_uv_for_height(
    base_uv: vec2<f32>,
    view_dir_uv: vec2<f32>,
    parallax_strength: f32,
    view_height: f32,
    min_steps: f32,
    max_steps: f32
) -> vec2<f32> {
    if (parallax_strength <= 0.0) {
        return base_uv;
    }

    let view_height_clamped = max(view_height, 1.0e-4);
    let view_dir = normalize(vec3<f32>(view_dir_uv, view_height_clamped));
    let ndotv = clamp(abs(view_dir.z), 0.0, 1.0);
    let steps_f = clamp(mix(max_steps, min_steps, ndotv), 1.0, max_steps);
    let steps = max(u32(steps_f + 0.5), 1u);
    let layer_depth = 1.0 / f32(steps);

    var curr_uv = base_uv;
    var curr_depth = 0.0;
    var height = 1.0 - sample_height(curr_uv);
    let parallax_dir = (view_dir.xy / max(view_dir.z, 1.0e-4)) * parallax_strength;
    let step_uv = parallax_dir * layer_depth;

    var i = 0u;
    loop {
        if (i >= steps || curr_depth >= height) {
            break;
        }
        curr_uv = curr_uv + step_uv;
        curr_depth = curr_depth + layer_depth;
        height = 1.0 - sample_height(curr_uv);
        i = i + 1u;
    }

    return curr_uv;
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

    let camera_pos = uniform_data.camera_pos_zoom_thickness.xy;
    let zoom = uniform_data.camera_pos_zoom_thickness.z;
    let view_size = uniform_data.view_size_grid.xy;

    let visible_width = view_size.x / zoom;
    let visible_height = view_size.y / zoom;

    let world_pos = vec2<f32>(
        camera_pos.x + pos.x * visible_width * 0.5,
        camera_pos.y - pos.y * visible_height * 0.5
    );

    out.clip_position = vec4<f32>(pos, 0.0, 1.0);
    out.screen_uv = vec2<f32>(
        (pos.x + 1.0) * 0.5,
        1.0 - (pos.y + 1.0) * 0.5
    );
    out.world_pos = world_pos;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let base = textureSample(base_texture, tex_sampler, in.screen_uv);
    let bounds = uniform_data.bounds;
    let bounds_size = vec2<f32>(bounds.z - bounds.x, bounds.w - bounds.y);
    let bounds_center = vec2<f32>(
        (bounds.x + bounds.z) * 0.5,
        (bounds.y + bounds.w) * 0.5
    );
    let expanded_size = bounds_size * 2.0;
    let height_uv = (in.world_pos - bounds_center) / expanded_size + vec2<f32>(0.5, 0.5);
    let height_uv_clamped = clamp(height_uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let height = textureSample(height_texture, tex_sampler, height_uv_clamped).r;

    // Project caustics onto terrain by marching along the heightfield.
    let camera_pos = uniform_data.camera_pos_zoom_thickness.xy;
    let view_dir_uv = (in.world_pos - camera_pos) / max(expanded_size, vec2<f32>(1.0e-4));
    let parallax_strength = uniform_data.padding.x * 0.5;
    let caustics_uv = clamp(
        parallax_uv_for_height(height_uv_clamped, view_dir_uv, parallax_strength, 1.0, 6.0, 20.0),
        vec2<f32>(0.0),
        vec2<f32>(1.0)
    );
    let intensity = textureSample(caustics_texture, tex_sampler, caustics_uv).r;
    // Favor water/low heights so we don't double-brighten terrain that's already baked.
    let water_factor = 1.0 - smoothstep(0.0, 1.0, height);
    let scaled_intensity = intensity * water_factor;

    let brighten_amount = 1.0 + 5.0 * scaled_intensity;
    let brightened = brighten(base, brighten_amount);
    let blue_tint = vec3<f32>(0.08, 0.15, 0.28) * scaled_intensity * 0.5;
    let tinted = vec4<f32>(clamp(brightened.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), brightened.a);
    return vec4<f32>(tinted.rgb, base.a);
}
