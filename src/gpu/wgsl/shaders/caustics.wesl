// Caustic overlay shader (world-bounded)
// CC0 license https://creativecommons.org/share-your-work/public-domain/cc0/

struct CausticsUniform {
    camera_pos_zoom_thickness: vec4<f32>,   // camera_x, camera_y, zoom, line_thickness_px
    view_size_grid: vec4<f32>,              // view_width, view_height, grid_spacing_px, grid_opacity
    grid_threshold_padding: vec4<f32>,      // grid_zoom_threshold, padding...
    border_color: vec4<f32>,
    bounds: vec4<f32>,                      // left, top, right, bottom
    padding: vec4<f32>,
    time_params: vec4<f32>,                 // time, reserved...
}

@group(0) @binding(0)
var planet_texture: texture_2d<f32>;

@group(0) @binding(1)
var height_texture: texture_2d<f32>;

@group(0) @binding(2)
var planet_sampler: sampler;

@group(0) @binding(3)
var<uniform> uniform_data: CausticsUniform;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
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
    let uv = vec2<f32>(
        (pos.x + 1.0) * 0.5,
        1.0 - (pos.y + 1.0) * 0.5
    );
    let bounds = uniform_data.bounds;
    out.uv = uv;
    out.world_pos = vec2<f32>(
        mix(bounds.x, bounds.z, uv.x),
        mix(bounds.y, bounds.w, uv.y)
    );
    return out;
}

// Permutation polynomial hash credit Stefan Gustavson
fn permute(t: vec4<f32>) -> vec4<f32> {
    return t * (t * 34.0 + 133.0);
}

fn mod289(v: vec4<f32>) -> vec4<f32> {
    return v - floor(v / 289.0) * 289.0;
}

fn mod2_vec3(v: vec3<f32>) -> vec3<f32> {
    return v - 2.0 * floor(v / 2.0);
}

// Gradient set is a normalized expanded rhombic dodecahedron
fn grad(hash: f32) -> vec3<f32> {
    let cube = mod2_vec3(floor(hash / vec3<f32>(1.0, 2.0, 4.0))) * 2.0 - 1.0;
    var cuboct = cube;
    let corner = u32(floor(hash / 16.0));
    if (corner == 0u) {
        cuboct.x = 0.0;
    } else if (corner == 1u) {
        cuboct.y = 0.0;
    } else {
        cuboct.z = 0.0;
    }

    let hash8 = i32(floor(hash / 8.0));
    let typ = f32(hash8 & 1);
    let rhomb = (1.0 - typ) * cube + typ * (cuboct + cross(cube, cuboct));
    var g = cuboct * 1.22474487139 + rhomb;
    g *= (1.0 - 0.042942436724648037 * typ) * 3.5946317686139184;
    return g;
}

// BCC lattice split up into 2 cube lattices
fn os2_noise_with_derivatives_part(X: vec3<f32>) -> vec4<f32> {
    let b = floor(X);
    let i4 = vec4<f32>(X - b, 2.5);

    let v1 = b + vec3<f32>(floor(dot(i4, vec4<f32>(0.25))));
    let v2 = b + vec3<f32>(1.0, 0.0, 0.0) + vec3<f32>(-1.0, 1.0, 1.0) * floor(dot(i4, vec4<f32>(-0.25, 0.25, 0.25, 0.35)));
    let v3 = b + vec3<f32>(0.0, 1.0, 0.0) + vec3<f32>(1.0, -1.0, 1.0) * floor(dot(i4, vec4<f32>(0.25, -0.25, 0.25, 0.35)));
    let v4 = b + vec3<f32>(0.0, 0.0, 1.0) + vec3<f32>(1.0, 1.0, -1.0) * floor(dot(i4, vec4<f32>(0.25, 0.25, -0.25, 0.35)));

    var hashes = permute(mod289(vec4<f32>(v1.x, v2.x, v3.x, v4.x)));
    hashes = permute(mod289(hashes + vec4<f32>(v1.y, v2.y, v3.y, v4.y)));
    hashes = mod289(permute(mod289(hashes + vec4<f32>(v1.z, v2.z, v3.z, v4.z))));
    hashes = hashes - floor(hashes / 48.0) * 48.0;

    let d1 = X - v1;
    let d2 = X - v2;
    let d3 = X - v3;
    let d4 = X - v4;
    let a = max(0.75 - vec4<f32>(dot(d1, d1), dot(d2, d2), dot(d3, d3), dot(d4, d4)), vec4<f32>(0.0));
    let aa = a * a;
    let aaaa = aa * aa;
    let g1 = grad(hashes.x);
    let g2 = grad(hashes.y);
    let g3 = grad(hashes.z);
    let g4 = grad(hashes.w);
    let extrapolations = vec4<f32>(dot(d1, g1), dot(d2, g2), dot(d3, g3), dot(d4, g4));

    let derivative = -8.0 * (mat4x3<f32>(d1, d2, d3, d4) * (aa * a * extrapolations))
        + (mat4x3<f32>(g1, g2, g3, g4) * aaaa);

    return vec4<f32>(derivative, dot(aaaa, extrapolations));
}

fn os2_noise_with_derivatives_improve_xy(X: vec3<f32>) -> vec4<f32> {
    let orthonormal_map = mat3x3<f32>(
        0.788675134594813, -0.211324865405187, -0.577350269189626,
        -0.211324865405187, 0.788675134594813, -0.577350269189626,
        0.577350269189626, 0.577350269189626, 0.577350269189626
    );

    let Xr = orthonormal_map * X;
    let result = os2_noise_with_derivatives_part(Xr) + os2_noise_with_derivatives_part(Xr + 144.5);
    let mapped = transpose(orthonormal_map) * result.xyz;
    return vec4<f32>(mapped, result.w);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let bounds = uniform_data.bounds;
    if (in.world_pos.x < bounds.x || in.world_pos.x > bounds.z || in.world_pos.y < bounds.y || in.world_pos.y > bounds.w) {
        discard;
    }

    let bounds_size = vec2<f32>(bounds.z - bounds.x, bounds.w - bounds.y);
    let parallax = 0.0; // 0 = locked to world, 1 = locked to screen
    let parallax_pos = in.world_pos - uniform_data.camera_pos_zoom_thickness.xy * parallax;
    let local = parallax_pos - bounds.xy;
    let uv = 2.0 * (local / bounds_size) * 8.0;

    let time = uniform_data.time_params.x;
    let time_loop = time - floor(time / 578.0) * 578.0;
    let X = vec3<f32>(uv, time_loop * 2.0);

    var noise_result = os2_noise_with_derivatives_improve_xy(X);
    noise_result = os2_noise_with_derivatives_improve_xy(X - noise_result.xyz / 16.0);
    let value = noise_result.w;
    let intensity = clamp(0.6 + 0.8 * value, 0.0, 1.0);
    let sharp = pow(intensity, 2.0); // higher = sharper (try 2.0–6.0)
    return vec4<f32>(vec3<f32>(sharp), 1.0);

}
