// Terrain horizon-based ambient occlusion (HBAO) for heightmaps.
//
// For each texel, sample multiple directions in the heightfield and compute the
// maximum elevation angle (horizon). Occlusion increases with this angle.
//
// Output is a shadow/occlusion factor in out_shadow:
//   1.0 = lit, 0.0 = fully occluded
//
// Performance: O(pixels * steps * ray_count). Intended for "run once" usage.

const WG_X : u32 = 8u;
const WG_Y : u32 = 8u;

// Use vec4-packed fields so host-side layouts stay unambiguous.
//
// dims = (width, height, max_steps, ray_count)
// p0   = (height_scale, bias, strength, step_size_texels)
// p1   = (unused, unused, unused, distance_falloff)
// p2   = (edge_softness_height_units, ray_spread_rad, min_shadow, dist_scale)
struct Params {
    dims : vec4<u32>,
    p0 : vec4<f32>,
    p1 : vec4<f32>,
    p2 : vec4<f32>,
};

@group(0) @binding(0)
var height_tex : texture_2d<f32>;

@group(0) @binding(1)
var height_samp : sampler;

@group(0) @binding(2)
var out_shadow : texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(3)
var<uniform> params : Params;

fn in_bounds(p : vec2<f32>, dims : vec2<f32>) -> bool {
    return (p.x >= 0.0) && (p.y >= 0.0) && (p.x < dims.x) && (p.y < dims.y);
}

fn sample_height_texel_space(p : vec2<f32>, dims : vec2<f32>, height_scale : f32) -> f32 {
    // Sample at texel-center coordinates in normalized UV space.
    let uv = (p + vec2<f32>(0.5, 0.5)) / dims;
    return textureSampleLevel(height_tex, height_samp, uv, 0.0).r * height_scale;
}

fn write_shadow(ix : u32, iy : u32, s : f32) {
    textureStore(out_shadow, vec2<i32>(i32(ix), i32(iy)), vec4<f32>(s, 0.0, 0.0, 1.0));
}

@compute @workgroup_size(WG_X, WG_Y, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width = params.dims.x;
    let height = params.dims.y;
    if (gid.x >= width || gid.y >= height) { return; }

    let dims_f = vec2<f32>(f32(width), f32(height));

    let max_steps = max(params.dims.z, 1u);
    let ray_count = max(params.dims.w, 1u);

    let height_scale = params.p0.x;
    let bias = params.p0.y;
    let strength = params.p0.z;
    let step_size = max(params.p0.w, 0.25);

    let falloff = max(params.p1.w, 0.0);

    let edge_soft = max(params.p2.x, 1.0e-4);
    let ray_spread = max(params.p2.y, 0.0);
    let min_shadow = clamp(params.p2.z, 0.0, 1.0);
    let dist_scale = max(params.p2.w, 0.0);

    let p0 = vec2<f32>(f32(gid.x), f32(gid.y));
    let h0 = sample_height_texel_space(p0, dims_f, height_scale);

    var sum_occ : f32 = 0.0;
    var sum_w : f32 = 0.0;

    for (var ri : u32 = 0u; ri < ray_count; ri = ri + 1u) {
        let t = (f32(ri) + 0.5) / f32(ray_count);
        let ang = t * 6.2831853 * max(ray_spread, 1.0);
        let dir = vec2<f32>(cos(ang), sin(ang));

        var max_angle : f32 = 0.0;

        let steps = max_steps;

        for (var si : u32 = 1u; si <= steps; si = si + 1u) {
            let dist = f32(si) * step_size;
            let p = p0 + dir * dist;
            if (!in_bounds(p, dims_f)) { break; }

            let terrain_h = sample_height_texel_space(p, dims_f, height_scale);
            let dh = (terrain_h - h0) - bias;
            let angle = atan2(max(dh, 0.0), max(dist, 1.0e-4));
            let atten = exp(-falloff * dist) / (1.0 + dist * dist_scale);
            max_angle = max(max_angle, angle * atten);
        }
        let occ = clamp(max_angle / 1.5707963, 0.0, 1.0);
        sum_occ = sum_occ + occ;
        sum_w = sum_w + 1.0;
    }

    let occlusion = clamp((sum_occ / max(sum_w, 1.0)) * strength, 0.0, 1.0);
    let shadow_out = max(1.0 - occlusion, min_shadow);
    write_shadow(gid.x, gid.y, shadow_out);
}
