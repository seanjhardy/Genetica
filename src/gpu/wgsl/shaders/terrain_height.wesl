// Terrain heightmap generator (static)

struct TerrainHeightUniforms {
    seed: u32,
    base_frequency: f32,
    octave_count: u32,
    _padding0: u32,
    frequency_falloff: f32,
    amplitude_falloff: f32,
    floor_height: f32,
    _padding1: f32,
    offset: vec2<f32>,
    resolution: vec2<f32>,
    warp_frequency: f32,
    warp_strength: f32,
    warp_octave_count: u32,
    _padding2: u32,
    warp_frequency_falloff: f32,
    warp_amplitude_falloff: f32,
    _padding3: f32,
    _padding4: f32,
    band_min: f32,
    band_max: f32,
    band_steps: u32,
    _padding5: u32,
    detail_frequency_mult: f32,
    detail_amplitude: f32,
    _padding6: f32,
    _padding7: f32,
}

@group(0) @binding(0)
var<uniform> uniforms: TerrainHeightUniforms;

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

fn get_permutation(index: i32) -> i32 {
    let i = index & 255;
    if i < 32 {
        let values = array<i32, 32>(
            151, 160, 137, 91, 90, 15, 131, 13,
            201, 95, 96, 53, 194, 233, 7, 225,
            140, 36, 103, 30, 69, 142, 8, 99,
            37, 240, 21, 10, 23, 190, 6, 148
        );
        return values[i];
    } else if i < 64 {
        let values = array<i32, 32>(
            247, 120, 234, 75, 0, 26, 197, 62,
            94, 252, 219, 203, 117, 35, 11, 32,
            57, 177, 33, 88, 237, 149, 56, 87,
            174, 20, 125, 136, 171, 168, 68, 175
        );
        return values[i - 32];
    } else if i < 96 {
        let values = array<i32, 32>(
            74, 165, 71, 134, 139, 48, 27, 166,
            77, 146, 158, 231, 83, 111, 229, 122,
            60, 211, 133, 230, 220, 105, 92, 41,
            55, 46, 245, 40, 244, 102, 143, 54
        );
        return values[i - 64];
    } else if i < 128 {
        let values = array<i32, 32>(
            65, 25, 63, 161, 1, 216, 80, 73,
            209, 76, 132, 187, 208, 89, 18, 169,
            200, 196, 135, 130, 116, 188, 159, 86,
            164, 100, 109, 198, 173, 186, 3, 64
        );
        return values[i - 96];
    } else if i < 160 {
        let values = array<i32, 32>(
            52, 217, 226, 250, 124, 123, 5, 202,
            38, 147, 118, 126, 255, 82, 85, 212,
            207, 206, 59, 227, 47, 16, 58, 17,
            182, 189, 28, 42, 223, 183, 170, 213
        );
        return values[i - 128];
    } else if i < 192 {
        let values = array<i32, 32>(
            119, 248, 152, 2, 44, 154, 163, 70,
            221, 153, 101, 155, 167, 43, 172, 9,
            129, 22, 39, 253, 19, 98, 108, 110,
            79, 113, 224, 232, 178, 185, 112, 104
        );
        return values[i - 160];
    } else if i < 224 {
        let values = array<i32, 32>(
            218, 246, 97, 228, 251, 34, 242, 193,
            238, 210, 144, 12, 191, 179, 162, 241,
            81, 51, 145, 235, 249, 14, 239, 107,
            49, 192, 214, 31, 181, 199, 106, 157
        );
        return values[i - 192];
    } else {
        let values = array<i32, 32>(
            184, 84, 204, 176, 115, 121, 50, 45,
            127, 4, 150, 254, 138, 236, 205, 93,
            222, 114, 67, 29, 24, 72, 243, 141,
            128, 195, 78, 66, 215, 61, 156, 180
        );
        return values[i - 224];
    }
}

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn grad(hash: i32, x: f32, y: f32, z: f32) -> f32 {
    let h = hash & 15;
    let u = select(y, x, h < 8);
    let v = select(select(z, x, h == 12 || h == 14), y, h < 4);
    let sign_u = select(-1.0, 1.0, (h & 1) == 0);
    let sign_v = select(-1.0, 1.0, (h & 2) == 0);
    return sign_u * u + sign_v * v;
}

fn noise3D(x: f32, y: f32, z: f32) -> f32 {
    let _x = floor(x);
    let _y = floor(y);
    let _z = floor(z);
    let seed_x = i32(uniforms.seed & 255u);
    let seed_y = i32((uniforms.seed >> 8u) & 255u);
    let seed_z = i32((uniforms.seed >> 16u) & 255u);
    let ix = (i32(_x) + seed_x) & 255;
    let iy = (i32(_y) + seed_y) & 255;
    let iz = (i32(_z) + seed_z) & 255;
    let fx = x - _x;
    let fy = y - _y;
    let fz = z - _z;
    let u = fade(fx);
    let v = fade(fy);
    let w = fade(fz);
    let A = (get_permutation(ix) + iy) & 255;
    let B = (get_permutation(ix + 1) + iy) & 255;
    let AA = (get_permutation(A) + iz) & 255;
    let AB = (get_permutation(A + 1) + iz) & 255;
    let BA = (get_permutation(B) + iz) & 255;
    let BB = (get_permutation(B + 1) + iz) & 255;
    let p0 = grad(get_permutation(AA), fx, fy, fz);
    let p1 = grad(get_permutation(BA), fx - 1.0, fy, fz);
    let p2 = grad(get_permutation(AB), fx, fy - 1.0, fz);
    let p3 = grad(get_permutation(BB), fx - 1.0, fy - 1.0, fz);
    let p4 = grad(get_permutation(AA + 1), fx, fy, fz - 1.0);
    let p5 = grad(get_permutation(BA + 1), fx - 1.0, fy, fz - 1.0);
    let p6 = grad(get_permutation(AB + 1), fx, fy - 1.0, fz - 1.0);
    let p7 = grad(get_permutation(BB + 1), fx - 1.0, fy - 1.0, fz - 1.0);
    let q0 = mix(p0, p1, u);
    let q1 = mix(p2, p3, u);
    let q2 = mix(p4, p5, u);
    let q3 = mix(p6, p7, u);
    let r0 = mix(q0, q1, v);
    let r1 = mix(q2, q3, v);
    return mix(r0, r1, w);
}

fn octave_noise(tex_coord: vec3<f32>) -> f32 {
    var total = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    var amplitude_sum = 0.0;
    let seed_scalar = f32(uniforms.seed & 0x00FFFFFFu) / 16777216.0;
    let seed_offset = vec3<f32>(
        seed_scalar * 997.0,
        seed_scalar * 1319.0,
        seed_scalar * 2333.0,
    );

    var octave = 0u;
    loop {
        if (octave >= uniforms.octave_count) {
            break;
        }
        let sample_coord = tex_coord * frequency + seed_offset;
        let sample = noise3D(sample_coord.x, sample_coord.y, sample_coord.z);
        total += sample * amplitude;
        amplitude_sum += amplitude;
        frequency *= uniforms.frequency_falloff;
        amplitude *= uniforms.amplitude_falloff;
        octave += 1u;
    }
    if (amplitude_sum == 0.0) {
        return 0.0;
    }
    return total / amplitude_sum;
}

fn warp_noise(tex_coord: vec3<f32>) -> f32 {
    if (uniforms.warp_octave_count == 0u) {
        return 0.0;
    }
    var total = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    var amplitude_sum = 0.0;
    let seed_scalar = f32(uniforms.seed & 0x00FFFFFFu) / 16777216.0;
    let seed_offset = vec3<f32>(seed_scalar * 123.0, seed_scalar * 456.0, seed_scalar * 789.0);

    var octave = 0u;
    loop {
        if (octave >= uniforms.warp_octave_count) {
            break;
        }
        let sample_coord = tex_coord * frequency + seed_offset;
        let sample = noise3D(sample_coord.x, sample_coord.y, sample_coord.z);
        total += sample * amplitude;
        amplitude_sum += amplitude;
        frequency *= uniforms.warp_frequency_falloff;
        amplitude *= uniforms.warp_amplitude_falloff;
        octave += 1u;
    }
    if (amplitude_sum == 0.0) {
        return 0.0;
    }
    return total / amplitude_sum;
}

fn warp_coord(coord: vec2<f32>) -> vec2<f32> {
    if (uniforms.warp_strength == 0.0 || uniforms.warp_frequency == 0.0) {
        return coord;
    }
    let base = vec3<f32>(coord * uniforms.warp_frequency, 0.0);
    let warp_x = warp_noise(base);
    let warp_y = warp_noise(base + vec3<f32>(37.2, 17.1, 11.3));
    return coord + vec2<f32>(warp_x, warp_y) * uniforms.warp_strength;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let world = uniforms.offset + in.uv * uniforms.resolution;
    let skew = vec2<f32>(
        world.x + world.y * 0.31,
        world.y - world.x * 0.27
    );
    let coord = warp_coord(skew * uniforms.base_frequency);
    // Banded base layer (hardcoded, bespoke).
    let base_noise = octave_noise(vec3<f32>(coord, 0.0));
    let base_norm = clamp(base_noise * 2.0 + 0.5, 0.0, 1.0);
    let band_min = -0.20;
    let band_max = 0.998;
    let band_steps = 5u; // higher band frequency
    let band_range = band_max - band_min;
    let band_step = band_range / f32(band_steps);
    let band_index = clamp(
        i32(floor((base_norm - band_min) / max(band_step, 1.0e-6))),
        0,
        i32(band_steps) - 1
    );
    let band_index_f = f32(band_index);
    let lower = clamp(band_min + band_index_f * band_step, band_min, band_max);
    let upper = clamp(lower + band_step, band_min, band_max);
    let band_t = clamp((base_norm - lower) / max(band_step, 1.0e-6), 0.0, 1.0);
    // Soften shelf edges by blending near the top of each band.
    let band_blend = 0.1;
    let band_edge = smoothstep(1.0 - band_blend, 1.0, band_t);
    let banded = mix(lower, upper, band_edge);
    let banded_norm = pow(banded, 2);

    // Per-band detail noise: seed varies by band index to avoid cross-band continuity.
    let seed_scalar = f32(uniforms.seed & 0x00FFFFFFu) / 16777216.0;
    let band_seed = seed_scalar * 1000.0 + band_index_f * 131.5;
    let band_seed_next = seed_scalar * 1000.0 + (band_index_f + 1.0) * 131.5;
    let band_offset = vec3<f32>(band_seed, band_seed * 1.37, band_seed * 1.73);
    let band_offset_next = vec3<f32>(band_seed_next, band_seed_next * 1.37, band_seed_next * 1.73);
    let detail_coord = coord * 4.5 + vec2<f32>(band_index_f, -band_index_f) * 11.3;
    let detail = octave_noise(vec3<f32>(detail_coord, 0.0) + band_offset_next) * 0.5;
    let height = clamp(banded_norm + detail, 0.0, 1.0);
    let clamped = max(height, uniforms.floor_height);
    
    return vec4<f32>(clamped, clamped, clamped, 1.0);
}
