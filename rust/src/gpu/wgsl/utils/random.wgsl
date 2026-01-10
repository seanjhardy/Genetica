fn randFloat(seed: u32) -> f32 {
    var x = seed;
    x = (x ^ 61u) ^ (x >> 16u);
    x *= 9u;
    x = x ^ (x >> 4u);
    x *= 0x27d4eb2du;
    x = x ^ (x >> 15u);
    return f32(x) / 4294967296.0;
}

fn f32_to_u32_bits(x: f32) -> u32 {
    // Extract components of the IEEE 754 float
    let sign = select(0u, 1u, x < 0.0);
    let abs_x = abs(x);
    
    // Get exponent (biased by 127 for f32)
    let exp = select(0u, u32(floor(log2(abs_x))) + 127u, abs_x > 0.0);
    
    // Get mantissa (23 bits precision)
    let mantissa = select(0u, u32((abs_x / exp2(f32(exp) - 127.0) - 1.0) * 8388608.0), abs_x > 0.0);
    
    // Combine into a single u32: sign(1) + exp(8) + mantissa(23)
    return (sign << 31u) | ((exp & 255u) << 23u) | (mantissa & 8388607u);
}

fn randRange(seed: u32, min_val: f32, max_val: f32) -> f32 {
    let r = randFloat(seed); // [0.0, 1.0)
    return min_val + r * (max_val - min_val);
}

fn rand_01(seed: f32) -> f32 {
    return randFloat(f32_to_u32_bits(seed));
}

fn rand_11(seed: f32) -> f32 {
    return randRange(f32_to_u32_bits(seed), -1.0, 1.0);
}

fn rand_vec2(seed: f32) -> vec2<f32> {
    return vec2<f32>(rand_11(seed), rand_11(seed + 5.0));
}

// Seamless ring noise: sample 2D Perlin along a circle.
// fBm is done by increasing angular frequency per octave, with random phase per octave.
// This gives a perfectly closed loop (no seam) and avoids "all peaking together".

const TAU: f32 = 6.28318530717958647692;

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + t * (b - a);
}

// Hash -> u32
fn hash_u32(x: u32) -> u32 {
    var v = x;
    v ^= v >> 16u;
    v *= 0x7FEB352Du;
    v ^= v >> 15u;
    v *= 0x846CA68Bu;
    v ^= v >> 16u;
    return v;
}

// Hash 2D integer coords + seed
fn hash2(ix: i32, iy: i32, seed: u32) -> u32 {
    let ux = bitcast<u32>(ix);
    let uy = bitcast<u32>(iy);
    return hash_u32(ux ^ hash_u32(uy ^ (seed * 0x9E3779B9u)));
}

// Hash -> [0,1)
fn hash01(x: u32) -> f32 {
    let h = hash_u32(x);
    return f32(h & 0x00FFFFFFu) / f32(0x01000000u);
}

// 8-direction gradients
fn grad2_from_hash(h: u32) -> vec2<f32> {
    let idx: u32 = h & 7u;
    switch idx {
        case 0u: { return vec2<f32>( 1.0,  0.0); }
        case 1u: { return vec2<f32>(-1.0,  0.0); }
        case 2u: { return vec2<f32>( 0.0,  1.0); }
        case 3u: { return vec2<f32>( 0.0, -1.0); }
        case 4u: { return vec2<f32>( 0.70710678,  0.70710678); }
        case 5u: { return vec2<f32>(-0.70710678,  0.70710678); }
        case 6u: { return vec2<f32>( 0.70710678, -0.70710678); }
        default: { return vec2<f32>(-0.70710678, -0.70710678); }
    }
}

// 2D Perlin gradient noise ~[-1,1]
fn perlin2(p: vec2<f32>, seed: u32) -> f32 {
    let base = floor(p);
    let pi = vec2<i32>(i32(base.x), i32(base.y));
    let f = p - base;

    let p00 = pi;
    let p10 = pi + vec2<i32>(1, 0);
    let p01 = pi + vec2<i32>(0, 1);
    let p11 = pi + vec2<i32>(1, 1);

    let g00 = grad2_from_hash(hash2(p00.x, p00.y, seed));
    let g10 = grad2_from_hash(hash2(p10.x, p10.y, seed));
    let g01 = grad2_from_hash(hash2(p01.x, p01.y, seed));
    let g11 = grad2_from_hash(hash2(p11.x, p11.y, seed));

    let d00 = f - vec2<f32>(0.0, 0.0);
    let d10 = f - vec2<f32>(1.0, 0.0);
    let d01 = f - vec2<f32>(0.0, 1.0);
    let d11 = f - vec2<f32>(1.0, 1.0);

    let v00 = dot(g00, d00);
    let v10 = dot(g10, d10);
    let v01 = dot(g01, d01);
    let v11 = dot(g11, d11);

    let u = vec2<f32>(fade(f.x), fade(f.y));
    let x0 = lerp(v00, v10, u.x);
    let x1 = lerp(v01, v11, u.x);
    let v  = lerp(x0,  x1,  u.y);

    // scale to use more of [-1,1]
    return 1.8 * v;
}

// Seamless ring fBm:
// - t in [0,1)
// - each octave uses a different phase offset so peaks don't align
// - smoothness is controlled by "radius": smaller radius => smoother changes between samples
fn ring_fbm(
    t: f32,
    seed: u32,
    amplitude: f32,
    frequency: f32,   // cycles around the ring for octave 0
    octaves: u32,
    lacunarity: f32,  // frequency multiplier per octave (typical 2.0)
    gain: f32,        // amplitude multiplier per octave (typical 0.5)
    radius: f32       // key smoothness control (try 0.15..0.6)
) -> f32 {
    var sum: f32 = 0.0;
    var amp: f32 = 1.0;
    var freq: f32 = frequency;

    let max_oct: u32 = min(octaves, 10u);

    // Add a constant offset so the circle isn't centered exactly on an integer lattice corner
    // (not required, but it avoids some "structured" cases).
    let offset = vec2<f32>(12.345, -67.89);

    for (var o: u32 = 0u; o < max_oct; o = o + 1u) {
        // Random phase per octave in [0,1)
        let phase: f32 = hash01(seed ^ (o * 0xA511E9B3u));

        // Angle around the ring; phase shifts the wave so octaves don't line up
        let ang: f32 = TAU * (t * freq + phase);

        // Closed loop in 2D => perfectly seamless at wrap
        let p: vec2<f32> = offset + radius * vec2<f32>(cos(ang), sin(ang));

        sum += amp * perlin2(p, seed + o * 101u);

        freq *= lacunarity;
        amp *= gain;
    }

    return sum * amplitude;
}

// Generate 20 samples around the circle.
// IMPORTANT: value[0] and value[19] will generally NOT be equal (they're 18° apart),
// but the step 19->0 will be as smooth as any other step when radius/frequency are sensible.
fn generate_noise20_circle(
    seed: u32,
    amplitude: f32,
    frequency: f32,
    octaves: u32
) -> array<f32, 20> {
    var out: array<f32, 20>;

    let lacunarity: f32 = 2.0;
    let gain: f32 = 0.5;

    // This is the big one:
    // smaller radius => smoother gradients between adjacent samples.
    // Start with 0.25 for "roughly smooth" with 20 points.
    let radius: f32 = 0.25;

    for (var i: u32 = 0u; i < 20u; i = i + 1u) {
        let t: f32 = f32(i) / 20.0; // samples around the ring
        out[i] = ring_fbm(t, seed, amplitude, frequency, octaves, lacunarity, gain, radius);
    }

    return out;
}
