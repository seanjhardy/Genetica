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
    return vec2<f32>(rand_11(seed), rand_11(seed + 1));
}