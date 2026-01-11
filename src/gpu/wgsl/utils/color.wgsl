fn brighten(color: vec4<f32>, amount: f32) -> vec4<f32> {
  let r = clamp(color.r * amount, 0.0, 1.0);
  let g = clamp(color.g * amount, 0.0, 1.0);
  let b = clamp(color.b * amount, 0.0, 1.0);
  return vec4<f32>(r, g, b, color.a);
}

fn alpha(rgba: vec4<f32>, a: f32) -> vec4<f32> {
    return vec4<f32>(rgba.rgb, clamp(a, 0.0, 1.0));
}

fn over(bg: vec4<f32>, fg: vec4<f32>) -> vec4<f32> {
    let a = fg.a + bg.a * (1.0 - fg.a);
    let rgb = (fg.rgb * fg.a + bg.rgb * bg.a * (1.0 - fg.a)) / a;
    return vec4<f32>(rgb, a);
}

fn rgb_to_hsl(rgb: vec4<f32>) -> vec4<f32> {
    let maxc = max(max(rgb.r, rgb.g), rgb.b);
    let minc = min(min(rgb.r, rgb.g), rgb.b);
    let l = (maxc + minc) * 0.5;
    let delta = maxc - minc;

    var h: f32 = 0.0;
    var s: f32 = 0.0;

    if (delta > 0.00001) {
        s = delta / (1.0 - abs(2.0 * l - 1.0));

        if (maxc == rgb.r) {
            h = ((rgb.g - rgb.b) / delta) % 6.0;
        } else if (maxc == rgb.g) {
            h = ((rgb.b - rgb.r) / delta) + 2.0;
        } else {
            h = ((rgb.r - rgb.g) / delta) + 4.0;
        }
        h = h / 6.0;
        if (h < 0.0) {
            h = h + 1.0;
        }
    }

    return vec4<f32>(h, s, l, rgb.a);
}

fn hsl_to_rgb(hsl: vec4<f32>) -> vec4<f32> {
    let h = hsl.x;
    let s = hsl.y;
    let l = hsl.z;

    let c = (1.0 - abs(2.0 * l - 1.0)) * s;
    let hp = h * 6.0;
    let x = c * (1.0 - abs((hp % 2.0) - 1.0));

    var r: f32 = 0.0;
    var g: f32 = 0.0;
    var b: f32 = 0.0;

    if (0.0 <= hp && hp < 1.0) {
        r = c; g = x; b = 0.0;
    } else if (1.0 <= hp && hp < 2.0) {
        r = x; g = c; b = 0.0;
    } else if (2.0 <= hp && hp < 3.0) {
        r = 0.0; g = c; b = x;
    } else if (3.0 <= hp && hp < 4.0) {
        r = 0.0; g = x; b = c;
    } else if (4.0 <= hp && hp < 5.0) {
        r = x; g = 0.0; b = c;
    } else {
        r = c; g = 0.0; b = x;
    }

    let m = l - 0.5 * c;
    return vec4<f32>(r + m, g + m, b + m, hsl.a);
}

fn saturate(rgb: vec4<f32>, sat_add: f32) -> vec4<f32> {
    var hsl = rgb_to_hsl(rgb);
    hsl.y = hsl.y + (sat_add - 1.0);
    return hsl_to_rgb(hsl);
}


fn srgb(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(srgb_to_linear(color.r), srgb_to_linear(color.g), srgb_to_linear(color.b), color.a);
}
fn srgb_to_linear(c: f32) -> f32 {
        if c <= 0.04045 {
        return c / 12.92;
    } else {
        return pow((c + 0.055) / 1.055, 2.4);
    }
}