// Distortion utility functions for texture sampling effects

// Apply fisheye distortion to UV coordinates
// uv: normalized coordinates from -1 to 1 (centered)
// strength: how strong the fisheye effect is (0.0 = no distortion, higher = more bulging)
fn fisheye_distortion(uv: vec2<f32>, strength: f32) -> vec2<f32> {
    let dist_from_center = length(uv);
    let distortion_radius = strength * dist_from_center * dist_from_center;
    return uv + uv * distortion_radius;
}

// Alternative: radial distortion (more subtle than fisheye)
fn radial_distortion(uv: vec2<f32>, strength: f32) -> vec2<f32> {
    let dist_from_center = length(uv);
    let distortion = strength * dist_from_center;
    return uv + uv * distortion;
}