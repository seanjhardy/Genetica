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

// Water wave distortion in world space
// world_pos: position in world coordinates
// time: time parameter for animation
// Returns offset to apply to world position
// Generate subtle water wave distortion
fn water_distortion(world_pos: vec2<f32>, time: f32) -> vec2<f32> {
    let scaled_pos = world_pos * 0.002;
    let speed = 10.0;
    
    // Multiple sine waves at different frequencies and directions for natural look
    let wave1 = sin(scaled_pos.x * 40.0 + time * 1.0 * speed) * cos(scaled_pos.y * 30.0 + time * 0.3 * speed);
    let wave2 = sin(scaled_pos.x * 24.0 - time * 0.6 * speed) * cos(scaled_pos.y * 35.0 - time * 0.8 * speed);
    let wave3 = sin((scaled_pos.x + scaled_pos.y) * 20.0 + time * 0.2 * speed);
    
    // Combine waves with different weights for complexity
    let distortion_x = (wave1 * 0.4 + wave2 * 0.3 + wave3 * 0.2) * 1.0;
    let distortion_y = (wave2 * 0.4 + wave1 * 0.3 - wave3 * 0.2) * 1.0;
    
    return vec2<f32>(distortion_x, distortion_y);
}

