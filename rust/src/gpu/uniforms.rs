// Uniforms structure for GPU shader data

use bytemuck;

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug)]
pub struct Uniforms {
    pub delta_time: f32, // 4 bytes at offset 0
    _padding1: f32, // 4 bytes at offset 4
    _padding2: f32, // 4 bytes at offset 8
    _padding3: f32, // 4 bytes at offset 12 (totals 16 bytes to align vec2)
    pub camera_pos: [f32; 2], // 8 bytes at offset 16
    pub zoom: f32, // 4 bytes at offset 24
    pub point_radius: f32, // 4 bytes at offset 28
    pub bounds: [f32; 4], // 16 bytes at offset 32
    pub view_size: [f32; 2], // 8 bytes at offset 48
    _padding5: [f32; 2], // 8 bytes at offset 56 to reach 64 bytes
    _final_padding: [f32; 4], // 16 bytes at offset 64 to reach 80 bytes
}

// Manually implement Pod and Zeroable since we have explicit padding
// This is safe because the struct has #[repr(C)] and explicit padding fields
unsafe impl bytemuck::Pod for Uniforms {}
unsafe impl bytemuck::Zeroable for Uniforms {}

impl Uniforms {
    pub fn new(
        delta_time: f32,
        camera_pos: [f32; 2],
        zoom: f32,
        point_radius: f32,
        left: f32,
        top: f32,
        right: f32,
        bottom: f32,
        view_width: f32,
        view_height: f32,
    ) -> Self {
        Self {
            delta_time,
            _padding1: 0.0,
            _padding2: 0.0,
            _padding3: 0.0,
            camera_pos,
            zoom,
            point_radius,
            bounds: [left, top, right, bottom],
            view_size: [view_width, view_height],
            _padding5: [0.0, 0.0],
            _final_padding: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

