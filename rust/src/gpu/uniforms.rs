// Uniforms structure for GPU shader data

use bytemuck;

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug)]
pub struct Uniforms {
    /// (delta_time, zoom, view_width, view_height)
    pub sim_params: [f32; 4],
    /// (cell_count, reserved0, reserved1, reserved2)
    pub cell_count: [f32; 4],
    /// (camera_x, camera_y, reserved0, reserved1)
    pub camera: [f32; 4],
    /// (bounds_left, bounds_top, bounds_right, bounds_bottom)
    pub bounds: [f32; 4],
}

// Manually implement Pod and Zeroable since we have explicit padding
// This is safe because the struct has #[repr(C)] and explicit padding fields
unsafe impl bytemuck::Pod for Uniforms {}
unsafe impl bytemuck::Zeroable for Uniforms {}

impl Uniforms {
    /// Create a zeroed uniform (for buffer initialization)
    pub fn zeroed() -> Self {
        Self {
            sim_params: [0.0, 1.0, 0.0, 0.0],
            cell_count: [0.0, 0.0, 0.0, 0.0],
            camera: [0.0, 0.0, 0.0, 0.0],
            bounds: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

impl Uniforms {
    pub fn new(
        delta_time: f32,
        camera_pos: [f32; 2],
        zoom: f32,
        left: f32,
        top: f32,
        right: f32,
        bottom: f32,
        view_width: f32,
        view_height: f32,
        cell_count: f32,
    ) -> Self {
        Self {
            sim_params: [delta_time, zoom, view_width, view_height],
            cell_count: [cell_count, 0.0, 0.0, 0.0],
            camera: [camera_pos[0], camera_pos[1], 0.0, 0.0],
            bounds: [left, top, right, bottom],
        }
    }
}

