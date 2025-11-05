use wgpu;

// Forward declarations
#[derive(Debug, Clone)]
pub struct Viewport {
    pub texture: Option<wgpu::Texture>,
    pub texture_view: Option<wgpu::TextureView>,
    pub width: u32,
    pub height: u32,
}

impl Viewport {
    pub fn new() -> Self {
        Self {
            texture: None,
            texture_view: None,
            width: 0,
            height: 0,
        }
    }
}