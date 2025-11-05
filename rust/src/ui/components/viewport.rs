// Forward declarations
#[derive(Debug, Clone)]
pub struct Viewport {
    pub texture: Option<wgpu::TextureView>,
}

impl Viewport {
    pub fn new() -> Self {
        Self {
            texture: None,
        }
    }
}