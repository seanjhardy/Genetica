
#[derive(Debug, Clone)]
pub struct Text {
    pub content: String,
    pub font_size: f32,
    pub color: super::Color,
    pub text_align: super::TextAlign,
    // Cached bounds to avoid recalculating every frame
    cached_width: f32,
    cached_height: f32,
}

impl Text {
    pub fn new(content: String) -> Self {
        let font_size = 16.0;
        let (cached_width, cached_height) = crate::utils::gpu::text_renderer::calculate_text_bounds(&content, font_size);
        Self {
            content,
            font_size,
            color: super::Color::black(),
            text_align: super::TextAlign::Left,
            cached_width,
            cached_height,
        }
    }

    /// Update cached text bounds - only call when content or font_size changes
    pub fn update_bounds(&mut self) {
        let (width, height) = crate::utils::gpu::text_renderer::calculate_text_bounds(&self.content, self.font_size);
        self.cached_width = width;
        self.cached_height = height;
    }

    /// Get cached width
    pub fn cached_width(&self) -> f32 {
        self.cached_width
    }

    /// Get cached height
    pub fn cached_height(&self) -> f32 {
        self.cached_height
    }

    /// Set font size and update cached bounds
    pub fn set_font_size(&mut self, font_size: f32) -> bool {
        if self.font_size != font_size {
            self.font_size = font_size;
            self.update_bounds();
            return true; // Font size changed
        }
        false // No change
    }
}