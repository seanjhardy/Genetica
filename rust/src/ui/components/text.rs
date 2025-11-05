
#[derive(Debug, Clone)]
pub struct Text {
    pub content: String,
    pub font_size: f32,
    pub color: super::Color,
}

impl Text {
    pub fn new(content: String) -> Self {
        Self {
            content,
            font_size: 16.0,
            color: super::Color::black(),
        }
    }

    pub fn with_font_size(mut self, size: f32) -> Self {
        self.font_size = size;
        self
    }

    pub fn with_color(mut self, color: super::Color) -> Self {
        self.color = color;
        self
    }
}