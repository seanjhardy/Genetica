
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
}