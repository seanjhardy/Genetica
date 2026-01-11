// General inheritance system for UI properties
// Simple approach: just use a struct with Option fields for inheritable properties

use super::styles::Color;

/// Properties that can be inherited from parent to child components
/// To add a new inheritable property, just add it here and use it in the parser
#[derive(Debug, Clone, Default)]
pub struct InheritableProperties {
    pub text_color: Option<Color>,
    pub font_size: Option<f32>,
    // Add more inheritable properties here as Option fields
    // Example: pub line_height: Option<f32>,
}

impl InheritableProperties {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new set of properties inheriting from this one
    pub fn inherit_from(&self) -> Self {
        self.clone()
    }
}
