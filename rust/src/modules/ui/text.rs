// Text rendering UI component

use crate::modules::math::Vec2;

/// UI text overlay information
pub struct TextOverlay {
    pub text: String,
    pub position: Vec2,
    pub color: [f32; 4], // RGBA
}

impl TextOverlay {
    pub fn new(text: String, position: Vec2, color: [f32; 4]) -> Self {
        Self {
            text,
            position,
            color,
        }
    }
}

/// UI state for overlays and text
pub struct UiState {
    pub framerate: f32,
    pub step: u64,
    pub text_overlays: Vec<TextOverlay>,
}

impl UiState {
    pub fn new() -> Self {
        Self {
            framerate: 0.0,
            step: 0,
            text_overlays: Vec::new(),
        }
    }

    pub fn update(&mut self, framerate: f32, step: u64) {
        self.framerate = framerate;
        self.step = step;
        self.update_text_overlays();
    }

    fn update_text_overlays(&mut self) {
        self.text_overlays.clear();
        
        // Framerate display in top left
        let fps_text = format!("FPS: {:.1}", self.framerate);
        self.text_overlays.push(TextOverlay::new(
            fps_text,
            Vec2::new(10.0, 10.0),
            [1.0, 1.0, 1.0, 1.0], // White
        ));
        
        // Step counter below framerate
        let step_text = format!("Step: {}", self.step);
        self.text_overlays.push(TextOverlay::new(
            step_text,
            Vec2::new(10.0, 40.0),
            [1.0, 1.0, 1.0, 1.0], // White
        ));
    }
}

