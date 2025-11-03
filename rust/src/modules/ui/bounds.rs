// Bounds border rendering

use crate::modules::math::{Rect, Vec2};

/// Bounds border for visualization
pub struct BoundsBorder {
    pub bounds: Rect,
    pub color: [f32; 4], // RGBA
    pub line_width: f32,
}

impl BoundsBorder {
    pub fn new(bounds: Rect) -> Self {
        Self {
            bounds,
            color: [0.0, 1.0, 0.0, 1.0], // Green
            line_width: 2.0,
        }
    }

    /// Get the 4 corners of the bounds rectangle for rendering
    pub fn get_corners(&self) -> [Vec2; 4] {
        [
            Vec2::new(self.bounds.left, self.bounds.top),
            Vec2::new(self.bounds.right(), self.bounds.top),
            Vec2::new(self.bounds.right(), self.bounds.bottom()),
            Vec2::new(self.bounds.left, self.bounds.bottom()),
        ]
    }

    /// Update bounds
    pub fn set_bounds(&mut self, bounds: Rect) {
        self.bounds = bounds;
    }
}

