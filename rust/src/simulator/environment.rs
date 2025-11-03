// Environment module - manages bounds, points, and physics simulation

use crate::modules::math::{Rect, Vec2};
use crate::modules::ui::DragHandler;

/// Point structure for physics simulation
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Point {
    pub pos: [f32; 2],
    pub prev_pos: [f32; 2],
    pub velocity: [f32; 2],
}

impl Point {
    pub fn new(pos: [f32; 2], prev_pos: [f32; 2], velocity: [f32; 2]) -> Self {
        Self {
            pos,
            prev_pos,
            velocity,
        }
    }
}

/// Environment manages the simulation bounds, points, and their physics
pub struct Environment {
    bounds: Rect,
    drag_handler: DragHandler,
    num_points: usize,
}

impl Environment {
    pub fn new(initial_bounds: Rect, num_points: usize) -> Self {
        Self {
            bounds: initial_bounds,
            drag_handler: DragHandler::new(),
            num_points,
        }
    }

    /// Initialize points with random positions within bounds
    pub fn initialize_points(&self) -> Vec<Point> {
        let mut points = Vec::with_capacity(self.num_points);
        let center_x = self.bounds.left + self.bounds.width / 2.0;
        let center_y = self.bounds.top + self.bounds.height / 2.0;

        for i in 0..self.num_points {
            let angle = (i as f32) * 0.1;
            let radius = 50.0 + (i as f32 % 100.0);
            let pos = [
                center_x + radius * angle.cos(),
                center_y + radius * angle.sin(),
            ];
            
            points.push(Point::new(pos, pos, [0.0, 0.0]));
        }

        points
    }

    /// Update environment state (bounds dragging, etc.)
    pub fn update(&mut self, mouse_world_pos: Vec2, zoom: f32, ui_hovered: bool) {
        if !ui_hovered {
            let sensitivity = 15.0 / zoom;
            let delta_bounds = self.drag_handler.update(mouse_world_pos, self.bounds, sensitivity);

            if delta_bounds.width != 0.0
                || delta_bounds.height != 0.0
                || delta_bounds.left != 0.0
                || delta_bounds.top != 0.0
            {
                // Round bounds to grid (20px grid)
                let new_left = (self.bounds.left + delta_bounds.left) / 20.0;
                let new_top = (self.bounds.top + delta_bounds.top) / 20.0;
                let new_width = (self.bounds.width + delta_bounds.width) / 20.0;
                let new_height = (self.bounds.height + delta_bounds.height) / 20.0;

                self.bounds = Rect::new(
                    new_left.round() * 20.0,
                    new_top.round() * 20.0,
                    new_width.round() * 20.0,
                    new_height.round() * 20.0,
                );
            }
        } else {
            self.drag_handler.reset();
        }
    }

    /// Handle mouse press events
    pub fn handle_mouse_press(&mut self, pressed: bool) {
        self.drag_handler.handle_mouse_press(pressed);
    }

    /// Get current bounds
    pub fn get_bounds(&self) -> Rect {
        self.bounds
    }

    /// Set bounds
    pub fn set_bounds(&mut self, bounds: Rect) {
        self.bounds = bounds;
    }

    /// Check if dragging
    pub fn is_dragging(&self) -> bool {
        self.drag_handler.is_dragging()
    }

    /// Get cursor hint for drag handle
    pub fn get_cursor_hint(&self) -> Option<&'static str> {
        self.drag_handler.get_cursor_hint()
    }

    /// Get number of points
    pub fn num_points(&self) -> usize {
        self.num_points
    }

}
