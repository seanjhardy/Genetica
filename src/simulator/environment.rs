// Environment module - manages bounds, lifeforms, cells, and physics simulation

use crate::utils::gpu::GpuDevice;
use crate::utils::math::{Rect, Vec2};
use crate::utils::ui::DragHandler;
use crate::simulator::planet::Planet;

/// Environment manages the simulation bounds and genetic data (genomes/GRNs)
/// NOTE: Cells and lifeforms are stored on GPU in GpuBuffers, not here
/// GRNs are stored here on CPU but can be uploaded to GPU for parallel simulation
/// The GRN is essentially a recurrent neural network - we'll use GPU compute kernels
/// to simulate the nodes in parallel
pub struct Environment {
    bounds: Rect,
    drag_handler: DragHandler,
    /// Planet background renderer
    planet: Planet,
}

impl Environment {
    pub fn new(_initial_bounds: Rect, _gpu: &GpuDevice) -> Self {
        let mut planet = Planet::new_delune();
        planet.set_bounds(_initial_bounds);
        
        Self {
            bounds: _initial_bounds,
            drag_handler: DragHandler::new(),
            planet,
        }
    }

    /// Update environment state (bounds dragging, etc.)
    pub fn update(&mut self, mouse_world_pos: Vec2, zoom: f32, ui_hovered: bool) {
        if !ui_hovered {
            let sensitivity = 50.0 / zoom;
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
                
                // Update planet bounds when simulation bounds change
                self.planet.set_bounds(self.bounds);
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
    
    /// Set bounds (for reset functionality)
    pub fn set_bounds(&mut self, bounds: Rect) {
        self.bounds = bounds;
        self.planet.set_bounds(bounds);
    }

    /// Reset environment bounds and reseed planet noise
    pub fn reset_with_new_seed(&mut self, bounds: Rect) {
        self.set_bounds(bounds);
        self.planet.reseed();
    }
    /// Get cursor hint for drag handle
    pub fn get_cursor_hint(&self) -> Option<&'static str> {
        self.drag_handler.get_cursor_hint()
    }
    
    /// Get mutable reference to planet
    pub fn planet_mut(&mut self) -> &mut Planet {
        &mut self.planet
    }
    
    /// Get reference to planet
    pub fn planet(&self) -> &Planet {
        &self.planet
    }

}
