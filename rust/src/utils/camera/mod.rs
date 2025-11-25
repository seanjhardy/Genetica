// Camera controller module

use crate::utils::math::{Rect, Vec2};

pub struct Camera {
    position: Vec2,
    zoom_level: f32,
    move_speed: f32,
    locked: bool,
    scene_bounds: Option<Rect>,
    view_size: Vec2,
}

impl Camera {
    pub fn new(view_size: Vec2, scene_bounds: Option<Rect>) -> Self {
        let position = scene_bounds
            .map(|b| b.center())
            .unwrap_or(Vec2::new(view_size.x / 2.0, view_size.y / 2.0));
        
        let zoom_level = if let Some(bounds) = scene_bounds {
            let zoom_x = view_size.x / bounds.width;
            let zoom_y = view_size.y / bounds.height;
            zoom_x.min(zoom_y)
        } else {
            1.0
        };

        Self {
            position,
            zoom_level,
            move_speed: 1000.0,
            locked: false,
            scene_bounds,
            view_size,
        }
    }

    pub fn update(&mut self, delta_time: f32, key_states: &KeyStates) {
        if self.locked {
            return;
        }

        let mut movement = Vec2::zero();
        let mut did_update = false;

        if key_states.w {
            movement.y -= self.move_speed;
            did_update = true;
        }
        if key_states.s {
            movement.y += self.move_speed;
            did_update = true;
        }
        if key_states.a {
            movement.x -= self.move_speed;
            did_update = true;
        }
        if key_states.d {
            movement.x += self.move_speed;
            did_update = true;
        }

        if did_update {
            self.position = self.position + movement * (delta_time / self.zoom_level);
            self.constrain_to_bounds();
        }
    }

    pub fn zoom(&mut self, delta: f32, mouse_pos: Vec2) {
        if self.locked {
            return;
        }

        // Minimum zoom level
        let min_zoom = if let Some(bounds) = self.scene_bounds {
            let zoom_x = self.view_size.x / (2.0 * bounds.width);
            let zoom_y = self.view_size.y / (2.0 * bounds.height);
            zoom_x.min(zoom_y)
        } else {
            0.1
        };

        // Convert mouse position to world coordinates before zoom
        let mouse_world_before = self.screen_to_world(mouse_pos);

        // Calculate new zoom level
        let new_zoom = self.zoom_level * 1.05f32.powf(delta);
        self.zoom_level = new_zoom.max(min_zoom);

        // Convert mouse position to world coordinates after zoom
        let mouse_world_after = self.screen_to_world(mouse_pos);

        // Adjust position to keep mouse position stable
        self.position = self.position + (mouse_world_before - mouse_world_after);
        self.constrain_to_bounds();
    }

    /// Convert screen pixel coordinates to world coordinates
    /// winit uses Y=0 at top (unlike SFML which uses Y=0 at bottom)
    /// Transformation: world = (screen / view_size - 0.5) * visible_size + camera_pos
    pub fn screen_to_world(&self, screen_pos: Vec2) -> Vec2 {
        let view_size = self.visible_size();
        Vec2::new(
            (screen_pos.x / self.view_size.x - 0.5) * view_size.x + self.position.x,
            (screen_pos.y / self.view_size.y - 0.5) * view_size.y + self.position.y,
        )
    }

    pub fn visible_size(&self) -> Vec2 {
        self.view_size * (1.0 / self.zoom_level)
    }

    pub fn get_position(&self) -> Vec2 {
        self.position
    }

    pub fn get_zoom(&self) -> f32 {
        self.zoom_level
    }

    pub fn set_view_size(&mut self, size: Vec2) {
        self.view_size = size;
        self.constrain_to_bounds();
    }

    pub fn set_scene_bounds(&mut self, bounds: Option<Rect>) {
        self.scene_bounds = bounds;
        self.constrain_to_bounds();
    }
    
    pub fn set_position(&mut self, position: Vec2) {
        self.position = position;
        self.constrain_to_bounds();
    }
    
    pub fn set_zoom(&mut self, zoom: f32) {
        if let Some(bounds) = self.scene_bounds {
            let zoom_x = self.view_size.x / bounds.width;
            let zoom_y = self.view_size.y / bounds.height;
            let min_zoom = zoom_x.min(zoom_y);
            self.zoom_level = zoom.max(min_zoom);
        } else {
            self.zoom_level = zoom.max(0.1);
        }
    }


    fn constrain_to_bounds(&mut self) {
        if let Some(bounds) = self.scene_bounds {
            // Constrain camera position to keep the visible area within reasonable bounds
            // Match C++ implementation: allow camera to move up to visible_size/8 outside bounds
            let visible_size = self.visible_size();
            let max_distance = visible_size * (1.0 / 8.0);

            let _old_pos = self.position;
            
            // Clamp X: allow camera to go slightly outside bounds, but not too far
            let min_x = bounds.left - max_distance.x;
            let max_x = bounds.right() + max_distance.x;
            self.position.x = self.position.x.max(min_x).min(max_x);
            
            // Clamp Y: allow camera to go slightly outside bounds, but not too far  
            let min_y = bounds.top - max_distance.y;
            let max_y = bounds.bottom() + max_distance.y;
            self.position.y = self.position.y.max(min_y).min(max_y);
        }
    }
}

#[derive(Default, Clone, Copy)]
pub struct KeyStates {
    pub w: bool,
    pub a: bool,
    pub s: bool,
    pub d: bool,
}

