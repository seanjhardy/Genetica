// Drag handler for resizing bounding boxes

use crate::utils::math::{Rect, Vec2};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DragHandle {
    None,
    Left,
    Right,
    Top,
    Bottom,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

pub struct DragHandler {
    drag_handle: DragHandle,
    last_mouse_pos: Vec2,
    dragging: bool,
}

impl DragHandler {
    pub fn new() -> Self {
        Self {
            drag_handle: DragHandle::None,
            last_mouse_pos: Vec2::zero(),
            dragging: false,
        }
    }

    pub fn handle_mouse_press(&mut self, pressed: bool) {
        if pressed {
            if self.drag_handle != DragHandle::None {
                self.dragging = true;
            }
        } else {
            self.dragging = false;
        }
    }

    pub fn update(&mut self, mouse_pos: Vec2, bounds: Rect, sensitivity: f32) -> Rect {
        let delta = mouse_pos - self.last_mouse_pos;
        self.last_mouse_pos = mouse_pos;

        // Calculate distances to edges (exactly matching C++ implementation)
        let dist_left = (mouse_pos.x - bounds.left).abs();
        let dist_right = (mouse_pos.x - bounds.right()).abs();
        let dist_top = (mouse_pos.y - bounds.top).abs();
        let dist_bottom = (mouse_pos.y - bounds.bottom()).abs();

        // Determine which handle is being hovered (only when not dragging)
        // Match C++ logic exactly: check corners first, then edges, with bounds checks
        let new_handle = if !self.dragging {
            let corner_range = sensitivity * 2.0;
            let edge_range = sensitivity;
            
            // Check corners first (corner_range = sensitivity * 2)
            if dist_left < corner_range && dist_top < corner_range {
                DragHandle::TopLeft
            } else if dist_right < corner_range && dist_top < corner_range {
                DragHandle::TopRight
            } else if dist_left < corner_range && dist_bottom < corner_range {
                DragHandle::BottomLeft
            } else if dist_right < corner_range && dist_bottom < corner_range {
                DragHandle::BottomRight
            }
            // Check edges with bounds constraints (must be within bounds for perpendicular dimension)
            else if dist_left < edge_range && 
                    mouse_pos.y >= bounds.top && mouse_pos.y <= bounds.bottom() {
                DragHandle::Left
            } else if dist_right < edge_range && 
                      mouse_pos.y >= bounds.top && mouse_pos.y <= bounds.bottom() {
                DragHandle::Right
            } else if dist_top < edge_range && 
                      mouse_pos.x >= bounds.left && mouse_pos.x <= bounds.right() {
                DragHandle::Top
            } else if dist_bottom < edge_range && 
                      mouse_pos.x >= bounds.left && mouse_pos.x <= bounds.right() {
                DragHandle::Bottom
            } else {
                DragHandle::None
            }
        } else {
            self.drag_handle
        };

        if new_handle != self.drag_handle && !self.dragging {
            self.drag_handle = new_handle;
        }

        // Calculate delta for dragging (exactly matching C++ implementation)
        if self.dragging {
            let horizontal = self.horizontal_direction();
            let vertical = self.vertical_direction();

            // C++ returns: { horizontal == -1 ? delta.x : 0, vertical == -1 ? delta.y : 0,
            //              horizontal * delta.x, vertical * delta.y }
            return Rect::new(
                if horizontal == -1 { delta.x } else { 0.0 },
                if vertical == -1 { delta.y } else { 0.0 },
                (horizontal as f32) * delta.x,
                (vertical as f32) * delta.y,
            );
        }

        Rect::new(0.0, 0.0, 0.0, 0.0)
    }

    pub fn horizontal_direction(&self) -> i32 {
        match self.drag_handle {
            DragHandle::Left | DragHandle::TopLeft | DragHandle::BottomLeft => -1,
            DragHandle::Right | DragHandle::TopRight | DragHandle::BottomRight => 1,
            _ => 0,
        }
    }

    pub fn vertical_direction(&self) -> i32 {
        match self.drag_handle {
            DragHandle::Top | DragHandle::TopLeft | DragHandle::TopRight => -1,
            DragHandle::Bottom | DragHandle::BottomLeft | DragHandle::BottomRight => 1,
            _ => 0,
        }
    }


    pub fn reset(&mut self) {
        self.dragging = false;
        self.drag_handle = DragHandle::None;
    }

    pub fn get_cursor_hint(&self) -> Option<&'static str> {
        match self.drag_handle {
            DragHandle::Left | DragHandle::Right => Some("ew-resize"),
            DragHandle::Top | DragHandle::Bottom => Some("ns-resize"),
            DragHandle::TopLeft | DragHandle::BottomRight => Some("nwse-resize"),
            DragHandle::TopRight | DragHandle::BottomLeft => Some("nesw-resize"),
            DragHandle::None => None,
        }
    }
}

