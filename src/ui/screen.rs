// Screen - similar to C++ Screen class
// Manages a collection of UI elements

use super::components::{Component, ComponentType};
use crate::utils::math::Rect;
use std::collections::HashMap;

pub struct Screen {
    elements: Vec<Component>,
    keys: HashMap<String, usize>, // Map key to element index
}

impl Screen {
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            keys: HashMap::new(),
        }
    }
    
    pub fn add_element(&mut self, mut element: Component) {
        element.layout.computed_width = 0.0; // Will be set by resize
        element.layout.computed_height = 0.0;
        element.layout.position_x = 0.0;
        element.layout.position_y = 0.0;
        
        let index = self.elements.len();
        
        // Add key if present
        if let Some(ref key) = element.key {
            self.keys.insert(key.clone(), index);
        }
        
        // Add keys from children (recursively)
        self.add_element_keys(&mut element, index);
        
        self.elements.push(element);
    }
    
    fn add_element_keys(&mut self, element: &mut Component, parent_index: usize) {
        if let Some(ref key) = element.key {
            // For child elements, we'll need a different approach
            // For now, just store the parent index
            self.keys.insert(key.clone(), parent_index);
        }
        
        match &mut element.component_type {
            ComponentType::View(view) => {
                for child in &mut view.children {
                    self.add_element_keys(child, parent_index);
                }
            }
            _ => {}
        }
    }
    
    pub fn find_element_by_id(&mut self, id: &str) -> Option<&mut Component> {
        for element in &mut self.elements {
            if let Some(found) = element.find_by_id(id) {
                return Some(found);
            }
        }
        None
    }
    
    pub fn resize(&mut self, width: f32, height: f32) {
        for element in &mut self.elements {
            element.layout.computed_width = width;
            element.layout.computed_height = height;
            element.layout.mark_dirty(); // Mark as dirty to trigger layout recalculation

            // Layout the element
            Self::layout_element_at_size(element, width, height);
        }
    }
    
    fn layout_element_at_size(element: &mut Component, width: f32, height: f32) {
        // Calculate element size based on style
        // Set computed size
        match element.style.width {
            super::styles::Size::Pixels(value) => element.layout.computed_width = value,
            super::styles::Size::Percent(value) => element.layout.computed_width = width * value / 100.0,
            super::styles::Size::Flex(_) | super::styles::Size::Auto => element.layout.computed_width = width,
        }
        
        match element.style.height {
            super::styles::Size::Pixels(value) => element.layout.computed_height = value,
            super::styles::Size::Percent(value) => element.layout.computed_height = height * value / 100.0,
            super::styles::Size::Flex(_) | super::styles::Size::Auto => element.layout.computed_height = height,
        }
        
        if let ComponentType::View(ref mut view) = element.component_type {
            view.update_layout(0.0, 0.0, element.layout.computed_width, element.layout.computed_height, element.style.padding);
        }
    }

    pub fn update(&mut self, _dt: f32, mouse_pos: (f32, f32)) -> (bool, Option<&'static str>) {
        let mut hovered = false;
        let mut cursor_hint = None;

        for element in &mut self.elements {
            // Update hover states for all components (including children)
            let element_hovered = element.update_hover(mouse_pos.0, mouse_pos.1);
            if element_hovered {
                hovered = true;

                // Check if this element or any of its hovered children has cursor-pointer style
                let element_cursor = element.get_cursor_hint(mouse_pos.0, mouse_pos.1);
                if let Some(hint) = element_cursor {
                    cursor_hint = Some(hint);
                }
            }
        }

        (hovered, cursor_hint)
    }
    
    /// Handle mouse click event - returns the handler name if a handler was triggered
    /// This follows the C++ pattern where clicks propagate down the tree
    pub fn handle_click(&mut self, mouse_pos: (f32, f32)) -> Option<String> {
        for element in &mut self.elements {
            if let Some(handler) = Self::handle_click_recursive(element, mouse_pos, 0.0, 0.0) {
                return Some(handler);
            }
        }
        None
    }
    
    /// Recursively handle clicks down the component tree (like C++ View::handleEvent)
    /// Returns handler name if consumed, propagates up if not
    /// Uses deterministic position calculation that matches renderer exactly
    fn handle_click_recursive(
        component: &mut Component,
        mouse_pos: (f32, f32),
        parent_x: f32,
        parent_y: f32,
    ) -> Option<String> {
        if !component.visible {
            return None;
        }
        
        // Calculate absolute position deterministically (same as renderer)
        let (x, y) = component.calculate_absolute_position(parent_x, parent_y);
        
        // First, let children handle the event (like C++ - children get priority)
        if let ComponentType::View(view) = &mut component.component_type {
            // Iterate children in reverse order (top layer first)
            for child_idx in (0..view.children.len()).rev() {
                let child = &mut view.children[child_idx];
                if let Some(handler) = Self::handle_click_recursive(child, mouse_pos, x, y) {
                    return Some(handler); // Event was consumed by child
                }
            }
        }
        
        // If no child consumed the event, check if this component handles it
        // Only Views with explicit on_click handlers can consume clicks
        // Images, Text, Viewports, and Views without handlers are transparent to clicks
        if let ComponentType::View(view) = &component.component_type {
            if let Some(handler) = &view.on_click {
                // This view has a click handler - check if click is within bounds
                let contains_point = component.contains_point(mouse_pos.0, mouse_pos.1, parent_x, parent_y);
        
        if contains_point && component.visible && component.allow_click {
                    // Click is within bounds and component is clickable - consume the event
                    return Some(handler.clone());
                }
            }
        }
        
        None
    }
    
    /// Check if a point is over any UI element (for blocking world interaction)
    #[allow(dead_code)] // Public API - may be used by external code
    pub fn is_point_over_ui(&self, mouse_pos: (f32, f32)) -> bool {
        self.elements
            .iter()
            .any(|element| Self::is_point_over_component(element, mouse_pos, 0.0, 0.0))
    }
    
    /// Recursively check if a point is over a component or any of its children
    /// Uses deterministic position calculation matching renderer and click detection
    #[allow(dead_code)] // Used by is_point_over_ui
    fn is_point_over_component(
        component: &Component,
        mouse_pos: (f32, f32),
        parent_x: f32,
        parent_y: f32,
    ) -> bool {
        if !component.visible {
            return false;
        }

        // Check if point is within this component's bounds AND it has a non-transparent background
        if component.contains_point(mouse_pos.0, mouse_pos.1, parent_x, parent_y) &&
           component.style.background_color.a != 0.0 {
            return true;
        }

        // Check children recursively
        if let ComponentType::View(view) = &component.component_type {
            let (x, y) = component.calculate_absolute_position(parent_x, parent_y);
            for child in &view.children {
                if Self::is_point_over_component(child, mouse_pos, x, y) {
                    return true;
                }
            }
        }

        false
    }
    
    pub fn get_elements_mut(&mut self) -> &mut [Component] {
        &mut self.elements
    }

    pub fn find_component_bounds(&self, id: &str) -> Option<Rect> {
        for element in &self.elements {
            if let Some(bounds) = Self::find_component_bounds_recursive(element, id, 0.0, 0.0) {
                return Some(bounds);
            }
        }
        None
    }

    fn find_component_bounds_recursive(
        component: &Component,
        id: &str,
        parent_x: f32,
        parent_y: f32,
    ) -> Option<Rect> {
        if let Some(component_id) = &component.id {
            if component_id == id {
                let (x, y) = component.calculate_absolute_position(parent_x, parent_y);
                return Some(Rect::new(
                    x,
                    y,
                    component.layout.computed_width,
                    component.layout.computed_height,
                ));
            }
        }

        if let ComponentType::View(view) = &component.component_type {
            let (x, y) = component.calculate_absolute_position(parent_x, parent_y);
            for child in &view.children {
                if let Some(bounds) = Self::find_component_bounds_recursive(child, id, x, y) {
                    return Some(bounds);
                }
            }
        }

        None
    }
}

impl Default for Screen {
    fn default() -> Self {
        Self::new()
    }
}
