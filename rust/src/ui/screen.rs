// Screen - similar to C++ Screen class
// Manages a collection of UI elements

use super::component::Component;
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
            super::component::ComponentType::View(view) => {
                for child in &mut view.children {
                    self.add_element_keys(child, parent_index);
                }
            }
            _ => {}
        }
    }
    
    pub fn get_element(&mut self, key: &str) -> Option<&mut Component> {
        self.keys.get(key)
            .and_then(|&index| self.elements.get_mut(index))
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
        
        // Layout children if this is a View
        if let super::component::ComponentType::View(ref mut view) = element.component_type {
            view.update_layout(0.0, 0.0, element.layout.computed_width, element.layout.computed_height, element.style.padding);
        }
    }
    
    pub fn handle_event(&mut self, _event: &winit::event::WindowEvent) -> bool {
        // Handle events - this is a placeholder
        // In a real implementation, you'd call handle_event on each element
        false
    }
    
    pub fn update(&mut self, _dt: f32, mouse_pos: (f32, f32)) -> bool {
        let mut hovered = false;
        for element in &mut self.elements {
            // Update elements - placeholder
            // In a real implementation, you'd check if mouse is over element
            if element.contains(mouse_pos) && element.visible && element.allow_click {
                hovered = true;
            }
        }
        hovered
    }
    
    pub fn get_elements(&self) -> &[Component] {
        &self.elements
    }
    
    pub fn get_elements_mut(&mut self) -> &mut [Component] {
        &mut self.elements
    }
    
    pub fn reset(&mut self) {
        self.elements.clear();
        self.keys.clear();
    }
}

impl Default for Screen {
    fn default() -> Self {
        Self::new()
    }
}

