// Component system for UI elements - similar to UIElement in C++

use super::styles::{Style, Size};
use super::components::{View, Text, Viewport, Image};

// Minimal Layout struct - just stores position and size
// Flexbox layout is handled in View, not here
#[derive(Debug, Clone)]
pub struct Layout {
    pub position_x: f32,  // Relative position within parent
    pub position_y: f32,  // Relative position within parent
    pub computed_width: f32,
    pub computed_height: f32,
}

impl Layout {
    pub fn new() -> Self {
        Self {
            position_x: 0.0,
            position_y: 0.0,
            computed_width: 0.0,
            computed_height: 0.0,
        }
    }
    
    pub fn default() -> Self {
        Self::new()
    }
}


#[derive(Debug, Clone)]
pub enum ComponentType {
    View(View),
    Text(Text),
    Viewport(Viewport),
    Image(Image),
}

#[derive(Debug, Clone)]
pub struct Component {
    pub component_type: ComponentType,
    pub style: Style,
    pub base_style: Style,
    pub hover_style: Option<Style>,
    pub group_hover_style: Option<Style>, // Styles to apply when parent group is hovered
    pub is_group: bool, // True if this element has the "group" class
    pub layout: Layout,
    pub visible: bool,
    pub allow_click: bool,
    pub absolute: bool,
    pub id: Option<String>,
    pub key: Option<String>,
    pub hovered: bool,
}

impl Component {
    pub fn new(component_type: ComponentType) -> Self {
        Self {
            component_type,
            style: Style::default(),
            base_style: Style::default(),
            hover_style: None,
            group_hover_style: None,
            is_group: false,
            layout: Layout::default(),
            visible: true,
            allow_click: true,
            absolute: false,
            id: None,
            key: None,
            hovered: false,
        }
    }
    
    pub fn contains(&self, point: (f32, f32)) -> bool {
        point.0 >= self.layout.position_x && 
        point.0 <= self.layout.position_x + self.layout.computed_width &&
        point.1 >= self.layout.position_y && 
        point.1 <= self.layout.position_y + self.layout.computed_height
    }


    /// Find a component by ID in the component tree
    pub fn find_by_id(&mut self, id: &str) -> Option<&mut Component> {
        if let Some(component_id) = &self.id {
            if component_id == id {
                return Some(self);
            }
        }

        match &mut self.component_type {
            ComponentType::View(view) => {
                for child in &mut view.children {
                    if let Some(found) = child.find_by_id(id) {
                        return Some(found);
                    }
                }
            }
            _ => {}
        }

        None
    }

    /// Update text content if this is a Text component
    pub fn update_text(&mut self, new_text: &str) {
        if let ComponentType::Text(text) = &mut self.component_type {
            text.content = new_text.to_string();
        }
    }
    
    /// Update hover state and apply hover styles if present
    pub fn update_hover(&mut self, mouse_x: f32, mouse_y: f32) -> bool {
        self.update_hover_recursive(mouse_x, mouse_y, 0.0, 0.0, false, false)
    }
    
    fn update_hover_recursive(&mut self, mouse_x: f32, mouse_y: f32, parent_x: f32, parent_y: f32, parent_group_hovered: bool, parent_group_was_hovered: bool) -> bool {
        let was_hovered = self.hovered;
        
        // Use deterministic position calculation (matches renderer exactly)
        let (abs_x, abs_y) = self.calculate_absolute_position(parent_x, parent_y);
        
        // Check if mouse is within bounds
        self.hovered = self.visible && 
                      mouse_x >= abs_x && 
                      mouse_x <= abs_x + self.layout.computed_width &&
                      mouse_y >= abs_y && 
                      mouse_y <= abs_y + self.layout.computed_height;
        
        // Apply group-hover styles if parent group is hovered (explicit group/hover feature)
        // Only apply group-hover if the element itself is NOT directly hovered
        // Direct hover takes precedence over group-hover
        let should_apply_group_hover = parent_group_hovered && self.group_hover_style.is_some();
        let was_applying_group_hover = parent_group_was_hovered && self.group_hover_style.is_some();
        

        // Apply hover style transition if THIS element's hover state changed
        if self.hovered != was_hovered {
            if self.hovered {
                // Mouse entered - apply hover style defined in HTML/CSS
                if let Some(hover_style) = &self.hover_style {
                    self.style = hover_style.clone();
                }
                
                // Apply hover image state if defined in HTML/CSS
                if let ComponentType::Image(ref mut image) = self.component_type {
                    image.apply_hover_state();
                }
            } else {
                // Mouse left - restore base style
                self.style = self.base_style.clone();
                
                // Restore base image state
                if let ComponentType::Image(ref mut image) = self.component_type {
                    image.restore_base_state();
                }
            }
        }
        
        if should_apply_group_hover != was_applying_group_hover || self.hovered != was_hovered {
            if should_apply_group_hover {
                // Parent group is hovered and element is not directly hovered - apply group-hover style
                if let Some(group_hover_style) = &self.group_hover_style {
                    self.style = group_hover_style.clone();
                }
                
                // Apply group-hover image state
                if let ComponentType::Image(ref mut image) = self.component_type {
                    image.apply_group_hover_state();
                }
            } else {
                // Parent group no longer hovered or element is now directly hovered - restore base style
                if !self.hovered {
                    // Only restore if we're not directly hovered
                    self.style = self.base_style.clone();
                }
                
                // Restore base image state (from group-hover)
                if let ComponentType::Image(ref mut image) = self.component_type {
                    image.restore_from_group_hover_state();
                }
            }
        }
        
        // Recursively update children hover state
        // If this element has the "group" class and is hovered, pass that to children
        let is_group_hovered = self.is_group && self.hovered;
        let was_group_hovered = self.is_group && was_hovered;
        
        if let ComponentType::View(view) = &mut self.component_type {
            for child in &mut view.children {
                child.update_hover_recursive(mouse_x, mouse_y, abs_x, abs_y, is_group_hovered, was_group_hovered);
            }
        }
        
        self.hovered
    }
    
    /// Calculate absolute screen position by walking up from root
    /// This is deterministic and matches exactly what the renderer calculates
    pub fn calculate_absolute_position(&self, parent_x: f32, parent_y: f32) -> (f32, f32) {
        let x = parent_x + self.layout.position_x + self.style.margin.left;
        let y = parent_y + self.layout.position_y + self.style.margin.top;
        (x, y)
    }
    
    /// Check if a point is within this component's bounds
    /// Uses deterministic calculation that matches renderer
    pub fn contains_point(&self, mouse_x: f32, mouse_y: f32, parent_x: f32, parent_y: f32) -> bool {
        if !self.visible {
            return false;
        }
        
        let (x, y) = self.calculate_absolute_position(parent_x, parent_y);
        let width = self.layout.computed_width.max(0.0);
        let height = self.layout.computed_height.max(0.0);
        
        mouse_x >= x && mouse_x <= x + width && mouse_y >= y && mouse_y <= y + height
    }
    
    pub fn calculate_width(&self) -> Size {
        match &self.component_type {
            ComponentType::View(view) => view.calculate_width(self.style.padding),
            ComponentType::Text(text) => Size::Pixels(text.font_size * text.content.len() as f32 * 0.6), // Approximate
            ComponentType::Viewport(_) => Size::Auto, // Viewports use explicit sizing from style
            ComponentType::Image(image) => image.calculate_width(self.style.padding),
        }
    }
    
    pub fn calculate_height(&self) -> Size {
        match &self.component_type {
            ComponentType::View(view) => view.calculate_height(self.style.padding),
            ComponentType::Text(text) => Size::Pixels(text.font_size),
            ComponentType::Viewport(_) => Size::Auto, // Viewports use explicit sizing from style
            ComponentType::Image(image) => image.calculate_height(self.style.padding),
        }
    }

    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    pub fn is_visible(&self) -> bool {
        self.visible
    }
}


