// Component system for UI elements - similar to UIElement in C++

use super::styles::{Style, Size};
use super::components::{View, Text, Viewport};

// Minimal Layout struct - just stores position and size
// Flexbox layout is handled in View, not here
#[derive(Debug, Clone)]
pub struct Layout {
    pub position_x: f32,
    pub position_y: f32,
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
}

#[derive(Debug, Clone)]
pub struct Component {
    pub component_type: ComponentType,
    pub style: Style,
    pub layout: Layout,
    pub visible: bool,
    pub allow_click: bool,
    pub absolute: bool,
    pub id: Option<String>,
    pub key: Option<String>,
}

impl Component {
    pub fn new(component_type: ComponentType) -> Self {
        Self {
            component_type,
            style: Style::default(),
            layout: Layout::default(),
            visible: true,
            allow_click: true,
            absolute: false,
            id: None,
            key: None,
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
    
    pub fn calculate_width(&self) -> Size {
        match &self.component_type {
            ComponentType::View(view) => view.calculate_width(self.style.padding),
            ComponentType::Text(text) => Size::Pixels(text.font_size * text.content.len() as f32 * 0.6), // Approximate
            ComponentType::Viewport(_) => Size::Auto, // Viewports use explicit sizing from style
        }
    }
    
    pub fn calculate_height(&self) -> Size {
        match &self.component_type {
            ComponentType::View(view) => view.calculate_height(self.style.padding),
            ComponentType::Text(text) => Size::Pixels(text.font_size),
            ComponentType::Viewport(_) => Size::Auto, // Viewports use explicit sizing from style
        }
    }
}


