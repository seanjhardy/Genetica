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
    pub layout_dirty: bool,  // Flag to indicate layout needs recalculation
    pub size_changed: bool,  // Flag to indicate if size changed since last layout
}

impl Layout {
    pub fn new() -> Self {
        Self {
            position_x: 0.0,
            position_y: 0.0,
            computed_width: 0.0,
            computed_height: 0.0,
            layout_dirty: true,  // Start with dirty flag set
            size_changed: false,
        }
    }

    pub fn default() -> Self {
        Self::new()
    }

    pub fn mark_dirty(&mut self) {
        self.layout_dirty = true;
        self.size_changed = false; // Reset size changed flag when marking dirty
    }

    pub fn clear_dirty(&mut self) {
        self.layout_dirty = false;
        self.size_changed = false; // Clear size changed flag too
    }

    pub fn is_dirty(&self) -> bool {
        self.layout_dirty
    }

    pub fn mark_size_changed(&mut self) {
        self.size_changed = true;
        self.layout_dirty = true; // Size changes require layout recalculation
    }

    pub fn has_size_changed(&self) -> bool {
        self.size_changed
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
            // Only update if the text actually changed
            if text.content != new_text {
                // Store old dimensions before updating
                let old_width = text.cached_width();
                let old_height = text.cached_height();

                // Update text content and bounds
                text.content = new_text.to_string();
                text.update_bounds();

                // Check if size actually changed
                let new_width = text.cached_width();
                let new_height = text.cached_height();

                if (new_width - old_width).abs() > 0.1 || (new_height - old_height).abs() > 0.1 {
                    // Size changed - mark for size-based selective layout
                    self.layout.mark_size_changed();
                } else {
                    // Content changed but size didn't - still need layout update for rendering
                    self.layout.mark_dirty();
                }
            }
        }
    }

    pub fn set_font_size(&mut self, font_size: f32) {
        if let ComponentType::Text(text) = &mut self.component_type {
            if text.set_font_size(font_size) {
                // Font size changed, mark layout as dirty
                self.layout.mark_dirty();
            }
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

            // Sync text color from current style
            self.sync_text_color_from_style();
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

            // Sync text color from current style
            self.sync_text_color_from_style();
        }
        
        // Recursively update children hover state
        // If this element has the "group" class and is hovered, pass that to children
        let is_group_hovered = self.is_group && self.hovered;
        let was_group_hovered = self.is_group && was_hovered;
        
        let mut is_ui_hovered = self.hovered && (
            self.style.background_color.a != 0.0 ||
            (matches!(self.component_type, ComponentType::View(ref view) if view.on_click.is_some()))
        );

        if let ComponentType::View(view) = &mut self.component_type {
            for child in &mut view.children {
                let child_hovered = child.update_hover_recursive(mouse_x, mouse_y, abs_x, abs_y, is_group_hovered, was_group_hovered);
                if child_hovered {
                    is_ui_hovered = true;
                }
            }
        }
        
        is_ui_hovered
    }

    /// Sync text color from the current style to the text component
    fn sync_text_color_from_style(&mut self) {
        if let ComponentType::Text(ref mut text) = self.component_type {
            if let Some(text_color) = self.style.text_color {
                text.color = text_color;
            }
        }
    }

    /// Calculate absolute screen position by walking up from root
    /// This is deterministic and matches exactly what the renderer calculates
    pub fn calculate_absolute_position(&self, parent_x: f32, parent_y: f32) -> (f32, f32) {
        let x = parent_x + self.layout.position_x;
        let y = parent_y + self.layout.position_y;
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
        // Check if explicit width is set in style
        match &self.style.width {
            Size::Pixels(value) => return Size::Pixels(*value),
            Size::Percent(value) => return Size::Percent(*value),
            // Don't return Size::Flex here - fall through to intrinsic calculation
            Size::Flex(_) => {}
            Size::Auto => {} // Fall through to component-specific calculation
        }
        
        match &self.component_type {
            ComponentType::View(view) => view.calculate_width(self.style.padding),
            ComponentType::Text(text) => {
                // Return the content width plus padding (consistent with renderer)
                let content_width = text.cached_width() + self.style.padding.left + self.style.padding.right;
                Size::Pixels(content_width)
            },
            ComponentType::Viewport(_) => Size::Auto,
            ComponentType::Image(image) => Size::Pixels(image.natural_width),
        }
    }

    pub fn calculate_height(&self) -> Size {
        // Check if explicit height is set in style
        match &self.style.height {
            Size::Pixels(value) => return Size::Pixels(*value),
            Size::Percent(value) => return Size::Percent(*value),
            // Don't return Size::Flex here - fall through to intrinsic calculation
            Size::Flex(_) => {}
            Size::Auto => {} // Fall through to component-specific calculation
        }

        match &self.component_type {
            ComponentType::View(view) => view.calculate_height(self.style.padding),
            ComponentType::Text(text) => {
                // Return the content height plus padding (consistent with renderer)
                let content_height = text.cached_height() + self.style.padding.top + self.style.padding.bottom;
                Size::Pixels(content_height)
            },
            ComponentType::Viewport(_) => Size::Auto,
            ComponentType::Image(image) => Size::Pixels(image.natural_height)
        }
    }

    /// Get cursor hint for this component and its children when hovered
    pub fn get_cursor_hint(&self, mouse_x: f32, mouse_y: f32) -> Option<&'static str> {
        self.get_cursor_hint_recursive(mouse_x, mouse_y, 0.0, 0.0)
    }

    fn get_cursor_hint_recursive(&self, mouse_x: f32, mouse_y: f32, parent_x: f32, parent_y: f32) -> Option<&'static str> {
        if !self.visible {
            return None;
        }

        // Calculate absolute position
        let (abs_x, abs_y) = self.calculate_absolute_position(parent_x, parent_y);

        // Check if mouse is within bounds
        let is_hovered = mouse_x >= abs_x &&
                        mouse_x <= abs_x + self.layout.computed_width &&
                        mouse_y >= abs_y &&
                        mouse_y <= abs_y + self.layout.computed_height;

        if is_hovered {
            // Check children first (they take priority)
            if let ComponentType::View(view) = &self.component_type {
                for child in &view.children {
                    if let Some(hint) = child.get_cursor_hint_recursive(mouse_x, mouse_y, abs_x, abs_y) {
                        return Some(hint);
                    }
                }
            }

            // If no child has a cursor hint, check this component's cursor style
            match self.style.cursor {
                super::styles::Cursor::Pointer => Some("pointer"),
                super::styles::Cursor::Default => None,
            }
        } else {
            None
        }
    }

}


