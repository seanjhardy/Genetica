// Component system for UI elements - similar to UIElement in C++

use super::styles::{Style, Size};
use super::layout::Layout;

#[derive(Debug, Clone)]
pub enum ComponentType {
    View(View),
    Text(Text),
    Button(Button),
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

    pub fn get_computed_width(&self) -> f32 {
        self.layout.computed_width
    }

    pub fn get_computed_height(&self) -> f32 {
        self.layout.computed_height
    }

    pub fn get_absolute_x(&self) -> f32 {
        self.layout.position_x
    }

    pub fn get_absolute_y(&self) -> f32 {
        self.layout.position_y
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
        } else if let ComponentType::Button(button) = &mut self.component_type {
            button.label = new_text.to_string();
        }
    }
    
    pub fn calculate_width(&self) -> Size {
        match &self.component_type {
            ComponentType::View(view) => view.calculate_width(),
            ComponentType::Text(text) => Size::Pixels(text.font_size * text.content.len() as f32 * 0.6), // Approximate
            ComponentType::Button(button) => Size::Pixels(button.font_size * button.label.len() as f32 * 0.6),
        }
    }
    
    pub fn calculate_height(&self) -> Size {
        match &self.component_type {
            ComponentType::View(view) => view.calculate_height(),
            ComponentType::Text(text) => Size::Pixels(text.font_size),
            ComponentType::Button(button) => Size::Pixels(button.font_size),
        }
    }
}

#[derive(Debug, Clone)]
pub struct View {
    pub children: Vec<Component>,
    // View-specific layout properties (moved from Layout for clarity)
    pub flex_direction: super::layout::FlexDirection,
    pub row_alignment: super::layout::Alignment,
    pub column_alignment: super::layout::Alignment,
    pub background_color: super::styles::Color,
    pub shadow: super::styles::Shadow,
    pub gap: f32,
    // Layers for absolute positioning
    pub layers: Vec<Vec<usize>>, // Indices into children
    // Click handler
    pub on_click: Option<String>,
}

impl View {
    pub fn new() -> Self {
        Self {
            children: Vec::new(),
            flex_direction: super::layout::FlexDirection::Row,
            row_alignment: super::layout::Alignment::Start,
            column_alignment: super::layout::Alignment::Start,
            background_color: super::styles::Color::transparent(),
            shadow: super::styles::Shadow::none(),
            gap: 0.0,
            layers: Vec::new(),
            on_click: None,
        }
    }

    pub fn with_children(mut self, children: Vec<Component>) -> Self {
        // Organize children into layers (absolute vs normal)
        let mut base_layer = Vec::new();
        for (i, child) in children.iter().enumerate() {
            if child.absolute {
                self.layers.push(vec![i]);
            } else {
                base_layer.push(i);
            }
        }
        if !base_layer.is_empty() {
            self.layers.push(base_layer);
        }
        self.children = children;
        self
    }
    
    pub fn calculate_width(&self) -> Size {
        let mut total = 0.0;
        for layer in &self.layers {
            for &idx in layer {
                if idx < self.children.len() {
                    let child = &self.children[idx];
                    if !child.visible || child.absolute {
                        continue;
                    }
                    match child.calculate_width() {
                        Size::Pixels(value) => total += value,
                        _ => {}
                    }
                    total += self.gap;
                }
            }
        }
        Size::Pixels(total.max(0.0) - self.gap)
    }
    
    pub fn calculate_height(&self) -> Size {
        let mut total = 0.0;
        for layer in &self.layers {
            for &idx in layer {
                if idx < self.children.len() {
                    let child = &self.children[idx];
                    if !child.visible || child.absolute {
                        continue;
                    }
                    match child.calculate_height() {
                        Size::Pixels(value) => total += value,
                        _ => {}
                    }
                    total += self.gap;
                }
            }
        }
        Size::Pixels(total.max(0.0) - self.gap)
    }
    
    // Layout update method - similar to C++ View::updateLayout()
    pub fn update_layout(&mut self, parent_x: f32, parent_y: f32, parent_width: f32, parent_height: f32, parent_padding: super::styles::Padding) {
        for layer in &self.layers.clone() {
            self.update_layer_layout(layer, parent_x, parent_y, parent_width, parent_height, parent_padding);
        }
    }
    
    fn update_layer_layout(&mut self, layer: &[usize], parent_x: f32, parent_y: f32, parent_width: f32, parent_height: f32, parent_padding: super::styles::Padding) {
        // Get visible children indices
        let visible_indices: Vec<usize> = layer.iter()
            .filter(|&&idx| idx < self.children.len() && self.children[idx].visible)
            .copied()
            .collect();
            
        if visible_indices.is_empty() {
            return;
        }
        
        // Calculate available space (accounting for padding)
        let padding = parent_padding;
        
        let container_width = parent_width - padding.left - padding.right;
        let container_height = parent_height - padding.top - padding.bottom;
        
        let available_main = if self.flex_direction == super::layout::FlexDirection::Row {
            container_width
        } else {
            container_height
        };
        let available_cross = if self.flex_direction == super::layout::FlexDirection::Row {
            container_height
        } else {
            container_width
        };
        
        let available_main = available_main - self.gap * (visible_indices.len() as f32 - 1.0);
        
        // First pass: calculate sizes
        let mut total_fixed_main = 0.0;
        let mut flex_item_count = 0;
        let mut total_flex_grow = 0.0;
        
        for &idx in &visible_indices {
            if idx >= self.children.len() {
                continue;
            }
            let child = &self.children[idx];
            let main_size = if self.flex_direction == super::layout::FlexDirection::Row {
                &child.style.width
            } else {
                &child.style.height
            };
            
            match main_size {
                Size::Pixels(value) => total_fixed_main += value,
                Size::Percent(value) => total_fixed_main += available_main * value / 100.0,
                Size::Flex(value) => {
                    flex_item_count += 1;
                    total_flex_grow += value.max(0.0);
                }
                Size::Auto => {
                    flex_item_count += 1;
                    total_flex_grow += 1.0;
                }
            }
        }
        
        let remaining_main = (available_main - total_fixed_main).max(0.0);
        let flex_unit = if flex_item_count > 0 && total_flex_grow > 0.0 {
            remaining_main / total_flex_grow
        } else {
            0.0
        };
        
        // Second pass: set sizes and positions
        let mut current_main_pos = if self.flex_direction == super::layout::FlexDirection::Row {
            parent_x + padding.left
        } else {
            parent_y + padding.top
        };
        
        for &idx in &visible_indices {
            if idx >= self.children.len() {
                continue;
            }
            let child = &mut self.children[idx];
            
            // Calculate main axis size
            let main_size = if self.flex_direction == super::layout::FlexDirection::Row {
                &child.style.width
            } else {
                &child.style.height
            };
            
            let item_main_size = match main_size {
                Size::Pixels(value) => *value,
                Size::Percent(value) => available_main * value / 100.0,
                Size::Flex(value) => flex_unit * value.max(0.0),
                Size::Auto => flex_unit * 1.0,
            };
            
            // Calculate cross axis size
            let cross_size = if self.flex_direction == super::layout::FlexDirection::Row {
                &child.style.height
            } else {
                &child.style.width
            };
            
            let item_cross_size = match cross_size {
                Size::Pixels(value) => *value,
                Size::Percent(value) => available_cross * value / 100.0,
                Size::Flex(_) | Size::Auto => available_cross,
            };
            
            // Set component size
            if self.flex_direction == super::layout::FlexDirection::Row {
                child.layout.computed_width = item_main_size;
                child.layout.computed_height = item_cross_size;
            } else {
                child.layout.computed_width = item_cross_size;
                child.layout.computed_height = item_main_size;
            }
            
            // Set position
            let alignment = if self.flex_direction == super::layout::FlexDirection::Row {
                self.column_alignment
            } else {
                self.row_alignment
            };
            
            let cross_pos = if self.flex_direction == super::layout::FlexDirection::Row {
                parent_y + padding.top
            } else {
                parent_x + padding.left
            };
            
            let adjusted_cross_pos = match alignment {
                super::layout::Alignment::Center => cross_pos + (available_cross - item_cross_size) / 2.0,
                super::layout::Alignment::End => {
                    cross_pos + available_cross - item_cross_size - 
                    if self.flex_direction == super::layout::FlexDirection::Row {
                        padding.bottom
                    } else {
                        padding.right
                    }
                }
                _ => cross_pos,
            };
            
            if self.flex_direction == super::layout::FlexDirection::Row {
                child.layout.position_x = current_main_pos;
                child.layout.position_y = adjusted_cross_pos;
            } else {
                child.layout.position_x = adjusted_cross_pos;
                child.layout.position_y = current_main_pos;
            }
            
            // Recursively layout child if it's a View
            match &mut child.component_type {
                ComponentType::View(ref mut view) => {
                    view.update_layout(
                        child.layout.position_x,
                        child.layout.position_y,
                        item_main_size,
                        item_cross_size,
                        child.style.padding
                    );
                }
                _ => {}
            }
            
            current_main_pos += item_main_size + self.gap;
        }
        
        // Apply main axis alignment
        let total_space_taken = current_main_pos - self.gap - 
            if self.flex_direction == super::layout::FlexDirection::Row {
                parent_x + padding.left
            } else {
                parent_y + padding.top
            };
        let extra_space = (available_main - total_space_taken).max(0.0);
        
        if extra_space > 0.0 {
            let main_alignment = if self.flex_direction == super::layout::FlexDirection::Row {
                self.row_alignment
            } else {
                self.column_alignment
            };
            
            match main_alignment {
                super::layout::Alignment::Center => {
                    let offset = extra_space / 2.0;
                    for &idx in &visible_indices {
                        if idx < self.children.len() {
                            if self.flex_direction == super::layout::FlexDirection::Row {
                                self.children[idx].layout.position_x += offset;
                            } else {
                                self.children[idx].layout.position_y += offset;
                            }
                        }
                    }
                }
                super::layout::Alignment::End => {
                    for &idx in &visible_indices {
                        if idx < self.children.len() {
                            if self.flex_direction == super::layout::FlexDirection::Row {
                                self.children[idx].layout.position_x += extra_space;
                            } else {
                                self.children[idx].layout.position_y += extra_space;
                            }
                        }
                    }
                }
                super::layout::Alignment::SpaceBetween => {
                    if visible_indices.len() > 1 {
                        let gap = extra_space / (visible_indices.len() - 1) as f32;
                        let mut offset = 0.0;
                        for &idx in &visible_indices[1..] {
                            if idx < self.children.len() {
                                offset += gap;
                                if self.flex_direction == super::layout::FlexDirection::Row {
                                    self.children[idx].layout.position_x += offset;
                                } else {
                                    self.children[idx].layout.position_y += offset;
                                }
                            }
                        }
                    }
                }
                super::layout::Alignment::SpaceAround => {
                    if !visible_indices.is_empty() {
                        let gap = extra_space / visible_indices.len() as f32;
                        let mut offset = gap / 2.0;
                        for &idx in &visible_indices[1..] {
                            if idx < self.children.len() {
                                offset += gap;
                                if self.flex_direction == super::layout::FlexDirection::Row {
                                    self.children[idx].layout.position_x += offset;
                                } else {
                                    self.children[idx].layout.position_y += offset;
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

impl Default for View {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct Text {
    pub content: String,
    pub font_size: f32,
    pub color: super::styles::Color,
}

impl Text {
    pub fn new(content: String) -> Self {
        Self {
            content,
            font_size: 16.0,
            color: super::styles::Color::black(),
        }
    }

    pub fn with_font_size(mut self, size: f32) -> Self {
        self.font_size = size;
        self
    }

    pub fn with_color(mut self, color: super::styles::Color) -> Self {
        self.color = color;
        self
    }
}

#[derive(Debug, Clone)]
pub struct Button {
    pub label: String,
    pub font_size: f32,
    pub text_color: super::styles::Color,
    pub on_click: Option<String>, // Event handler name
}

impl Button {
    pub fn new(label: String) -> Self {
        Self {
            label,
            font_size: 16.0,
            text_color: super::styles::Color::black(),
            on_click: None,
        }
    }

    pub fn with_font_size(mut self, size: f32) -> Self {
        self.font_size = size;
        self
    }

    pub fn with_text_color(mut self, color: super::styles::Color) -> Self {
        self.text_color = color;
        self
    }

    pub fn with_on_click(mut self, handler: String) -> Self {
        self.on_click = Some(handler);
        self
    }
}
