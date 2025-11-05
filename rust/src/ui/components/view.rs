
#[derive(Debug, Clone)]
pub struct View {
    pub children: Vec<super::Component>,
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

    // Constructs layers to render where absolute children are rendered on top of non-absolute children
    pub fn rebuild_layers(&mut self) {
        self.layers.clear();
        let mut base_layer = Vec::new();
        for (i, child) in self.children.iter().enumerate() {
            if child.absolute {
                self.layers.push(vec![i]);
            } else {
                base_layer.push(i);
            }
        }
        // Add base layer to the front so it is rendered first
        if !base_layer.is_empty() {
            self.layers.insert(0, base_layer);
        }
    }
    
    pub fn calculate_width(&self, padding: super::styles::Padding) -> super::Size {
        let mut max_width: f32 = 0.0;
        for layer in &self.layers {
            let mut layer_width = padding.left + padding.right;
            let mut visible_children = 0;
            for &idx in layer {
                if idx < self.children.len() {
                    let child = &self.children[idx];
                    if !child.visible || child.absolute {
                        continue;
                    }
                    visible_children += 1;
                    if let super::Size::Pixels(value) = child.calculate_width() {
                        layer_width += value;
                    }
                } 
            }
            layer_width += self.gap * (visible_children - 1) as f32;
            max_width = max_width.max(layer_width);
        }
        super::Size::Pixels(max_width)
    }
    
    // Calculate minimum height needed to fit children (including padding)
    // This matches the C++ View::calculateHeight() implementation exactly
    pub fn calculate_height(&self, padding: super::styles::Padding) -> super::Size {
        let mut max_height: f32 = 0.0;
        for layer in &self.layers {
            let mut layer_height = padding.top + padding.bottom;
            let mut visible_children = 0;
            for &idx in layer {
                if idx < self.children.len() {
                    let child = &self.children[idx];
                    if !child.visible || child.absolute {
                        continue;
                    }
                    visible_children += 1;
                    if let super::Size::Pixels(value) = child.calculate_height() {
                        layer_height += value;
                    }
                }
            }
            layer_height += self.gap * (visible_children - 1) as f32;
            max_height = max_height.max(layer_height);
        }
        super::Size::Pixels(max_height)
    }
    
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
        
        // Check if this layer contains absolute children
        // If the first child is absolute, all children in this layer are absolute
        let is_absolute_layer = visible_indices.first()
            .map(|&idx| idx < self.children.len() && self.children[idx].absolute)
            .unwrap_or(false);
        
        // For absolute children, position them relative to parent, not in flex flow
        if is_absolute_layer {
            // Position absolute children relative to parent
            for &idx in &visible_indices {
                if idx >= self.children.len() {
                    continue;
                }
                let child = &mut self.children[idx];
                
                // Calculate size (same as normal flow)
                let main_size = if self.flex_direction == super::layout::FlexDirection::Row {
                    &child.style.width
                } else {
                    &child.style.height
                };
                
                let item_main_size = match main_size {
                    super::Size::Pixels(value) => *value,
                    super::Size::Percent(value) => parent_width * value / 100.0,
                    super::Size::Flex(_) | super::Size::Auto => {
                        // For absolute, use parent width/height if not specified
                        if self.flex_direction == super::layout::FlexDirection::Row {
                            parent_width
                        } else {
                            parent_height
                        }
                    }
                };
                
                let cross_size = if self.flex_direction == super::layout::FlexDirection::Row {
                    &child.style.height
                } else {
                    &child.style.width
                };
                
                let item_cross_size = match cross_size {
                    super::Size::Pixels(value) => *value,
                    super::Size::Percent(value) => {
                        if self.flex_direction == super::layout::FlexDirection::Row {
                            parent_height * value / 100.0
                        } else {
                            parent_width * value / 100.0
                        }
                    }
                    super::Size::Flex(_) | super::Size::Auto => {
                        if self.flex_direction == super::layout::FlexDirection::Row {
                            parent_height
                        } else {
                            parent_width
                        }
                    }
                };
                
                // Set component size
                if self.flex_direction == super::layout::FlexDirection::Row {
                    child.layout.computed_width = item_main_size;
                    child.layout.computed_height = item_cross_size;
                } else {
                    child.layout.computed_width = item_cross_size;
                    child.layout.computed_height = item_main_size;
                }
                
                child.layout.position_x = 0.0; // Relative to parent
                child.layout.position_y = 0.0; // Relative to parent
                
                // Recursively layout child if it's a View
                // For absolute children, pass the parent's position plus the relative position
                // This ensures children of absolute elements are positioned correctly
                match &mut child.component_type {
                    super::ComponentType::View(ref mut view) => {
                        view.update_layout(
                            parent_x + child.layout.position_x,
                            parent_y + child.layout.position_y,
                            item_main_size,
                            item_cross_size,
                            child.style.padding
                        );
                    }
                    _ => {}
                }
            }
            return; // Absolute layer is done, don't process flex layout
        }
        
        // Calculate available space (accounting for padding) for non-absolute children
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
                super::Size::Pixels(value) => total_fixed_main += value,
                super::Size::Percent(value) => total_fixed_main += available_main * value / 100.0,
                super::Size::Flex(value) => {
                    flex_item_count += 1;
                    total_flex_grow += value.max(0.0);
                }
                super::Size::Auto => {
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
            // In C++: Size childWidth = child->width.getValue() == 0 ? child->calculateWidth() : child->width;
            let main_size = if self.flex_direction == super::layout::FlexDirection::Row {
                match &child.style.width {
                    super::Size::Auto => child.calculate_width(),
                    super::Size::Pixels(0.0) => child.calculate_width(),
                    _ => child.style.width.clone(),
                }
            } else {
                match &child.style.height {
                    super::Size::Auto => child.calculate_height(),
                    super::Size::Pixels(0.0) => child.calculate_height(),
                    _ => child.style.height.clone(),
                }
            };
            
            let item_main_size = match &main_size {
                super::Size::Pixels(value) => *value,
                super::Size::Percent(value) => available_main * value / 100.0,
                super::Size::Flex(value) => flex_unit * value.max(0.0),
                super::Size::Auto => flex_unit * 1.0,
            };
            
            // Calculate cross axis size
            // In C++: Size childHeight = child->height.getValue() == 0 ? child->calculateHeight() : child->height;
            let cross_size = if self.flex_direction == super::layout::FlexDirection::Row {
                match &child.style.height {
                    super::Size::Auto => child.calculate_height(),
                    super::Size::Pixels(0.0) => child.calculate_height(),
                    _ => child.style.height.clone(),
                }
            } else {
                match &child.style.width {
                    super::Size::Auto => child.calculate_width(),
                    super::Size::Pixels(0.0) => child.calculate_width(),
                    _ => child.style.width.clone(),
                }
            };
            
            let item_cross_size = match &cross_size {
                super::Size::Pixels(value) => *value,
                super::Size::Percent(value) => available_cross * value / 100.0,
                super::Size::Flex(_) | super::Size::Auto => available_cross,
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
                super::ComponentType::View(ref mut view) => {
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