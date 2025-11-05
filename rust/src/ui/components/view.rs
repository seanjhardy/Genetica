// Flexbox layout enums
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FlexDirection {
    Row,
    Column,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Alignment {
    Start,
    Center,
    End,
    Stretch,
    SpaceBetween,
    SpaceAround,
}

#[derive(Debug, Clone)]
pub struct View {
    pub children: Vec<super::Component>,
    // View-specific layout properties
    pub flex_direction: FlexDirection,
    pub row_alignment: Alignment,
    pub column_alignment: Alignment,
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
            flex_direction: FlexDirection::Row,
            row_alignment: Alignment::Start,
            column_alignment: Alignment::Start,
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
        // Match C++ exactly: calculateWidth uses child->calculateWidth().getValue()
        let mut current_width = padding.left + padding.right;
        for child in &self.children {
            if !child.visible || child.absolute {
                continue;
            }
            // Match C++: currentWidth += child->calculateWidth().getValue() + gap;
            if let super::Size::Pixels(value) = child.calculate_width() {
                current_width += value + self.gap;
            }
        }
        let result = (current_width - self.gap).max(padding.left + padding.right);
        super::Size::Pixels(result)
    }
    
    pub fn calculate_height(&self, padding: super::styles::Padding) -> super::Size {
        // Match C++ exactly: calculateHeight uses child->calculateHeight().getValue()
        let mut current_height = padding.top + padding.bottom;
        for child in &self.children {
            if !child.visible || child.absolute {
                continue;
            }
            // Match C++: currentHeight += child->calculateHeight().getValue() + gap;
            if let super::Size::Pixels(value) = child.calculate_height() {
                current_height += value + self.gap;
            }
        }
        let result = (current_height - self.gap).max(padding.top + padding.bottom);
        super::Size::Pixels(result)
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
                let main_size = if self.flex_direction == FlexDirection::Row {
                    &child.style.width
                } else {
                    &child.style.height
                };
                
                let item_main_size = match main_size {
                    super::Size::Pixels(value) => *value,
                    super::Size::Percent(value) => parent_width * value / 100.0,
                    super::Size::Flex(_) | super::Size::Auto => {
                        // For absolute, use parent width/height if not specified
                        if self.flex_direction == FlexDirection::Row {
                            parent_width
                        } else {
                            parent_height
                        }
                    }
                };
                
                let cross_size = if self.flex_direction == FlexDirection::Row {
                    &child.style.height
                } else {
                    &child.style.width
                };
                
                let item_cross_size = match cross_size {
                    super::Size::Pixels(value) => *value,
                    super::Size::Percent(value) => {
                        if self.flex_direction == FlexDirection::Row {
                            parent_height * value / 100.0
                        } else {
                            parent_width * value / 100.0
                        }
                    }
                    super::Size::Flex(_) | super::Size::Auto => {
                        if self.flex_direction == FlexDirection::Row {
                            parent_height
                        } else {
                            parent_width
                        }
                    }
                };
                
                // Set component size
                if self.flex_direction == FlexDirection::Row {
                    child.layout.computed_width = item_main_size;
                    child.layout.computed_height = item_cross_size;
                } else {
                    child.layout.computed_width = item_cross_size;
                    child.layout.computed_height = item_main_size;
                }
                
                // Apply alignment for absolute children
                // For Row: main axis is horizontal (row_alignment), cross axis is vertical (column_alignment)
                // For Column: main axis is vertical (row_alignment), cross axis is horizontal (column_alignment)
                // Note: col-* classes control cross-axis, row-* classes control main-axis
                let (main_alignment, cross_alignment) = if self.flex_direction == FlexDirection::Row {
                    (self.row_alignment, self.column_alignment)
                } else {
                    // For Column: main axis is vertical (row_alignment), cross axis is horizontal (column_alignment)
                    (self.row_alignment, self.column_alignment)
                };
                
                let container_width = parent_width - parent_padding.left - parent_padding.right;
                let container_height = parent_height - parent_padding.top - parent_padding.bottom;
                
                let main_size = if self.flex_direction == FlexDirection::Row {
                    container_width
                } else {
                    container_height
                };
                let cross_size = if self.flex_direction == FlexDirection::Row {
                    container_height
                } else {
                    container_width
                };
                
                // Calculate main axis position (x for Row, y for Column)
                let main_pos = match main_alignment {
                    Alignment::Center => (main_size - item_main_size) / 2.0,
                    Alignment::End => main_size - item_main_size,
                    _ => 0.0, // Start
                };
                
                // Calculate cross axis position (y for Row, x for Column)
                let cross_pos = match cross_alignment {
                    Alignment::Center => (cross_size - item_cross_size) / 2.0,
                    Alignment::End => cross_size - item_cross_size,
                    _ => 0.0, // Start
                };
                
                if self.flex_direction == FlexDirection::Row {
                    child.layout.position_x = parent_padding.left + main_pos;
                    child.layout.position_y = parent_padding.top + cross_pos;
                } else {
                    child.layout.position_x = parent_padding.left + cross_pos;
                    child.layout.position_y = parent_padding.top + main_pos;
                }
                
                // Recursively layout child if it's a View
                match &mut child.component_type {
                    super::ComponentType::View(ref mut view) => {
                        view.update_layout(
                            parent_x + child.layout.position_x,
                            parent_y + child.layout.position_y,
                            child.layout.computed_width,
                            child.layout.computed_height,
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
        
        let available_main = if self.flex_direction == FlexDirection::Row {
            container_width
        } else {
            container_height
        };
        let available_cross = if self.flex_direction == FlexDirection::Row {
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
            let main_size = if self.flex_direction == FlexDirection::Row {
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
        
        // In C++: totalSpaceTaken = flexItemCount > 0 ? availableMainSize : totalFixedMainSize;
        let total_space_taken = if flex_item_count > 0 {
            available_main
        } else {
            total_fixed_main
        };
        
        // Second pass: set sizes and positions
        // Match C++: currentMainPos starts at padding position
        let mut current_main_pos = if self.flex_direction == FlexDirection::Row {
            parent_x + padding.left
        } else {
            parent_y + padding.top
        };
        
        for &idx in &visible_indices {
            if idx >= self.children.len() {
                continue;
            }
            let child = &mut self.children[idx];
            
            // Match C++ exactly: childWidth = child->width.getValue() == 0 ? child->calculateWidth() : child->width;
            let child_width = if matches!(child.style.width, super::Size::Pixels(0.0)) {
                match child.calculate_width() {
                    super::Size::Pixels(val) => super::Size::Pixels(val),
                    _ => child.style.width.clone(),
                }
            } else {
                child.style.width.clone()
            };
            
            let child_height = if matches!(child.style.height, super::Size::Pixels(0.0)) {
                match child.calculate_height() {
                    super::Size::Pixels(val) => super::Size::Pixels(val),
                    _ => child.style.height.clone(),
                }
            } else {
                child.style.height.clone()
            };
            
            let main_size = if self.flex_direction == FlexDirection::Row {
                child_width.clone()
            } else {
                child_height.clone()
            };
            
            let cross_size = if self.flex_direction == FlexDirection::Row {
                child_height.clone()
            } else {
                child_width.clone()
            };
            
            // Calculate main axis size - match C++ switch statement exactly
            let item_main_size = match &main_size {
                super::Size::Pixels(value) => *value,
                super::Size::Percent(value) => available_main * value / 100.0,
                super::Size::Flex(value) => flex_unit * value.max(0.0),
                super::Size::Auto => flex_unit * 1.0,
            };
            
            // Calculate cross axis size - match C++ switch statement exactly
            let item_cross_size = match &cross_size {
                super::Size::Pixels(value) => *value,
                super::Size::Percent(value) => available_cross * value / 100.0,
                super::Size::Flex(_) | super::Size::Auto => available_cross,
            };
            
            // Set element size - match C++ exactly
            if self.flex_direction == FlexDirection::Row {
                child.layout.computed_width = item_main_size;
                child.layout.computed_height = item_cross_size;
            } else {
                child.layout.computed_width = item_cross_size;
                child.layout.computed_height = item_main_size;
            }
            
            // Set element position - match C++: alignment for cross axis
            // For Row: cross axis is vertical (column_alignment)
            // For Column: cross axis is horizontal (row_alignment) - note: C++ uses rowAlignment for Column cross axis
            let cross_alignment = if self.flex_direction == FlexDirection::Row {
                self.column_alignment
            } else {
                self.row_alignment
            };
            
            let cross_pos = if self.flex_direction == FlexDirection::Row {
                parent_y + padding.top
            } else {
                parent_x + padding.left
            };
            
            let adjusted_cross_pos = match cross_alignment {
                Alignment::Center => {
                    cross_pos + (available_cross - item_cross_size) / 2.0
                }
                Alignment::End => {
                    cross_pos + available_cross - item_cross_size -
                    (if self.flex_direction == FlexDirection::Row {
                        padding.bottom
                    } else {
                        padding.right
                    })
                }
                _ => cross_pos,
            };
            
            // Set element position - match C++ exactly
            if self.flex_direction == FlexDirection::Row {
                child.layout.position_x = current_main_pos - parent_x;
                child.layout.position_y = adjusted_cross_pos - parent_y;
            } else {
                child.layout.position_x = adjusted_cross_pos - parent_x;
                child.layout.position_y = current_main_pos - parent_y;
            }
            
            // Recursively layout child if it's a View
            match &mut child.component_type {
                super::ComponentType::View(ref mut view) => {
                    view.update_layout(
                        parent_x + child.layout.position_x,
                        parent_y + child.layout.position_y,
                        child.layout.computed_width,
                        child.layout.computed_height,
                        child.style.padding
                    );
                }
                _ => {}
            }
            
            current_main_pos += item_main_size + self.gap;
        }
        
        // Apply main axis alignment - match C++ exactly
        // Match C++: mainAlignment = (flexDirection == Row) ? rowAlignment : columnAlignment
        let extra_space = (available_main - total_space_taken).max(0.0);
        
        if extra_space > 0.0 {
            // Match C++ line 264 exactly
            let main_alignment = if self.flex_direction == FlexDirection::Row {
                self.row_alignment
            } else {
                self.column_alignment
            };
            
            match main_alignment {
                Alignment::Center => {
                    let offset = extra_space / 2.0;
                    for &idx in &visible_indices {
                        if idx < self.children.len() {
                            if self.flex_direction == FlexDirection::Row {
                                self.children[idx].layout.position_x += offset;
                            } else {
                                self.children[idx].layout.position_y += offset;
                            }
                        }
                    }
                }
                Alignment::End => {
                    for &idx in &visible_indices {
                        if idx < self.children.len() {
                            if self.flex_direction == FlexDirection::Row {
                                self.children[idx].layout.position_x += extra_space;
                            } else {
                                self.children[idx].layout.position_y += extra_space;
                            }
                        }
                    }
                }
                Alignment::SpaceBetween => {
                    if visible_indices.len() > 1 {
                        let gap = extra_space / (visible_indices.len() - 1) as f32;
                        let mut offset = 0.0;
                        for &idx in &visible_indices[1..] {
                            if idx < self.children.len() {
                                offset += gap;
                                if self.flex_direction == FlexDirection::Row {
                                    self.children[idx].layout.position_x += offset;
                                } else {
                                    self.children[idx].layout.position_y += offset;
                                }
                            }
                        }
                    }
                }
                Alignment::SpaceAround => {
                    if !visible_indices.is_empty() {
                        let gap = extra_space / visible_indices.len() as f32;
                        let mut offset = gap / 2.0;
                        for &idx in &visible_indices[1..] {
                            if idx < self.children.len() {
                                offset += gap;
                                if self.flex_direction == FlexDirection::Row {
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