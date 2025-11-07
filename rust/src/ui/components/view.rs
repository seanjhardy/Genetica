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
        let mut current_width = padding.left + padding.right;
        for child in &self.children {
            if !child.visible || child.absolute {
                continue;
            }
            let child_width = child.calculate_width();
            if let super::Size::Pixels(value) = child_width {
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
        let layers_snapshot = self.layers.clone();
        for layer in layers_snapshot {
            self.update_layer_layout(&layer, parent_x, parent_y, parent_width, parent_height, parent_padding);
        }
    }

    fn update_layer_layout(&mut self, layer: &[usize], parent_x: f32, parent_y: f32, parent_width: f32, parent_height: f32, parent_padding: super::styles::Padding) {
        let visible_indices: Vec<usize> = layer
            .iter()
            .copied()
            .filter(|&idx| idx < self.children.len() && self.children[idx].visible)
            .collect();

        if visible_indices.is_empty() {
            return;
        }

        let is_row = self.flex_direction == FlexDirection::Row;
        let padding = parent_padding;

        let container_width = (parent_width - padding.left - padding.right).max(0.0);
        let container_height = (parent_height - padding.top - padding.bottom).max(0.0);

        let mut available_main_size = if is_row { container_width } else { container_height };
        let available_cross_size = if is_row { container_height } else { container_width };

        let num_children = visible_indices.len();
        let total_gap = self.gap * (num_children as f32 - 1.0);
        available_main_size = (available_main_size - total_gap).max(0.0);
        
        // Debug logging for button parent containers and their parents
        let has_button_child = visible_indices.iter().any(|&idx| {
            if let Some(ref id) = self.children[idx].id {
                id.contains("Btn")
            } else {
                false
            }
        });
        
        // Also check if any child is a parent of buttons (button group)
        let has_button_grandchild = visible_indices.iter().any(|&idx| {
            if let super::ComponentType::View(ref view) = self.children[idx].component_type {
                view.children.iter().any(|gc| {
                    if let Some(ref id) = gc.id {
                        id.contains("Btn")
                    } else {
                        false
                    }
                })
            } else {
                false
            }
        });
        
        // Check if this is the controlPanel
        let is_control_panel = visible_indices.iter().any(|&idx| {
            if let Some(ref id) = self.children[idx].id {
                id == "controlPanel"
            } else {
                false
            }
        });

        let mut total_fixed_main = 0.0;
        let mut flex_item_count = 0;
        let mut total_flex_grow = 0.0;

        for &idx in &visible_indices {
            let child = &self.children[idx];
            let main_size = if is_row { &child.style.width } else { &child.style.height };

            match main_size {
                super::Size::Pixels(value) => total_fixed_main += *value,
                super::Size::Percent(value) => total_fixed_main += available_main_size * value / 100.0,
                super::Size::Flex(value) => {
                    flex_item_count += 1;
                    total_flex_grow += value.max(0.0);
                }
                super::Size::Auto => {
                    let natural_main = if is_row {
                        match child.calculate_width() {
                            super::Size::Pixels(val) => val,
                            super::Size::Percent(percent) => available_main_size * percent / 100.0,
                            super::Size::Flex(val) => available_main_size * val.max(0.0),
                            super::Size::Auto => 0.0,
                        }
                    } else {
                        match child.calculate_height() {
                            super::Size::Pixels(val) => val,
                            super::Size::Percent(percent) => available_main_size * percent / 100.0,
                            super::Size::Flex(val) => available_main_size * val.max(0.0),
                            super::Size::Auto => 0.0,
                        }
                    };
                    total_fixed_main += natural_main;
                }
            }
        }

        let remaining_main = (available_main_size - total_fixed_main).max(0.0);
        let flex_unit = if flex_item_count > 0 && total_flex_grow > 0.0 {
            remaining_main / total_flex_grow
        } else {
            0.0
        };

        let total_space_taken = if flex_item_count > 0 {
            available_main_size
        } else {
            total_fixed_main
        };

        let mut current_main_pos = if is_row {
            parent_x + padding.left
        } else {
            parent_y + padding.top
        };

        for &idx in &visible_indices {
            let child = &mut self.children[idx];

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

            let main_size = if is_row {
                child_width.clone()
            } else {
                child_height.clone()
            };

            let cross_size = if is_row {
                child_height.clone()
            } else {
                child_width.clone()
            };

            let natural_main = if is_row {
                match child.calculate_width() {
                    super::Size::Pixels(val) => val,
                    super::Size::Percent(percent) => available_main_size * percent / 100.0,
                    super::Size::Flex(val) => available_main_size * val.max(0.0),
                    super::Size::Auto => 0.0,
                }
            } else {
                match child.calculate_height() {
                    super::Size::Pixels(val) => val,
                    super::Size::Percent(percent) => available_main_size * percent / 100.0,
                    super::Size::Flex(val) => available_main_size * val.max(0.0),
                    super::Size::Auto => 0.0,
                }
            };

            let natural_cross = if is_row {
                match child.calculate_height() {
                    super::Size::Pixels(val) => val,
                    super::Size::Percent(percent) => available_cross_size * percent / 100.0,
                    super::Size::Flex(val) => available_cross_size * val.max(0.0),
                    super::Size::Auto => 0.0,
                }
            } else {
                match child.calculate_width() {
                    super::Size::Pixels(val) => val,
                    super::Size::Percent(percent) => available_cross_size * percent / 100.0,
                    super::Size::Flex(val) => available_cross_size * val.max(0.0),
                    super::Size::Auto => 0.0,
                }
            };

            let item_main_size = match &main_size {
                super::Size::Pixels(value) => *value,
                super::Size::Percent(value) => available_main_size * value / 100.0,
                super::Size::Flex(value) => flex_unit * value.max(0.0),
                super::Size::Auto => {
                    if natural_main > 0.0 {
                        natural_main
                    } else if flex_item_count > 0 && total_flex_grow > 0.0 {
                        flex_unit
                    } else {
                        available_main_size
                    }
                }
            };

            let item_cross_size = match &cross_size {
                super::Size::Pixels(value) => *value,
                super::Size::Percent(value) => available_cross_size * value / 100.0,
                super::Size::Flex(_) => available_cross_size,
                super::Size::Auto => {
                    if natural_cross > 0.0 {
                        natural_cross
                    } else {
                        available_cross_size
                    }
                }
            };

            if is_row {
                child.layout.computed_width = item_main_size;
                child.layout.computed_height = item_cross_size;
            } else {
                child.layout.computed_width = item_cross_size;
                child.layout.computed_height = item_main_size;
            }

            let cross_alignment = self.column_alignment;

            let cross_pos = if is_row {
                parent_y + padding.top
            } else {
                parent_x + padding.left
            };

            let adjusted_cross_pos = match cross_alignment {
                Alignment::Center => cross_pos + (available_cross_size - item_cross_size) / 2.0,
                Alignment::End => cross_pos + available_cross_size - item_cross_size
                    - if is_row {
                        padding.bottom
                    } else {
                        padding.right
                    },
                _ => cross_pos,
            };

            if is_row {
                child.layout.position_x = current_main_pos - parent_x;
                child.layout.position_y = adjusted_cross_pos - parent_y;
            } else {
                child.layout.position_x = adjusted_cross_pos - parent_x;
                child.layout.position_y = current_main_pos - parent_y;
            }

            if let super::ComponentType::View(ref mut view) = child.component_type {
                view.update_layout(
                    parent_x + child.layout.position_x,
                    parent_y + child.layout.position_y,
                    child.layout.computed_width,
                    child.layout.computed_height,
                    child.style.padding,
                );
            }

            current_main_pos += item_main_size + self.gap;
        }

        let extra_space = (available_main_size - total_space_taken).max(0.0);
        if extra_space > 0.0 {
            let main_alignment = self.row_alignment;

            match main_alignment {
                Alignment::Center => self.apply_main_axis_offset(&visible_indices, extra_space / 2.0),
                Alignment::End => self.apply_main_axis_offset(&visible_indices, extra_space),
                Alignment::SpaceBetween => self.distribute_main_axis_space(&visible_indices, extra_space, false),
                Alignment::SpaceAround => self.distribute_main_axis_space(&visible_indices, extra_space, true),
                _ => {}
            }
        }
    }

    fn apply_main_axis_offset(&mut self, indices: &[usize], offset: f32) {
        if offset == 0.0 {
            return;
        }

        let is_row = self.flex_direction == FlexDirection::Row;
        for &idx in indices {
            if idx >= self.children.len() {
                continue;
            }
            if !self.children[idx].visible {
                continue;
            }
            if is_row {
                self.children[idx].layout.position_x += offset;
            } else {
                self.children[idx].layout.position_y += offset;
            }
        }
    }

    fn distribute_main_axis_space(&mut self, indices: &[usize], space: f32, include_ends: bool) {
        if space <= 0.0 {
            return;
        }

        let is_row = self.flex_direction == FlexDirection::Row;
        let num_visible = indices
            .iter()
            .filter(|&&idx| idx < self.children.len() && self.children[idx].visible)
            .count();

        let divisions = if include_ends {
            num_visible + 1
        } else {
            num_visible.saturating_sub(1)
        };

        if divisions == 0 {
            return;
        }

        let gap = space / divisions as f32;
        let mut current_offset = if include_ends { gap } else { 0.0 };

        for &idx in indices {
            if idx >= self.children.len() {
                continue;
            }
            if !self.children[idx].visible {
                continue;
            }

            if is_row {
                self.children[idx].layout.position_x += current_offset;
            } else {
                self.children[idx].layout.position_y += current_offset;
            }

            current_offset += gap;
        }
    }
}

impl Default for View {
    fn default() -> Self {
        Self::new()
    }
}