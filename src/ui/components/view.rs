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
        if self.flex_direction == FlexDirection::Row {
            // Main axis: sum + gaps
            let mut current_width = padding.left + padding.right;
            for child in &self.children {
                if !child.visible || child.absolute { continue; }
                if let super::Size::Pixels(value) = child.calculate_width() {
                    current_width += value + self.gap;
                }
            }
            let result = (current_width - self.gap).max(padding.left + padding.right);
            super::Size::Pixels(result)
        } else {
            // Cross axis: max
            let mut max_width = 0.0f32;
            for child in &self.children {
                if !child.visible || child.absolute { continue; }
                if let super::Size::Pixels(value) = child.calculate_width() {
                    max_width = max_width.max(value);
                }
            }
            super::Size::Pixels(max_width + padding.left + padding.right)
        }
    }

    pub fn calculate_height(&self, padding: super::styles::Padding) -> super::Size {
        if self.flex_direction == FlexDirection::Column {
            // Main axis: sum + gaps  
            let mut current_height = padding.top + padding.bottom;
            for child in &self.children {
                if !child.visible || child.absolute { continue; }
                if let super::Size::Pixels(value) = child.calculate_height() {
                    current_height += value + self.gap;
                }
            }
            let result = (current_height - self.gap).max(padding.top + padding.bottom);
            super::Size::Pixels(result)
        } else {
            // Cross axis: max
            let mut max_height = 0.0f32;
            for child in &self.children {
                if !child.visible || child.absolute { continue; }
                if let super::Size::Pixels(value) = child.calculate_height() {
                    max_height = max_height.max(value);
                }
            }
            super::Size::Pixels(max_height + padding.top + padding.bottom)
        }
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

        // Check if this layer contains absolute positioned children
        let has_absolute_children = visible_indices.iter()
            .any(|&idx| self.children[idx].absolute);

        if has_absolute_children {
            self.update_absolute_layout(&visible_indices, parent_x, parent_y, parent_width, parent_height, parent_padding);
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
                            super::Size::Flex(_) => 0.0, // Flex items have no natural size
                            super::Size::Auto => 0.0,
                        }
                    } else {
                        match child.calculate_height() {
                            super::Size::Pixels(val) => val,
                            super::Size::Percent(percent) => available_main_size * percent / 100.0,
                            super::Size::Flex(_) => 0.0, // Flex items have no natural size
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
                // For text components, use calculated width even with explicit CSS width
                // This allows text to dynamically resize based on content
                if matches!(child.component_type, super::ComponentType::Text(_)) {
                    child.calculate_width()
                } else {
                    child.style.width.clone()
                }
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
                super::Size::Flex(value) => (flex_unit * value.max(0.0)).max(natural_main),
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

            // Resolve margin values based on child's computed dimensions
            let resolved_margin = child.style.margin.resolve_to_pixels(
                parent_width,
                parent_height
            );

            if is_row {
                child.layout.position_x = current_main_pos - parent_x + resolved_margin.left;
                child.layout.position_y = adjusted_cross_pos - parent_y + resolved_margin.top;
            } else {
                child.layout.position_x = adjusted_cross_pos - parent_x + resolved_margin.left;
                child.layout.position_y = current_main_pos - parent_y + resolved_margin.top;
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

    fn update_absolute_layout(&mut self, indices: &[usize], parent_x: f32, parent_y: f32, parent_width: f32, parent_height: f32, parent_padding: super::styles::Padding) {
        for &idx in indices {
            if idx >= self.children.len() {
                continue;
            }

            let child = &mut self.children[idx];
            if !child.absolute {
                continue; // Skip non-absolute children in absolute layout
            }

            // Calculate child dimensions first
            let child_width = if matches!(child.style.width, super::Size::Auto) {
                // For auto width, use calculated width
                match child.calculate_width() {
                    super::Size::Pixels(val) => val,
                    super::Size::Percent(p) => parent_width * p / 100.0,
                    _ => parent_width, // fallback
                }
            } else {
                child.style.width.to_pixels(parent_width)
            };

            let child_height = if matches!(child.style.height, super::Size::Auto) {
                // For auto height, use calculated height
                match child.calculate_height() {
                    super::Size::Pixels(val) => val,
                    super::Size::Percent(p) => parent_height * p / 100.0,
                    _ => parent_height, // fallback
                }
            } else {
                child.style.height.to_pixels(parent_height)
            };

            // Calculate position based on top/left/right/bottom properties
            let mut x = parent_x;
            let mut y = parent_y;

            // Handle horizontal positioning
            if let Some(left) = child.style.left {
                x += left.to_pixels(parent_width);
            } else if let Some(right) = child.style.right {
                x += parent_width - child_width - right.to_pixels(parent_width);
            } else {
                // Default to left: 0 if neither left nor right is specified
                x += 0.0;
            }

            // Handle vertical positioning
            if let Some(top) = child.style.top {
                y += top.to_pixels(parent_height);
            } else if let Some(bottom) = child.style.bottom {
                y += parent_height - child_height - bottom.to_pixels(parent_height);
            } else {
                // Default to top: 0 if neither top nor bottom is specified
                y += 0.0;
            }

            // Apply padding offset
            x += parent_padding.left;
            y += parent_padding.top;

            // Set computed layout
            child.layout.position_x = x - parent_x;
            child.layout.position_y = y - parent_y;
            child.layout.computed_width = child_width;
            child.layout.computed_height = child_height;

            // Recursively update child layout if it's a view
            if let super::ComponentType::View(ref mut view) = child.component_type {
                view.update_layout(
                    x,
                    y,
                    child_width,
                    child_height,
                    child.style.padding,
                );
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