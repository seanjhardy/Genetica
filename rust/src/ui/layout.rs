// Flexbox layout engine for UI components

use super::styles::Size;
use super::component::Component;

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
pub struct Layout {
    pub flex_direction: FlexDirection,
    pub justify_content: Alignment,
    pub align_items: Alignment,
    pub gap: f32,
    pub position_x: f32,
    pub position_y: f32,
    pub computed_width: f32,
    pub computed_height: f32,
}

impl Layout {
    pub fn new() -> Self {
        Self {
            flex_direction: FlexDirection::Row,
            justify_content: Alignment::Start,
            align_items: Alignment::Start,
            gap: 0.0,
            position_x: 0.0,
            position_y: 0.0,
            computed_width: 0.0,
            computed_height: 0.0,
        }
    }

    pub fn default() -> Self {
        Self::new()
    }

    pub fn compute_layout(&mut self, component: &mut Component, available_width: f32, available_height: f32) {
        // Calculate available space after padding
        let padding = component.style.padding;
        let available_content_width = available_width - padding.left - padding.right;
        let available_content_height = available_height - padding.top - padding.bottom;

        // Extract style and layout properties before borrowing component
        let style_width = component.style.width;
        let style_height = component.style.height;
        let flex_dir = self.flex_direction;
        // For View components, determine alignments based on flex direction (matching C++ exactly)
        // rowAlignment = main axis alignment when flexDirection == Row
        // columnAlignment = main axis alignment when flexDirection == Column
        // columnAlignment = cross axis alignment when flexDirection == Row
        // rowAlignment = cross axis alignment when flexDirection == Column
        // For View components, use View's properties directly (matching C++ exactly)
        // For non-View components, use Layout's properties
        let (main_alignment, cross_alignment, gap) = if matches!(component.component_type, super::component::ComponentType::View(_)) {
            if let super::component::ComponentType::View(ref view) = component.component_type {
                let (main, cross) = if flex_dir == FlexDirection::Row {
                    (view.row_alignment, view.column_alignment)
                } else {
                    (view.column_alignment, view.row_alignment)
                };
                (main, cross, view.gap)
            } else {
                (self.justify_content, self.align_items, self.gap)
            }
        } else {
            (self.justify_content, self.align_items, self.gap)
        };

        // Check component type and compute layout
        let is_view = matches!(component.component_type, super::component::ComponentType::View(_));
        
        if is_view {
            // For View components, we need to access children
            if let super::component::ComponentType::View(ref mut view) = component.component_type {
                // If width/height is Auto or 0, use calculate_width/calculate_height to get minimum size
                // This matches the C++ behavior where components only take up minimum space needed
                let min_width = match style_width {
                    Size::Auto | Size::Pixels(0.0) => {
                        match view.calculate_width(padding) {
                            Size::Pixels(value) => value,
                            _ => available_content_width,
                        }
                    }
                    _ => available_content_width,
                };
                
                let min_height = match style_height {
                    Size::Auto | Size::Pixels(0.0) => {
                        match view.calculate_height(padding) {
                            Size::Pixels(value) => value,
                            _ => available_content_height,
                        }
                    }
                    _ => available_content_height,
                };
                
                // Use minimum size for Auto/0, or available size for other sizes
                let effective_width = match style_width {
                    Size::Auto | Size::Pixels(0.0) => min_width,
                    _ => available_content_width,
                };
                
                let effective_height = match style_height {
                    Size::Auto | Size::Pixels(0.0) => min_height,
                    _ => available_content_height,
                };
                
                // Compute flex layout with the children
                // Pass main_alignment as justify_content and cross_alignment as align_items
                self.compute_flex_layout_with_style(
                    style_width,
                    style_height,
                    flex_dir,
                    main_alignment,
                    cross_alignment,
                    gap,
                    &mut view.children,
                    effective_width,
                    effective_height,
                );
                
                // If width/height was Auto or 0, use the calculated minimum size
                // This ensures components only take up minimum space needed (matching C++ behavior)
                if matches!(style_width, Size::Auto | Size::Pixels(0.0)) {
                    self.computed_width = min_width;
                }
                if matches!(style_height, Size::Auto | Size::Pixels(0.0)) {
                    self.computed_height = min_height;
                }
                
                // Also handle empty children case - if no children, use minimum size (padding only)
                if view.children.iter().all(|c| !c.visible || c.absolute) {
                    if matches!(style_width, Size::Auto | Size::Pixels(0.0)) {
                        self.computed_width = padding.left + padding.right;
                    }
                    if matches!(style_height, Size::Auto | Size::Pixels(0.0)) {
                        self.computed_height = padding.top + padding.bottom;
                    }
                }
            }
        } else {
            // For leaf components (Text, Button), set their size
            // Text/Button components need minimum height based on font size
            let min_height = match &component.component_type {
                super::component::ComponentType::Text(text) => text.font_size,
                super::component::ComponentType::Button(button) => button.font_size,
                _ => 0.0,
            };
            
            self.compute_component_size_for_style_with_min_height(
                style_width, 
                style_height, 
                available_content_width, 
                available_content_height,
                min_height,
            );
        }

        // Set final computed size - update component's layout directly
        component.layout.computed_width = self.computed_width;
        component.layout.computed_height = self.computed_height;
    }
    
    fn compute_component_size_for_style(&mut self, width: Size, height: Size, available_width: f32, available_height: f32) {
        self.compute_component_size_for_style_with_min_height(width, height, available_width, available_height, 0.0);
    }
    
    fn compute_component_size_for_style_with_min_height(&mut self, width: Size, height: Size, available_width: f32, available_height: f32, min_height: f32) {
        let computed_width = match width {
            Size::Pixels(value) => value,
            Size::Percent(value) => available_width * value / 100.0,
            Size::Flex(_) | Size::Auto => available_width,
        };

        let computed_height = match height {
            Size::Pixels(value) => value,
            Size::Percent(value) => available_height * value / 100.0,
            Size::Flex(_) | Size::Auto => available_height.max(min_height),
        };

        self.computed_width = computed_width;
        self.computed_height = computed_height.max(min_height);
    }

    fn compute_flex_layout_with_style(
        &mut self,
        parent_width: Size,
        parent_height: Size,
        flex_direction: FlexDirection,
        justify_content: Alignment,
        align_items: Alignment,
        gap: f32,
        children: &mut Vec<Component>,
        available_width: f32,
        available_height: f32,
    ) {
        // Set layout properties
        self.flex_direction = flex_direction;
        self.justify_content = justify_content;
        self.align_items = align_items;
        self.gap = gap;
        
        self.compute_flex_layout(children, available_width, available_height, parent_width, parent_height);
    }
    
    fn compute_flex_layout(
        &mut self,
        children: &mut Vec<Component>,
        available_width: f32,
        available_height: f32,
        parent_width: Size,
        parent_height: Size,
    ) {
        // Filter visible children
        let visible_children: Vec<usize> = children
            .iter()
            .enumerate()
            .filter(|(_, child)| child.visible)
            .map(|(i, _)| i)
            .collect();

        if visible_children.is_empty() {
            // Use the parent's style for size computation
            let width = match parent_width {
                Size::Pixels(value) => value,
                Size::Percent(value) => available_width * value / 100.0,
                Size::Flex(_) | Size::Auto => available_width,
            };
            let height = match parent_height {
                Size::Pixels(value) => value,
                Size::Percent(value) => available_height * value / 100.0,
                Size::Flex(_) | Size::Auto => available_height,
            };
            self.computed_width = width;
            self.computed_height = height;
            return;
        }

        let is_row = self.flex_direction == FlexDirection::Row;
        let main_size = if is_row { available_width } else { available_height };
        let cross_size = if is_row { available_height } else { available_width };

        // Calculate gap space
        let gap_space = self.gap * (visible_children.len() as f32 - 1.0);
        let available_main = main_size - gap_space;

        // First pass: calculate sizes for fixed and percent items, count flex items
        let mut total_fixed_main = 0.0;
        let mut flex_item_count = 0;
        let mut total_flex_grow = 0.0;

        for &idx in &visible_children {
            let child = &children[idx];
            let main_size_prop = if is_row { &child.style.width } else { &child.style.height };

            match main_size_prop {
                Size::Pixels(value) => total_fixed_main += value,
                Size::Percent(value) => total_fixed_main += available_main * value / 100.0,
                Size::Flex(value) => {
                    flex_item_count += 1;
                    total_flex_grow += value.max(0.0);
                }
                Size::Auto => {
                    flex_item_count += 1;
                    total_flex_grow += 1.0; // Default flex grow of 1
                }
            }
        }

        // Calculate flex unit
        let remaining_main = (available_main - total_fixed_main).max(0.0);
        let flex_unit = if flex_item_count > 0 && total_flex_grow > 0.0 {
            remaining_main / total_flex_grow
        } else {
            0.0
        };

        // Second pass: set sizes and positions
        let mut current_main_pos = 0.0;

        for &idx in &visible_children {
            let child = &mut children[idx];
            
            // Calculate main axis size
            let main_size_prop = if is_row { &child.style.width } else { &child.style.height };
            let item_main_size = match main_size_prop {
                Size::Pixels(value) => *value,
                Size::Percent(value) => available_main * value / 100.0,
                Size::Flex(value) => flex_unit * value.max(0.0),
                Size::Auto => flex_unit * 1.0,
            };

            // Calculate cross axis size
            let cross_size_prop = if is_row { &child.style.height } else { &child.style.width };
            let item_cross_size = match cross_size_prop {
                Size::Pixels(value) => *value,
                Size::Percent(value) => cross_size * value / 100.0,
                Size::Flex(_) | Size::Auto => cross_size,
            };

            // Set component size
            if is_row {
                child.layout.computed_width = item_main_size;
                child.layout.computed_height = item_cross_size;
            } else {
                child.layout.computed_width = item_cross_size;
                child.layout.computed_height = item_main_size;
            }

            // Set position
            if is_row {
                child.layout.position_x = current_main_pos;
                child.layout.position_y = 0.0; // Will be adjusted by align_items
            } else {
                child.layout.position_x = 0.0; // Will be adjusted by align_items
                child.layout.position_y = current_main_pos;
            }

            // Recursively compute layout for children
            let is_child_view = matches!(child.component_type, super::component::ComponentType::View(_));
            if is_child_view {
                let child_main = if is_row { item_main_size } else { item_cross_size };
                let child_cross = if is_row { item_cross_size } else { item_main_size };
                // Compute layout - create a temporary layout to compute, then copy back
                let mut temp_layout = child.layout.clone();
                temp_layout.compute_layout(child, child_main, child_cross);
                child.layout = temp_layout;
            }

            current_main_pos += item_main_size + self.gap;
        }

        // Apply main axis alignment
        let total_main_size = current_main_pos - self.gap;
        let extra_space = available_main - total_main_size;

        if extra_space > 0.0 {
            match self.justify_content {
                Alignment::Center => {
                    let offset = extra_space / 2.0;
                    for &idx in &visible_children {
                        if is_row {
                            children[idx].layout.position_x += offset;
                        } else {
                            children[idx].layout.position_y += offset;
                        }
                    }
                }
                Alignment::End => {
                    for &idx in &visible_children {
                        if is_row {
                            children[idx].layout.position_x += extra_space;
                        } else {
                            children[idx].layout.position_y += extra_space;
                        }
                    }
                }
                Alignment::SpaceBetween => {
                    if visible_children.len() > 1 {
                        let gap = extra_space / (visible_children.len() - 1) as f32;
                        let mut offset = 0.0;
                        for &idx in &visible_children[1..] {
                            offset += gap;
                            if is_row {
                                children[idx].layout.position_x += offset;
                            } else {
                                children[idx].layout.position_y += offset;
                            }
                        }
                    }
                }
                Alignment::SpaceAround => {
                    if !visible_children.is_empty() {
                        let gap = extra_space / visible_children.len() as f32;
                        let mut offset = gap / 2.0;
                        for &idx in &visible_children[1..] {
                            offset += gap;
                            if is_row {
                                children[idx].layout.position_x += offset;
                            } else {
                                children[idx].layout.position_y += offset;
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // Apply cross axis alignment
        for &idx in &visible_children {
            let child = &children[idx];
            let child_cross_size = if is_row { child.layout.computed_height } else { child.layout.computed_width };
            
            match self.align_items {
                Alignment::Center => {
                    let offset = (cross_size - child_cross_size) / 2.0;
                    if is_row {
                        children[idx].layout.position_y += offset;
                    } else {
                        children[idx].layout.position_x += offset;
                    }
                }
                Alignment::End => {
                    let offset = cross_size - child_cross_size;
                    if is_row {
                        children[idx].layout.position_y += offset;
                    } else {
                        children[idx].layout.position_x += offset;
                    }
                }
                Alignment::Stretch => {
                    if is_row {
                        children[idx].layout.computed_height = cross_size;
                    } else {
                        children[idx].layout.computed_width = cross_size;
                    }
                }
                _ => {}
            }
        }

        // Set parent size - update self (the layout) instead of parent
        // Only update if not already set (i.e., if width/height was Auto, it was already set above)
        let parent_main_size = if is_row { available_width } else { available_height };
        let parent_cross_size = if is_row { available_height } else { available_width };
        
        // Only update if not already computed (for Auto/0 sizes, we already set it)
        if self.computed_width == 0.0 {
            self.computed_width = if is_row { parent_main_size } else { parent_cross_size };
        }
        if self.computed_height == 0.0 {
            self.computed_height = if is_row { parent_cross_size } else { parent_main_size };
        }
    }

}

impl Default for Layout {
    fn default() -> Self {
        Self::new()
    }
}
