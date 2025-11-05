// HTML and CSS parser for UI definitions

use super::components::{Component, ComponentType, View, Text};
use super::styles::{Style, Color, Border, Shadow, Padding, Margin, Size};
use super::components::view::{FlexDirection, Alignment};
use std::collections::HashMap;

#[derive(Debug)]
pub struct UiParser {
    css_classes: HashMap<String, HashMap<String, String>>,
}

impl UiParser {
    pub fn new() -> Self {
        Self {
            css_classes: HashMap::new(),
        }
    }

    pub fn parse_from_files(html_path: &str, css_paths: &[&str]) -> Result<Component, String> {
        let html_content = std::fs::read_to_string(html_path)
            .map_err(|e| format!("Failed to read HTML file {}: {}", html_path, e))?;
        
        let mut parser = Self::new();
        
        // Parse all CSS files in order
        for css_path in css_paths {
            let css_content = std::fs::read_to_string(css_path)
                .map_err(|e| format!("Failed to read CSS file {}: {}", css_path, e))?;
            parser.parse_css(&css_content)?;
        }
        
        parser.parse_html(&html_content)
    }

    
    // Parse HTML and create a Screen with the root component
    pub fn parse_to_screen(html_path: &str, css_paths: &[&str]) -> Result<super::screen::Screen, String> {
        let component = Self::parse_from_files(html_path, css_paths)?;
        let mut screen = super::screen::Screen::new();
        screen.add_element(component);
        Ok(screen)
    }

    pub fn parse_css(&mut self, css_content: &str) -> Result<(), String> {
        // Remove comments
        let css_content = Self::remove_css_comments(css_content);
        
        // Simple CSS parser - handles classes and basic selectors
        let rule_pattern = regex::Regex::new(r"([^{]+)\{([^}]+)\}")
            .map_err(|e| format!("Failed to create regex: {}", e))?;

        for cap in rule_pattern.captures_iter(&css_content) {
            let selector = cap[1].trim().to_string();
            let properties_str = cap[2].trim();

            let mut properties = HashMap::new();
            for prop in properties_str.split(';') {
                let prop = prop.trim();
                if prop.is_empty() {
                    continue;
                }
                
                if let Some(colon_pos) = prop.find(':') {
                    let key = prop[..colon_pos].trim().to_string();
                    let value = prop[colon_pos + 1..].trim().to_string();
                    properties.insert(key, value);
                }
            }

            if selector.starts_with('.') {
                // CSS class
                let class_name = selector[1..].trim().to_string();
                self.css_classes.insert(class_name, properties);
            }
        }

        Ok(())
    }

    fn remove_css_comments(css: &str) -> String {
        let comment_pattern = regex::Regex::new(r"/\*.*?\*/").unwrap();
        comment_pattern.replace_all(css, "").to_string()
    }

    pub fn parse_html(&self, html_content: &str) -> Result<Component, String> {
        
        // Find the body or root element FIRST (before normalization removes body tags)
        let body_pattern = regex::Regex::new(r"<body[^>]*>(.*?)</body>")
            .map_err(|e| format!("Failed to create regex: {}", e))?;
        
        if let Some(body_cap) = body_pattern.captures(&html_content) {
            let mut body_content = body_cap[1].to_string();
            
            // Normalize the body content (remove comments, normalize whitespace, but don't remove body tags since they're already extracted)
            body_content = Self::normalize_html(&body_content);
            
            // Parse the first element in body
            let root_pattern = regex::Regex::new(r"<(\w+)([^>]*)>")
                .map_err(|e| format!("Failed to create regex: {}", e))?;
            
            if let Some(root_cap) = root_pattern.captures(&body_content) {
                let tag_name = root_cap[1].to_string();
                let attributes = Self::parse_attributes(&root_cap[2]);
               
                // Find the matching closing tag manually (regex backreferences don't work)
                let _tag_start = root_cap.get(0).unwrap().start();
                let tag_end = root_cap.get(0).unwrap().end(); 
                
                // Track depth to find the matching closing tag
                let mut depth = 1;
                let mut search_pos = tag_end;
                let mut inner_content = String::new();
                let mut iteration = 0;
                
                while search_pos < body_content.len() && depth > 0 {
                    iteration += 1;
                    if iteration > 100 {
                        break;
                    }
                        
                    // Look for both opening and closing tags
                    let open_tag_pattern = format!("<{}", tag_name);
                    let close_tag_pattern = format!("</{}>", tag_name);
                    
                    // Find the next occurrence of either opening or closing tag
                    let next_open = body_content[search_pos..].find(&open_tag_pattern);
                    let next_close = body_content[search_pos..].find(&close_tag_pattern);
                    
                    let next_pos = match (next_open, next_close) {
                        (Some(open_pos), Some(close_pos)) => {
                            if open_pos < close_pos {
                                Some((open_pos + search_pos, true))
                            } else {
                                Some((close_pos + search_pos, false))
                            }
                        }
                        (Some(open_pos), None) => Some((open_pos + search_pos, true)),
                        (None, Some(close_pos)) => Some((close_pos + search_pos, false)),
                        (None, None) => None,
                    };
                    
                    if let Some((pos, is_open)) = next_pos {
                        if is_open {
                            // Check if this is actually an opening tag (not part of a closing tag)
                            let tag_start_pos = pos;
                            let tag_end_pos = body_content[tag_start_pos..].find('>');
                            if let Some(end_pos) = tag_end_pos {
                                let tag_str = &body_content[tag_start_pos..tag_start_pos + end_pos + 1];
                                // Make sure it's not a self-closing tag or closing tag
                                if !tag_str.contains('/') && tag_str.trim().starts_with(&format!("<{}", tag_name)) {
                                    depth += 1;
                                    search_pos = tag_start_pos + end_pos + 1;
                                    continue;
                                } else {
                                    search_pos = tag_start_pos + end_pos + 1;
                                    continue;
                                }
                            }
                            search_pos = pos + 1;
                        } else {
                            // Found a closing tag
                            depth -= 1;
                            if depth == 0 {
                                // Found the matching closing tag
                                inner_content = body_content[tag_end..pos].to_string();
                                break;
                            }
                            search_pos = pos + close_tag_pattern.len();
                        }
                    } else {
                        // No more tags found
                        break;
                    }
                }
                
                if inner_content.is_empty() {
                }
                
                self.parse_element(&tag_name, &attributes, &inner_content, super::inheritance::InheritableProperties::new())
            } else {
                Err("Could not find root element in HTML body".to_string())
            }
        } else {
            // No body tag found, normalize and try to find root element directly
            let html_content = Self::normalize_html(html_content);
            
            // Find the first opening tag
            let root_pattern = regex::Regex::new(r"<(\w+)([^>]*)>")
                .map_err(|e| format!("Failed to create regex: {}", e))?;
            
            if let Some(root_cap) = root_pattern.captures(&html_content) {
                let tag_name = root_cap[1].to_string();
                let attributes = Self::parse_attributes(&root_cap[2]);
                
                let tag_end = root_cap.get(0).unwrap().end();
                
                // Track depth to find the matching closing tag
                let mut depth = 1;
                let mut search_pos = tag_end;
                let mut inner_content = String::new();
                let mut iteration = 0;
                
                while search_pos < html_content.len() && depth > 0 {
                    iteration += 1;
                    if iteration > 100 {
                        break;
                    }
                    
                    // Look for both opening and closing tags
                    let open_tag_pattern = format!("<{}", tag_name);
                    let close_tag_pattern = format!("</{}>", tag_name);
                    
                    // Find the next occurrence of either opening or closing tag
                    let next_open = html_content[search_pos..].find(&open_tag_pattern);
                    let next_close = html_content[search_pos..].find(&close_tag_pattern);
                    
                    let next_pos = match (next_open, next_close) {
                        (Some(open_pos), Some(close_pos)) => {
                            if open_pos < close_pos {
                                Some((open_pos + search_pos, true))
                            } else {
                                Some((close_pos + search_pos, false))
                            }
                        }
                        (Some(open_pos), None) => Some((open_pos + search_pos, true)),
                        (None, Some(close_pos)) => Some((close_pos + search_pos, false)),
                        (None, None) => None,
                    };
                    
                    if let Some((pos, is_open)) = next_pos {
                        if is_open {
                            // Check if this is actually an opening tag (not part of a closing tag)
                            let tag_start_pos = pos;
                            let tag_end_pos = html_content[tag_start_pos..].find('>');
                            if let Some(end_pos) = tag_end_pos {
                                let tag_str = &html_content[tag_start_pos..tag_start_pos + end_pos + 1];
                                // Make sure it's not a self-closing tag or closing tag
                                if !tag_str.contains('/') && tag_str.trim().starts_with(&format!("<{}", tag_name)) {
                                    depth += 1;
                                    search_pos = tag_start_pos + end_pos + 1;
                                    continue;
                                } else {
                                    search_pos = tag_start_pos + end_pos + 1;
                                    continue;
                                }
                            }
                            search_pos = pos + 1;
                        } else {
                            // Found a closing tag
                            depth -= 1;
                            if depth == 0 {
                                // Found the matching closing tag
                                inner_content = html_content[tag_end..pos].to_string();
                                break;
                            }
                            search_pos = pos + close_tag_pattern.len();
                        }
                    } else {
                        // No more tags found
                        break;
                    }
                }
                
                if inner_content.is_empty() {
                }
                
                self.parse_element(&tag_name, &attributes, &inner_content, super::inheritance::InheritableProperties::new())
            } else {
                Err("Could not find root element in HTML".to_string())
            }
        }
    }

    fn normalize_html(html: &str) -> String {
        // Remove DOCTYPE declarations
        let doctype_pattern = regex::Regex::new(r"<!DOCTYPE[^>]*>").unwrap();
        let html = doctype_pattern.replace_all(html, "");
        
        // Remove HTML comments (including multiline comments)
        // Use (?s) flag to make . match newlines, or use [\s\S] to match any character
        let comment_pattern = regex::Regex::new(r"(?s)<!--.*?-->").unwrap();
        let html = comment_pattern.replace_all(&html, "");
        
        // Remove html, head, and body tags (but keep their content)
        let html_tag_pattern = regex::Regex::new(r"<html[^>]*>|</html>").unwrap();
        let html = html_tag_pattern.replace_all(&html, "");
        
        let head_tag_pattern = regex::Regex::new(r"(?s)<head[^>]*>.*?</head>").unwrap();
        let html = head_tag_pattern.replace_all(&html, "");
        
        let body_tag_pattern = regex::Regex::new(r"<body[^>]*>|</body>").unwrap();
        let html = body_tag_pattern.replace_all(&html, "");
        
        // Normalize whitespace - but preserve structure for nested tags
        // Instead of joining everything, just collapse multiple whitespace
        let whitespace_pattern = regex::Regex::new(r"\s+").unwrap();
        whitespace_pattern.replace_all(&html, " ").to_string()
    }

    fn parse_attributes(attr_str: &str) -> HashMap<String, String> {
        let mut attrs = HashMap::new();
        // Use regular string instead of raw string to allow escaped quotes
        let attr_pattern = regex::Regex::new(r#"(\w+)="([^"]+)""#).unwrap();
        
        for cap in attr_pattern.captures_iter(attr_str) {
            attrs.insert(cap[1].to_string(), cap[2].to_string());
        }
        
        attrs
    }

    fn parse_element(
        &self,
        tag_name: &str,
        attributes: &HashMap<String, String>,
        content: &str,
        inherited_props: super::inheritance::InheritableProperties,
    ) -> Result<Component, String> {
        let mut component = match tag_name.to_lowercase().as_str() {
            "view" | "div" => Component::new(ComponentType::View(View::new())),
            "text" | "span" | "p" => {
                let text_content = Self::extract_text_content(content);
                Component::new(ComponentType::Text(Text::new(text_content)))
            }
            "viewport" => Component::new(ComponentType::Viewport(super::components::Viewport::new())),
            _ => return Err(format!("Unknown HTML tag: {}", tag_name)),
        };

        // Parse ID
        if let Some(id) = attributes.get("id") {
            component.id = Some(id.clone());
        }

        // Parse class
        let classes = attributes.get("class")
            .map(|c| c.split_whitespace().map(|s| s.to_string()).collect())
            .unwrap_or_else(Vec::new);

        // Track inheritable properties (start with parent's values)
        let mut inherited = inherited_props.inherit_from();
        
        // Apply CSS classes
        let mut style = Style::default();
        for class_name in &classes {
            if let Some(class_props) = self.css_classes.get(class_name) {
                style = Self::apply_css_properties(style, class_props)?;
                
                // Handle position: absolute from CSS classes
                if let Some(position) = class_props.get("position") {
                    if position == "absolute" {
                        component.absolute = true;
                    }
                }
                
                // Handle layout properties from CSS classes
                // Use align-row and align-col (old style names) instead of justify-content/align-items
                if let Some(align_row) = class_props.get("align-row") {
                    if let ComponentType::View(ref mut view) = component.component_type {
                        view.row_alignment = Self::parse_alignment(align_row)?;
                    }
                }
                if let Some(align_col) = class_props.get("align-col") {
                    if let ComponentType::View(ref mut view) = component.component_type {
                        view.column_alignment = Self::parse_alignment(align_col)?;
                    }
                }
                // Also support justify-content/align-items for compatibility
                if let Some(justify) = class_props.get("justify-content") {
                    if let ComponentType::View(ref mut view) = component.component_type {
                        view.row_alignment = Self::parse_alignment(justify)?;
                    }
                }
                if let Some(align) = class_props.get("align-items") {
                    if let ComponentType::View(ref mut view) = component.component_type {
                        view.column_alignment = Self::parse_alignment(align)?;
                    }
                }
                if let Some(flex_dir) = class_props.get("flex-direction") {
                    let flex_direction = match flex_dir.as_str() {
                        "row" => FlexDirection::Row,
                        "column" => FlexDirection::Column,
                        _ => FlexDirection::Row,
                    };
                    // For View components, set View's flex_direction (layout doesn't need it)
                    if let ComponentType::View(ref mut view) = component.component_type {
                        view.flex_direction = flex_direction;
                    }
                    // Non-View components don't use flex_direction, so we ignore it
                }
                if let Some(gap) = class_props.get("gap") {
                    let gap_value = Self::parse_size_value(gap)?.unwrap_or(0.0);
                    // For View components, set View's gap (layout uses it)
                    if let ComponentType::View(ref mut view) = component.component_type {
                        view.gap = gap_value;
                    }
                }
                
                // Handle text color from CSS classes
                // For Views, this sets the inherited text color for children
                // For Text/Button, this sets their own text color
                if let Some(color) = class_props.get("color") {
                    let parsed_color = Self::parse_color_value(color)?;
                    inherited.text_color = Some(parsed_color); // Set for inheritance
                    match &mut component.component_type {
                        ComponentType::Text(text) => {
                            text.color = parsed_color;
                        }
                        _ => {
                            // For View components, just set inherited for children
                        }
                    }
                }
                
                // Handle font-size from CSS classes
                if let Some(font_size) = class_props.get("font-size") {
                    if let Ok(size) = Self::parse_size_value(font_size) {
                        if let Some(size_val) = size {
                            inherited.font_size = Some(size_val); // Set for inheritance
                            match &mut component.component_type {
                                ComponentType::Text(text) => {
                                    text.font_size = size_val;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        // Apply inline style
        if let Some(inline_style) = attributes.get("style") {
            let inline_props = Self::parse_inline_style(inline_style);
            style = Self::apply_css_properties(style, &inline_props)?;
            
            // Handle display: none for visibility
            if let Some(display) = inline_props.get("display") {
                if display == "none" {
                    component.visible = false;
                }
            }
            
            // Handle position: absolute
            if let Some(position) = inline_props.get("position") {
                if position == "absolute" {
                    component.absolute = true;
                }
            }
            
            // Handle layout properties from inline style
            // Use align-row and align-col (old style names) instead of justify-content/align-items
            if let Some(align_row) = inline_props.get("align-row") {
                if let ComponentType::View(ref mut view) = component.component_type {
                    view.row_alignment = Self::parse_alignment(align_row)?;
                }
            }
            if let Some(align_col) = inline_props.get("align-col") {
                if let ComponentType::View(ref mut view) = component.component_type {
                    view.column_alignment = Self::parse_alignment(align_col)?;
                }
            }
            // Also support justify-content/align-items for compatibility
            if let Some(justify) = inline_props.get("justify-content") {
                if let ComponentType::View(ref mut view) = component.component_type {
                    view.row_alignment = Self::parse_alignment(justify)?;
                }
            }
            if let Some(align) = inline_props.get("align-items") {
                if let ComponentType::View(ref mut view) = component.component_type {
                    view.column_alignment = Self::parse_alignment(align)?;
                }
            }
        }

        component.style = style;

        // Parse layout attributes
        // Note: flex-direction is only used for View components and is stored in View, not Layout
        if let Some(flex_dir) = attributes.get("flex-direction") {
            let flex_direction = match flex_dir.as_str() {
                "row" => FlexDirection::Row,
                "column" => FlexDirection::Column,
                _ => FlexDirection::Row,
            };
            if let ComponentType::View(ref mut view) = component.component_type {
                view.flex_direction = flex_direction;
            }
        }

        if let Some(gap) = attributes.get("gap") {
            let gap_value = gap.parse::<f32>()
                .map_err(|_| format!("Invalid gap value: {}", gap))?;
            // For View components, set View's gap
            if let ComponentType::View(ref mut view) = component.component_type {
                view.gap = gap_value;
            }
        }


        // Parse component-specific attributes
        match &mut component.component_type {
            ComponentType::Text(text) => {
                // Apply inherited properties if not explicitly set
                if let Some(font_size) = attributes.get("font-size") {
                    let size_val = font_size.parse::<f32>()
                        .map_err(|_| format!("Invalid font-size: {}", font_size))?;
                    text.font_size = size_val;
                    inherited.font_size = Some(size_val);
                } else if let Some(parent_font_size) = inherited.font_size {
                    // Inherit from parent if not explicitly set
                    text.font_size = parent_font_size;
                }
                if let Some(color) = attributes.get("color") {
                    let parsed_color = Self::parse_color_value(color)?;
                    text.color = parsed_color;
                    inherited.text_color = Some(parsed_color);
                } else if let Some(parent_color) = inherited.text_color {
                    // Inherit from parent if not explicitly set
                    text.color = parent_color;
                }
            }
            ComponentType::View(view) => {
                if let Some(on_click) = attributes.get("on-click") {
                    view.on_click = Some(on_click.clone());
                }
            }
            _ => {}
        }

        // Parse children for View components
        if matches!(component.component_type, ComponentType::View(_)) {
            if let ComponentType::View(ref mut view) = component.component_type {
                view.children = self.parse_children(content, inherited)?;
                view.rebuild_layers();
            }
        }

        Ok(component)
    }

    fn parse_children(&self, content: &str, inherited_props: super::inheritance::InheritableProperties) -> Result<Vec<Component>, String> {
        let mut children = Vec::new();
        let content = content.trim();
        
        if content.is_empty() {
            return Ok(children);
        }

        // Find all top-level tags (not nested) and text nodes
        let mut pos = 0;
        while pos < content.len() {
            // Skip whitespace
            while pos < content.len() && content.chars().nth(pos).unwrap().is_whitespace() {
                pos += 1;
            }
            if pos >= content.len() {
                break;
            }

            // Check if we're at a tag or text node
            if content.chars().nth(pos) != Some('<') {
                // This is a text node - extract text until next tag
                let text_start = pos;
                let next_tag = content[pos..].find('<');
                let text_end = if let Some(tag_pos) = next_tag {
                    pos + tag_pos
                } else {
                    content.len()
                };
                
                let text_content = content[text_start..text_end].trim();
                if !text_content.is_empty() {
                    // Create a text component for this text node
                    let mut text_component = Component::new(ComponentType::Text(Text::new(text_content.to_string())));
                    // Inherit text color and font size from parent
                    if let ComponentType::Text(text) = &mut text_component.component_type {
                        if let Some(parent_color) = inherited_props.text_color {
                            text.color = parent_color;
                        }
                        if let Some(parent_font_size) = inherited_props.font_size {
                            text.font_size = parent_font_size;
                        }
                    }
                    children.push(text_component);
                }
                pos = text_end;
                continue;
            }

            let tag_start = pos;
            let tag_end = content[pos..].find('>')
                .map(|i| pos + i + 1)
                .ok_or_else(|| "Unclosed tag".to_string())?;

            let tag_str = &content[tag_start..tag_end];
            
            // Handle self-closing tags like <view /> or <view/>
            let is_self_closing = tag_str.ends_with("/>") || tag_str.ends_with(" />");
            let tag_pattern = if is_self_closing {
                regex::Regex::new(r"<(\w+)([^/>]*)/?>")
                    .map_err(|e| format!("Failed to create regex: {}", e))?
            } else {
                regex::Regex::new(r"<(\w+)([^>]*)>")
                    .map_err(|e| format!("Failed to create regex: {}", e))?
            };
            
            let tag_cap = tag_pattern.captures(tag_str)
                .ok_or_else(|| format!("Invalid tag format: {}", tag_str))?;
            
            let tag_name = &tag_cap[1];
            let attributes = Self::parse_attributes(&tag_cap[2]);
            
            // Handle self-closing tags
            if is_self_closing {
                let child = self.parse_element(tag_name, &attributes, "", inherited_props.clone())?;
                children.push(child);
                pos = tag_end;
                continue;
            }
            
            // Find matching closing tag
            let mut depth = 1;
            let mut search_pos = tag_end;
            let mut found_end = false;
            
            while search_pos < content.len() && depth > 0 {
                // Look for both opening and closing tags
                let open_pattern = format!("<{}", tag_name);
                let close_pattern = format!("</{}>", tag_name);
                
                let next_open = content[search_pos..].find(&open_pattern);
                let next_close = content[search_pos..].find(&close_pattern);
                
                let next_pos = match (next_open, next_close) {
                    (Some(open_pos), Some(close_pos)) => {
                        if open_pos < close_pos {
                            Some((open_pos + search_pos, true))
                        } else {
                            Some((close_pos + search_pos, false))
                        }
                    }
                    (Some(open_pos), None) => Some((open_pos + search_pos, true)),
                    (None, Some(close_pos)) => Some((close_pos + search_pos, false)),
                    (None, None) => None,
                };
                
                if let Some((found_pos, is_open)) = next_pos {
                    if is_open {
                        // Check if this is actually an opening tag (not part of a closing tag)
                        let tag_start_pos = found_pos;
                        let tag_end_pos = content[tag_start_pos..].find('>');
                        if let Some(end_pos) = tag_end_pos {
                            let tag_str = &content[tag_start_pos..tag_start_pos + end_pos + 1];
                            // Make sure it's not a self-closing tag or closing tag
                            if !tag_str.contains('/') && tag_str.trim().starts_with(&open_pattern) {
                                depth += 1;
                                search_pos = tag_start_pos + end_pos + 1;
                                continue;
                            }
                        }
                        search_pos = found_pos + 1;
                    } else {
                        // Found a closing tag
                        depth -= 1;
                        if depth == 0 {
                            let close_pos = found_pos;
                            let inner_content = &content[tag_end..close_pos];
                            let child = self.parse_element(tag_name, &attributes, inner_content, inherited_props.clone())?;
                            children.push(child);
                            // Advance past the closing tag: </tag_name>
                            pos = close_pos + close_pattern.len();
                            found_end = true;
                            break;
                        }
                        search_pos = found_pos + close_pattern.len();
                    }
                } else {
                    // No more tags found
                    break;
                }
            }
            
            if !found_end {
                // Self-closing or text-only element
                let inner_content = "";
                let child = self.parse_element(tag_name, &attributes, inner_content, inherited_props.clone())?;
                children.push(child);
                pos = tag_end;
            }
        }

        Ok(children)
    }

    fn extract_text_content(html: &str) -> String {
        // Remove all HTML tags
        let tag_pattern = regex::Regex::new(r"<[^>]+>").unwrap();
        tag_pattern.replace_all(html, "").trim().to_string()
    }

    fn parse_inline_style(style_str: &str) -> HashMap<String, String> {
        let mut props = HashMap::new();
        for prop in style_str.split(';') {
            let prop = prop.trim();
            if prop.is_empty() {
                continue;
            }
            if let Some(colon_pos) = prop.find(':') {
                let key = prop[..colon_pos].trim().to_string();
                let value = prop[colon_pos + 1..].trim().to_string();
                props.insert(key, value);
            }
        }
        props
    }
    
    fn apply_css_properties(
        mut style: Style,
        properties: &HashMap<String, String>,
    ) -> Result<Style, String> {
        for (key, value) in properties {
            match key.as_str() {
                "background-color" | "background" => {
                    style.background_color = Self::parse_color_value(value)?;
                }
                "border" => {
                    style.border = Self::parse_border_value(value)?;
                }
                "border-width" => {
                    style.border.width = Self::parse_size_value(value)?.unwrap_or(0.0);
                }
                "border-color" => {
                    style.border.color = Self::parse_color_value(value)?;
                }
                "border-radius" => {
                    style.border.radius = Self::parse_size_value(value)?.unwrap_or(0.0);
                }
                "shadow" | "box-shadow" => {
                    style.shadow = Self::parse_shadow_value(value)?;
                }
                "padding" => {
                    style.padding = Self::parse_padding_value(value)?;
                }
                "margin" => {
                    style.margin = Self::parse_margin_value(value)?;
                }
                "width" => {
                    style.width = Self::parse_size_css(value)?;
                }
                "height" => {
                    style.height = Self::parse_size_css(value)?;
                }
                "z-index" => {
                    if let Ok(z_index) = value.trim().parse::<i32>() {
                        style.z_index = z_index;
                    }
                }
                _ => {
                    // Ignore unknown properties
                }
            }
        }
        Ok(style)
    }

    fn parse_color_value(value: &str) -> Result<Color, String> {
        let value = value.trim();
        if value.starts_with('#') {
            Color::from_hex(value)
        } else if value.starts_with("rgba(") && value.ends_with(')') {
            // Parse rgba(r, g, b, a) format
            let content = value.strip_prefix("rgba(").unwrap().strip_suffix(')').unwrap();
            let parts: Vec<&str> = content.split(',').map(|s| s.trim()).collect();
            if parts.len() == 4 {
                let r = parts[0].parse::<f32>()
                    .map_err(|_| format!("Invalid rgba red value: {}", parts[0]))? / 255.0;
                let g = parts[1].parse::<f32>()
                    .map_err(|_| format!("Invalid rgba green value: {}", parts[1]))? / 255.0;
                let b = parts[2].parse::<f32>()
                    .map_err(|_| format!("Invalid rgba blue value: {}", parts[2]))? / 255.0;
                let a = parts[3].parse::<f32>()
                    .map_err(|_| format!("Invalid rgba alpha value: {}", parts[3]))? / 255.0;
                Ok(Color::new(r, g, b, a))
            } else {
                Err(format!("Invalid rgba format: {}", value))
            }
        } else if value.starts_with("rgb(") && value.ends_with(')') {
            // Parse rgb(r, g, b) format
            let content = value.strip_prefix("rgb(").unwrap().strip_suffix(')').unwrap();
            let parts: Vec<&str> = content.split(',').map(|s| s.trim()).collect();
            if parts.len() == 3 {
                let r = parts[0].parse::<f32>()
                    .map_err(|_| format!("Invalid rgb red value: {}", parts[0]))? / 255.0;
                let g = parts[1].parse::<f32>()
                    .map_err(|_| format!("Invalid rgb green value: {}", parts[1]))? / 255.0;
                let b = parts[2].parse::<f32>()
                    .map_err(|_| format!("Invalid rgb blue value: {}", parts[2]))? / 255.0;
                Ok(Color::rgb(r, g, b))
            } else {
                Err(format!("Invalid rgb format: {}", value))
            }
        } else {
            match value {
                "transparent" => Ok(Color::transparent()),
                "white" => Ok(Color::white()),
                "black" => Ok(Color::black()),
                "red" => Ok(Color::rgb(1.0, 0.0, 0.0)),
                "green" => Ok(Color::rgb(0.0, 1.0, 0.0)),
                "blue" => Ok(Color::rgb(0.0, 0.0, 1.0)),
                _ => Err(format!("Unknown color: {}", value)),
            }
        }
    }

    fn parse_size_value(value: &str) -> Result<Option<f32>, String> {
        let value = value.trim();
        if value.ends_with("px") {
            let num = value.trim_end_matches("px").parse::<f32>()
                .map_err(|_| format!("Invalid size: {}", value))?;
            Ok(Some(num))
        } else if let Ok(num) = value.parse::<f32>() {
            Ok(Some(num))
        } else {
            Ok(None)
        }
    }

    fn parse_size_css(value: &str) -> Result<Size, String> {
        let value = value.trim();
        if value == "auto" {
            Ok(Size::Auto)
        } else if value.ends_with("px") {
            let num = value.trim_end_matches("px").parse::<f32>()
                .map_err(|_| format!("Invalid size: {}", value))?;
            Ok(Size::Pixels(num))
        } else if value.ends_with('%') {
            let num = value.trim_end_matches('%').parse::<f32>()
                .map_err(|_| format!("Invalid size: {}", value))?;
            Ok(Size::Percent(num))
        } else if let Ok(num) = value.parse::<f32>() {
            Ok(Size::Pixels(num))
        } else {
            Err(format!("Invalid size: {}", value))
        }
    }

    fn parse_padding_value(value: &str) -> Result<Padding, String> {
        let value = value.trim();
        
        // Helper to parse a size value with units (px, etc.)
        let parse_size_with_unit = |s: &str| -> Result<f32, String> {
            let s = s.trim();
            if s.ends_with("px") {
                s.trim_end_matches("px").parse::<f32>()
                    .map_err(|_| format!("Invalid padding value: {}", s))
            } else if let Ok(num) = s.parse::<f32>() {
                Ok(num)
            } else {
                Err(format!("Invalid padding value: {}", s))
            }
        };
        
        if value.contains(' ') {
            let parts: Vec<&str> = value.split_whitespace().collect();
            if parts.len() == 4 {
                Ok(Padding::new(
                    parse_size_with_unit(parts[0])?,
                    parse_size_with_unit(parts[1])?,
                    parse_size_with_unit(parts[2])?,
                    parse_size_with_unit(parts[3])?,
                ))
            } else if parts.len() == 2 {
                let v = parse_size_with_unit(parts[0])?;
                let h = parse_size_with_unit(parts[1])?;
                Ok(Padding::new(v, h, v, h))
            } else {
                // Single value - uniform padding
                let num = parse_size_with_unit(value)?;
                Ok(Padding::uniform(num))
            }
        } else {
            // Single value - uniform padding
            let num = parse_size_with_unit(value)?;
            Ok(Padding::uniform(num))
        }
    }

    fn parse_margin_value(value: &str) -> Result<Margin, String> {
        let value = value.trim();
        
        // Helper to parse a size value with units (px, etc.)
        let parse_size_with_unit = |s: &str| -> Result<f32, String> {
            let s = s.trim();
            if s.ends_with("px") {
                s.trim_end_matches("px").parse::<f32>()
                    .map_err(|_| format!("Invalid margin value: {}", s))
            } else if let Ok(num) = s.parse::<f32>() {
                Ok(num)
            } else {
                Err(format!("Invalid margin value: {}", s))
            }
        };
        
        if value.contains(' ') {
            let parts: Vec<&str> = value.split_whitespace().collect();
            if parts.len() == 4 {
                Ok(Margin::new(
                    parse_size_with_unit(parts[0])?,
                    parse_size_with_unit(parts[1])?,
                    parse_size_with_unit(parts[2])?,
                    parse_size_with_unit(parts[3])?,
                ))
            } else if parts.len() == 2 {
                let v = parse_size_with_unit(parts[0])?;
                let h = parse_size_with_unit(parts[1])?;
                Ok(Margin::new(v, h, v, h))
            } else {
                // Single value - uniform margin
                let num = parse_size_with_unit(value)?;
                Ok(Margin::uniform(num))
            }
        } else {
            // Single value - uniform margin
            let num = parse_size_with_unit(value)?;
            Ok(Margin::uniform(num))
        }
    }

    fn parse_border_value(value: &str) -> Result<Border, String> {
        // Simple border parser: "2px solid #000000" or just "2px #000000"
        let parts: Vec<&str> = value.split_whitespace().collect();
        let mut width = 0.0;
        let mut color = Color::black();
        let radius = 0.0;

        for part in parts {
            if part.ends_with("px") {
                width = part.trim_end_matches("px").parse::<f32>()
                    .unwrap_or(0.0);
            } else if part.starts_with('#') {
                color = Self::parse_color_value(part)?;
            } else if part == "solid" || part == "dashed" || part == "dotted" {
                // Border style - ignore for now
            }
        }

        Ok(Border::new(width, color, radius))
    }

    fn parse_shadow_value(value: &str) -> Result<Shadow, String> {
        // Simple shadow parser: "0px 2px 4px #00000080" or "2px rgba(0, 0, 0, 50) 0px 3px"
        // For rgba values, we need to handle the case where the color spans multiple "words"
        let value = value.trim();
        
        // Find rgba(...) or rgb(...) patterns first
        let mut rgba_start = None;
        let mut rgba_end = None;
        if let Some(start) = value.find("rgba(") {
            rgba_start = Some(start);
            // Find matching closing paren
            let mut paren_count = 0;
            let mut found_start = false;
            for (i, ch) in value[start..].char_indices() {
                if ch == '(' {
                    if found_start {
                        paren_count += 1;
                    } else {
                        found_start = true;
                        paren_count = 1;
                    }
                } else if ch == ')' {
                    paren_count -= 1;
                    if paren_count == 0 {
                        rgba_end = Some(start + i + 1);
                        break;
                    }
                }
            }
        } else if let Some(start) = value.find("rgb(") {
            rgba_start = Some(start);
            // Find matching closing paren
            let mut paren_count = 0;
            let mut found_start = false;
            for (i, ch) in value[start..].char_indices() {
                if ch == '(' {
                    if found_start {
                        paren_count += 1;
                    } else {
                        found_start = true;
                        paren_count = 1;
                    }
                } else if ch == ')' {
                    paren_count -= 1;
                    if paren_count == 0 {
                        rgba_end = Some(start + i + 1);
                        break;
                    }
                }
            }
        }
        
        // Split by whitespace, but preserve rgba/rgb color values
        let mut parts = Vec::new();
        let mut current = String::new();
        let mut i = 0;
        while i < value.len() {
            if let (Some(start), Some(end)) = (rgba_start, rgba_end) {
                if i == start {
                    // Add the rgba/rgb value as a single part
                    parts.push(value[start..end].to_string());
                    i = end;
                    continue;
                }
            }
            
            let ch = value.chars().nth(i).unwrap();
            if ch.is_whitespace() {
                if !current.is_empty() {
                    parts.push(current.clone());
                    current.clear();
                }
            } else {
                current.push(ch);
            }
            i += 1;
        }
        if !current.is_empty() {
            parts.push(current);
        }
        
        let offset_x = parts.get(0)
            .and_then(|p| Self::parse_size_value(p).ok()?)
            .unwrap_or(0.0);
        let offset_y = parts.get(1)
            .and_then(|p| Self::parse_size_value(p).ok()?)
            .unwrap_or(0.0);
        let blur = parts.get(2)
            .and_then(|p| Self::parse_size_value(p).ok()?)
            .unwrap_or(0.0);
        
        // Find color value
        let color = parts.iter()
            .find(|p| p.starts_with('#') || p.starts_with("rgba(") || p.starts_with("rgb("))
            .map(|p| Self::parse_color_value(p))
            .transpose()?
            .unwrap_or(Color::new(0.0, 0.0, 0.0, 0.5));
        
        // Spread is usually after color if present
        let spread = parts.iter()
            .position(|p| p.starts_with('#') || p.starts_with("rgba(") || p.starts_with("rgb("))
            .and_then(|idx| parts.get(idx + 1))
            .and_then(|p| Self::parse_size_value(p).ok()?)
            .unwrap_or(parts.get(4)
                .and_then(|p| Self::parse_size_value(p).ok()?)
                .unwrap_or(0.0));

        Ok(Shadow::new(offset_x, offset_y, blur, color, spread))
    }

    fn parse_alignment(s: &str) -> Result<Alignment, String> {
        match s {
            "left" | "top" | "start" | "flex-start" => Ok(Alignment::Start),
            "center" => Ok(Alignment::Center),
            "right" | "bottom" | "end" | "flex-end" => Ok(Alignment::End),
            "stretch" => Ok(Alignment::Stretch),
            "space-between" => Ok(Alignment::SpaceBetween),
            "space-around" => Ok(Alignment::SpaceAround),
            _ => Err(format!("Invalid alignment: {}", s)),
        }
    }
}

impl Default for UiParser {
    fn default() -> Self {
        Self::new()
    }
}