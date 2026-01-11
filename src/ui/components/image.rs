use std::path::{Path, PathBuf};

use super::super::styles::{Color, Padding, Size};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageResizeMode {
    Contain,
    Cover,
    Stretch,
}

#[derive(Debug, Clone)]
pub struct Image {
    pub source: Option<String>,
    pub resolved_path: Option<PathBuf>,
    pub resize_mode: ImageResizeMode,
    pub tint: Color,
    pub natural_width: f32,
    pub natural_height: f32,
    
    // Hover state
    pub hover_source: Option<String>,
    pub hover_tint: Option<Color>,
    pub base_source: Option<String>,
    pub base_tint: Color,
    
    // Group-hover state (applied when parent group is hovered)
    pub group_hover_source: Option<String>,
    pub group_hover_tint: Option<Color>,
}

impl Image {
    pub fn new() -> Self {
        Self {
            source: None,
            resolved_path: None,
            resize_mode: ImageResizeMode::Contain,
            tint: Color::white(),
            natural_width: 0.0,
            natural_height: 0.0,
            hover_source: None,
            hover_tint: None,
            base_source: None,
            base_tint: Color::white(),
            group_hover_source: None,
            group_hover_tint: None,
        }
    }

    pub fn set_source(&mut self, value: &str) {
        self.source = Some(value.to_string());
        // Initialize base_source if not set (first time)
        if self.base_source.is_none() {
            self.base_source = Some(value.to_string());
        }
        if let Some(path) = resolve_image_path(value) {
            self.update_dimensions_from_path(&path);
            self.resolved_path = Some(path);
        } else {
            self.resolved_path = None;
            self.natural_width = 0.0;
            self.natural_height = 0.0;
        }
    }

    pub fn set_resize_mode(&mut self, value: &str) {
        let mode = match value.to_lowercase().as_str() {
            "cover" => ImageResizeMode::Cover,
            "stretch" | "fill" => ImageResizeMode::Stretch,
            _ => ImageResizeMode::Contain,
        };
        self.resize_mode = mode;
    }

    pub fn set_tint(&mut self, color: Color) {
        self.tint = color;
    }

    pub fn set_hover_source(&mut self, value: &str) {
        self.hover_source = Some(value.to_string());
    }

    pub fn set_hover_tint(&mut self, value: Color) {
        self.hover_tint = Some(value);
    }

    pub fn apply_hover_state(&mut self) {
        // Save base state if not already saved
        if self.base_source.is_none() {
            self.base_source = self.source.clone();
            self.base_tint = self.tint;
        }
        
        // Apply hover state
        if let Some(hover_src) = self.hover_source.clone() {
            self.set_source(&hover_src);
        }
        if let Some(hover_tint) = self.hover_tint {
            self.tint = hover_tint;
        }
    }

    pub fn restore_base_state(&mut self) {
        // Restore base state
        if let Some(base_src) = self.base_source.clone() {
            self.set_source(&base_src);
        }
        self.tint = self.base_tint;
    }
    
    pub fn set_group_hover_source(&mut self, value: &str) {
        self.group_hover_source = Some(value.to_string());
    }
    
    pub fn set_group_hover_tint(&mut self, value: Color) {
        self.group_hover_tint = Some(value);
    }
    
    pub fn apply_group_hover_state(&mut self) {
        // Save base state if not already saved
        if self.base_source.is_none() {
            self.base_source = self.source.clone();
            self.base_tint = self.tint;
        }
        
        // Apply group-hover state
        if let Some(group_hover_src) = self.group_hover_source.clone() {
            self.set_source(&group_hover_src);
        }
        if let Some(group_hover_tint) = self.group_hover_tint {
            self.tint = group_hover_tint;
        }
    }
    
    pub fn restore_from_group_hover_state(&mut self) {
        // Restore base state (only if not directly hovered)
        // This will be called when group-hover ends
        if let Some(base_src) = self.base_source.clone() {
            self.set_source(&base_src);
        }
        self.tint = self.base_tint;
    }

    pub fn calculate_width(&self, _padding: Padding) -> Size {
        // Images should use explicit sizing from CSS if available
        // Only use natural dimensions as a fallback
        // Return Auto so the parent will use explicit style.width if set
        Size::Auto
    }

    pub fn calculate_height(&self, _padding: Padding) -> Size {
        // Images should use explicit sizing from CSS if available
        // Only use natural dimensions as a fallback
        // Return Auto so the parent will use explicit style.height if set
        Size::Auto
    }

    fn update_dimensions_from_path(&mut self, path: &Path) {
        if let Ok((width, height)) = image::image_dimensions(path) {
            self.natural_width = width as f32;
            self.natural_height = height as f32;
        }
    }
}

pub fn resolve_image_path_public(source: &str) -> Option<PathBuf> {
    resolve_image_path(source)
}

/// Convert camelCase to snake_case (e.g., "pauseHighlighted" -> "pause_highlighted")
fn camel_to_snake_case(s: &str) -> String {
    let mut result = String::new();
    let mut chars = s.chars().peekable();
    
    while let Some(c) = chars.next() {
        if c.is_uppercase() {
            if !result.is_empty() {
                result.push('_');
            }
            result.push(c.to_lowercase().next().unwrap());
        } else {
            result.push(c);
        }
    }
    
    result
}

/// Convert kebab-case to camelCase (e.g., "slow-down" -> "slowDown")
fn kebab_to_camel_case(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = false;
    
    for c in s.chars() {
        if c == '-' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(c.to_uppercase().next().unwrap());
            capitalize_next = false;
        } else {
            result.push(c);
        }
    }
    
    result
}

fn resolve_image_path(source: &str) -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project_root = manifest_dir
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| manifest_dir.clone());

    let source_path = Path::new(source);
    let mut candidates = Vec::new();

    if source_path.is_absolute() {
        candidates.push(source_path.to_path_buf());
    } else {
        candidates.push(manifest_dir.join(source_path));
        candidates.push(project_root.join(source_path));
    }

    let has_extension = source_path.extension().is_some();
    let mut name_variants: Vec<PathBuf> = Vec::new();
    if has_extension {
        name_variants.push(source_path.to_path_buf());
    } else {
        // Convert camelCase to snake_case (e.g., "pauseHighlighted" -> "pause_highlighted")
        let snake_case_name = camel_to_snake_case(source);
        // Convert kebab-case to camelCase (e.g., "slow-down" -> "slowDown")
        let camel_case_name = if source.contains('-') {
            kebab_to_camel_case(source)
        } else {
            source.to_string()
        };
        
        name_variants.push(PathBuf::from(source));
        name_variants.push(PathBuf::from(format!("{source}.png")));
        name_variants.push(PathBuf::from(format!("{source}.jpg")));
        name_variants.push(PathBuf::from(format!("{source}.jpeg")));
        
        // Also try snake_case versions
        if snake_case_name != source {
            name_variants.push(PathBuf::from(&snake_case_name));
            name_variants.push(PathBuf::from(format!("{snake_case_name}.png")));
            name_variants.push(PathBuf::from(format!("{snake_case_name}.jpg")));
            name_variants.push(PathBuf::from(format!("{snake_case_name}.jpeg")));
        }
        
        // Also try camelCase versions (for kebab-case inputs)
        if camel_case_name != source {
            name_variants.push(PathBuf::from(&camel_case_name));
            name_variants.push(PathBuf::from(format!("{camel_case_name}.png")));
            name_variants.push(PathBuf::from(format!("{camel_case_name}.jpg")));
            name_variants.push(PathBuf::from(format!("{camel_case_name}.jpeg")));
        }
    }

    let asset_dirs = [
        manifest_dir.join("assets/icons"),
        manifest_dir.join("assets/textures"),
        manifest_dir.join("assets/ui"),
        project_root.join("assets/icons"),
        project_root.join("assets/textures"),
        project_root.join("assets/ui"),
    ];

    for name in &name_variants {
        for dir in &asset_dirs {
            candidates.push(dir.join(name));
        }
    }

    for candidate in candidates {
        if candidate.exists() {
            return Some(candidate);
        }
    }

    None
}

