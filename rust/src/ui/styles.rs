// Style system for UI components - supports colors, borders, shadows, spacing, etc.

#[derive(Debug, Clone, Copy)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    pub fn transparent() -> Self {
        Self { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }
    }

    pub fn white() -> Self {
        Self { r: 1.0, g: 1.0, b: 1.0, a: 1.0 }
    }

    pub fn black() -> Self {
        Self { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }
    }

    pub fn from_hex(hex: &str) -> Result<Self, String> {
        let hex = hex.trim_start_matches('#');
        if hex.len() != 6 && hex.len() != 8 {
            return Err(format!("Invalid hex color: {}", hex));
        }
        
        let r = u8::from_str_radix(&hex[0..2], 16)
            .map_err(|_| "Invalid hex color")? as f32 / 255.0;
        let g = u8::from_str_radix(&hex[2..4], 16)
            .map_err(|_| "Invalid hex color")? as f32 / 255.0;
        let b = u8::from_str_radix(&hex[4..6], 16)
            .map_err(|_| "Invalid hex color")? as f32 / 255.0;
        let a = if hex.len() == 8 {
            u8::from_str_radix(&hex[6..8], 16)
                .map_err(|_| "Invalid hex color")? as f32 / 255.0
        } else {
            1.0
        };
        
        Ok(Self { r, g, b, a })
    }

    pub fn to_array(&self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Border {
    pub width: f32,
    pub color: Color,
    pub radius: f32, // Legacy: uniform radius for all corners
    pub radius_tl: f32, // Top-left
    pub radius_tr: f32, // Top-right
    pub radius_br: f32, // Bottom-right
    pub radius_bl: f32, // Bottom-left
}

impl Border {
    pub fn new(width: f32, color: Color, radius: f32) -> Self {
        Self { 
            width, 
            color, 
            radius,
            radius_tl: radius,
            radius_tr: radius,
            radius_br: radius,
            radius_bl: radius,
        }
    }
    
    pub fn with_corner_radii(width: f32, color: Color, tl: f32, tr: f32, br: f32, bl: f32) -> Self {
        Self {
            width,
            color,
            radius: tl.max(tr).max(br).max(bl), // Use max for legacy compatibility
            radius_tl: tl,
            radius_tr: tr,
            radius_br: br,
            radius_bl: bl,
        }
    }

    pub fn none() -> Self {
        Self::new(0.0, Color::transparent(), 0.0)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Shadow {
    pub offset_x: f32,
    pub offset_y: f32,
    pub blur: f32,
    pub color: Color,
    pub spread: f32,
}

impl Shadow {
    pub fn new(offset_x: f32, offset_y: f32, blur: f32, color: Color, spread: f32) -> Self {
        Self { offset_x, offset_y, blur, color, spread }
    }

    pub fn none() -> Self {
        Self {
            offset_x: 0.0,
            offset_y: 0.0,
            blur: 0.0,
            color: Color::transparent(),
            spread: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Padding {
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
    pub left: f32,
}

impl Padding {
    pub fn new(top: f32, right: f32, bottom: f32, left: f32) -> Self {
        Self { top, right, bottom, left }
    }

    pub fn uniform(value: f32) -> Self {
        Self { top: value, right: value, bottom: value, left: value }
    }

    pub fn zero() -> Self {
        Self { top: 0.0, right: 0.0, bottom: 0.0, left: 0.0 }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Margin {
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
    pub left: f32,
}

impl Margin {
    pub fn new(top: f32, right: f32, bottom: f32, left: f32) -> Self {
        Self { top, right, bottom, left }
    }

    pub fn uniform(value: f32) -> Self {
        Self { top: value, right: value, bottom: value, left: value }
    }

    pub fn zero() -> Self {
        Self { top: 0.0, right: 0.0, bottom: 0.0, left: 0.0 }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Size {
    Pixels(f32),
    Percent(f32),
    Flex(f32),
    Auto,
}

impl Size {
    pub fn pixels(value: f32) -> Self {
        Self::Pixels(value)
    }

    pub fn percent(value: f32) -> Self {
        Self::Percent(value)
    }

    pub fn flex(value: f32) -> Self {
        Self::Flex(value)
    }

    pub fn auto() -> Self {
        Self::Auto
    }
}

#[derive(Debug, Clone)]
pub struct Style {
    pub background_color: Color,
    pub text_color: Option<Color>,
    pub border: Border,
    pub shadow: Shadow,
    pub padding: Padding,
    pub margin: Margin,
    pub width: Size,
    pub height: Size,
    pub z_index: i32,
    pub cursor: Cursor,
}

impl Style {
    pub fn new() -> Self {
        Self {
            background_color: Color::transparent(),
            text_color: None,
            border: Border::none(),
            shadow: Shadow::none(),
            padding: Padding::zero(),
            margin: Margin::zero(),
            width: Size::Auto,
            height: Size::Auto,
            z_index: 0,
            cursor: Cursor::Default,
        }
    }

    pub fn default() -> Self {
        Self::new()
    }
}

impl Default for Style {
    fn default() -> Self {
        Self::new()
    }
}


#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Cursor {
    Default,
    Pointer,
}

#[derive(Debug, Clone, Copy)]
pub enum TextAlign {
    Left,
    Center,
    Right,
}