// Math utilities module

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len > 0.0 {
            Self {
                x: self.x / len,
                y: self.y / len,
            }
        } else {
            *self
        }
    }

    pub fn distance(&self, other: &Self) -> f32 {
        (*self - *other).length()
    }
}

impl std::ops::Add for Vec2 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl std::ops::Mul<f32> for Vec2 {
    type Output = Self;

    fn mul(self, scalar: f32) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    pub fn new(left: f32, top: f32, width: f32, height: f32) -> Self {
        Self {
            left,
            top,
            width,
            height,
        }
    }

    pub fn from_center(center: Vec2, size: Vec2) -> Self {
        Self {
            left: center.x - size.x / 2.0,
            top: center.y - size.y / 2.0,
            width: size.x,
            height: size.y,
        }
    }

    pub fn center(&self) -> Vec2 {
        Vec2::new(
            self.left + self.width / 2.0,
            self.top + self.height / 2.0,
        )
    }

    pub fn size(&self) -> Vec2 {
        Vec2::new(self.width, self.height)
    }

    pub fn right(&self) -> f32 {
        self.left + self.width
    }

    pub fn bottom(&self) -> f32 {
        self.top + self.height
    }

    pub fn contains(&self, point: Vec2) -> bool {
        point.x >= self.left
            && point.x <= self.right()
            && point.y >= self.top
            && point.y <= self.bottom()
    }

    pub fn expand(&self, amount: f32) -> Self {
        Self {
            left: self.left - amount,
            top: self.top - amount,
            width: self.width + amount * 2.0,
            height: self.height + amount * 2.0,
        }
    }
}

impl std::ops::Add<Rect> for Rect {
    type Output = Self;

    fn add(self, delta: Self) -> Self {
        Self {
            left: self.left + delta.left,
            top: self.top + delta.top,
            width: self.width + delta.width,
            height: self.height + delta.height,
        }
    }
}

