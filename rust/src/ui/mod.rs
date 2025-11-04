// UI system for declarative UI definitions with HTML/CSS parsing
// Similar structure to C++ UI system with Screen and UIManager

pub mod component;
pub mod styles;
pub mod layout;
pub mod parser;
pub mod renderer;
pub mod screen;
pub mod manager;

pub use component::{Component, ComponentType, View};
pub use styles::{Color, Size};
pub use parser::UiParser;
pub use renderer::UiRenderer;
pub use screen::Screen;
pub use manager::UIManager;
