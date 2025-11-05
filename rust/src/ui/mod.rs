// UI system for declarative UI definitions with HTML/CSS parsing
// Similar structure to C++ UI system with Screen and UIManager

pub mod styles;
pub mod layout;
pub mod parser;
pub mod renderer;
pub mod screen;
pub mod manager;
pub mod components;

pub use components::{Component, ComponentType, View, Text};
pub use styles::{Color, Size};
pub use parser::UiParser;
pub use renderer::UiRenderer;
pub use screen::Screen;
pub use manager::UIManager;
