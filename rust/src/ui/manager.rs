// UIManager - similar to C++ UIManager
// Manages multiple screens and handles window events

use super::screen::Screen;
use std::collections::HashMap;

pub struct UIManager {
    screens: HashMap<String, Screen>,
    current_screen: Option<String>,
    window_width: f32,
    window_height: f32,
}

impl UIManager {
    pub fn new(window_width: f32, window_height: f32) -> Self {
        Self {
            screens: HashMap::new(),
            current_screen: None,
            window_width,
            window_height,
        }
    }
    
    pub fn add_screen(&mut self, name: String, mut screen: Screen) {
        screen.resize(self.window_width, self.window_height);
        self.screens.insert(name.clone(), screen);
        
        // Set as current screen if none is set
        if self.current_screen.is_none() {
            self.current_screen = Some(name);
        }
    }
    
    pub fn draw(&self, _renderer: &mut super::renderer::UiRenderer) {
        if let Some(ref _screen_name) = self.current_screen {
            if let Some(_screen) = self.screens.get(_screen_name) {
                // Draw screen elements
                // This will be handled by the renderer
            }
        }
    }
    
    pub fn handle_event(&mut self, event: &winit::event::WindowEvent) -> bool {
        // Handle window resize
        if let winit::event::WindowEvent::Resized(physical_size) = event {
            self.window_width = physical_size.width as f32;
            self.window_height = physical_size.height as f32;
            
            if let Some(ref screen_name) = self.current_screen {
                if let Some(screen) = self.screens.get_mut(screen_name) {
                    screen.resize(self.window_width, self.window_height);
                }
            }
        }
        
        // Forward event to current screen
        if let Some(ref screen_name) = self.current_screen {
            if let Some(screen) = self.screens.get_mut(screen_name) {
                return screen.handle_event(event);
            }
        }
        
        false
    }
    
    pub fn update(&mut self, dt: f32, mouse_pos: (f32, f32)) -> bool {
        if let Some(ref screen_name) = self.current_screen {
            if let Some(screen) = self.screens.get_mut(screen_name) {
                return screen.update(dt, mouse_pos);
            }
        }
        false
    }
    
    pub fn set_current_screen(&mut self, screen_name: String) {
        if self.screens.contains_key(&screen_name) {
            self.current_screen = Some(screen_name.clone());
            
            // Resize the screen
            if let Some(screen) = self.screens.get_mut(&screen_name) {
                screen.resize(self.window_width, self.window_height);
            }
        }
    }
    
    pub fn get_screen(&mut self, name: &str) -> Option<&mut Screen> {
        self.screens.get_mut(name)
    }
    
    pub fn resize(&mut self, width: f32, height: f32) {
        self.window_width = width;
        self.window_height = height;
        
        if let Some(ref screen_name) = self.current_screen {
            if let Some(screen) = self.screens.get_mut(screen_name) {
                screen.resize(width, height);
            }
        }
    }
}

