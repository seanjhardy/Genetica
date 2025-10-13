#include "simulator/windowHelper.hpp"
#include <iostream>

#ifdef __APPLE__
#import <Cocoa/Cocoa.h>
#import <SFML/Window.hpp>

sf::Vector2u getActualWindowSize(sf::RenderWindow& window) {
    // Get the native window handle from SFML
    NSWindow* nsWindow = (__bridge NSWindow*)window.getSystemHandle();
    
    if (nsWindow) {
        // Get the content view size in points (not backing/pixel dimensions)
        // SFML works in points, not pixels, even on Retina displays
        NSRect contentRect = [[nsWindow contentView] frame];
        
        return sf::Vector2u(static_cast<unsigned int>(contentRect.size.width),
                           static_cast<unsigned int>(contentRect.size.height));
    }
    
    // Fallback to SFML's reported size
    return window.getSize();
}

#else
// Non-macOS implementation
sf::Vector2u getActualWindowSize(sf::RenderWindow& window) {
    return window.getSize();
}
#endif

