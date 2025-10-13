#ifndef WINDOW_HELPER_HPP
#define WINDOW_HELPER_HPP

#include <SFML/Graphics.hpp>

// Helper function to get the actual window size on macOS
// This works around SFML not detecting Rectangle window resizes
sf::Vector2u getActualWindowSize(sf::RenderWindow& window);

#endif

