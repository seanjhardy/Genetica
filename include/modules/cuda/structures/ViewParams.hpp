#include <SFML/Graphics.hpp>

class ViewParams {
    sf::FloatRect screenBounds;
    int zoomLevel;

    ViewParams(sf::FloatRect screenBounds, int zoomLevel) : screenBounds(screenBounds), zoomLevel(zoomLevel) {}
};