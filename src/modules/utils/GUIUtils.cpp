#include <SFML/Graphics.hpp>
#include <modules/utils/GUIUtils.hpp>

sf::Color brightness(sf::Color color, float brightness) {
    return {
      static_cast<sf::Uint8>(color.r * brightness),
      static_cast<sf::Uint8>(color.g * brightness),
      static_cast<sf::Uint8>(color.b * brightness),
      color.a
    };
}