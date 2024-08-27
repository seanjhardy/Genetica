#include <SFML/Graphics.hpp>
#include <modules/utils/GUIUtils.hpp>
#include <vector_types.h>

sf::Color brightness(sf::Color color, float brightness) {
    return {
      static_cast<sf::Uint8>(color.r * brightness),
      static_cast<sf::Uint8>(color.g * brightness),
      static_cast<sf::Uint8>(color.b * brightness),
      color.a
    };
}

sf::FloatRect computeBoundingBox(std::vector<float2> points) {
    if (points.empty()) return {};

    float minX = points[0].x;
    float maxX = points[0].x;
    float minY = points[0].y;
    float maxY = points[0].y;

    for (const auto& p : points) {
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
    }

    return {minX, minY, maxX - minX, maxY - minY};
}