#ifndef GUIUTILS_HPP
#define GUIUTILS_HPP

#include <cmath>
#include <vector>
#include "../verlet/point.hpp"
#include <SFML/Graphics.hpp>

inline void drawPolygon(sf::RenderWindow& window, const std::vector<float2>& points, const sf::Color& color) {
    if (points.size() < 3) return; // A polygon needs at least 3 points

    sf::ConvexShape polygon;
    polygon.setPointCount(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        polygon.setPoint(i, sf::Vector2f(points[i].x, points[i].y));
    }
    polygon.setFillColor(color);
    window.draw(polygon);
}

inline void drawCircle(sf::RenderWindow& window, sf::Vector2f position, float radius, sf::Color color) {
    sf::CircleShape circle(50);
    //circle.setFillColor(color);
    //std::cout << position.x << " " << position.y << " " << radius << " " <<  std::endl;
    //circle.setPosition({position.x - radius, position.y - radius});
    //window.draw(circle);
}

#endif // GUIUTILS_HPP