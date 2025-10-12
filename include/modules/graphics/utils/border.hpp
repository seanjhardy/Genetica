#ifndef BORDER
#define BORDER

#include "SFML/Graphics.hpp"
#include <array>
#include "modules/utils/gpu/mathUtils.hpp"

class Border {
public:
    explicit Border(float value, sf::Color color = sf::Color::Black,
        float topLeftRadius = 0.0f, float topRightRadius = 0.0f,
        float bottomRightRadius = 0.0f, float bottomLeftRadius = 0.0f)
        : m_value(value), m_color(color) {
        m_radii = { topLeftRadius, topRightRadius, bottomRightRadius, bottomLeftRadius };
    }

    [[nodiscard]] sf::Color getColor() const { return m_color; }
    [[nodiscard]] sf::FloatRect getRadius() const { return m_radii; }
    [[nodiscard]] float getStroke() const { return m_value; }

private:
    float m_value;
    sf::FloatRect m_radii{};
    sf::Color m_color;
};

#endif