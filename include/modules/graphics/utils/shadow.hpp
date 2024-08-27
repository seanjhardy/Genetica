#ifndef SHADOW
#define SHADOW

#include "SFML/Graphics.hpp"
#include <array>

class Shadow {
public:
    explicit Shadow(float value, sf::Color color = sf::Color::Black,
                        float offsetLeft = 0.0f, float offsetTop = 0.0f)
                        : m_value(value), m_color(color) {
        m_offset = {offsetLeft, offsetTop};
    }

    [[nodiscard]] sf::Color getColor() const { return m_color; }
    [[nodiscard]] std::array<float, 2> getOffset() const { return m_offset; }
    [[nodiscard]] float getSize() const { return m_value; }

private:
    float m_value;
    std::array<float, 2> m_offset{};
    sf::Color m_color;
};

#endif