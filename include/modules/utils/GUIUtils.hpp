#ifndef GUIUtils
#define GUIUtils

#include <SFML/Graphics.hpp>
#include "modules/utils/vector_types.hpp"

enum PIGMENT {
    Red,
    Green,
    Blue,
};

sf::Color brightness(sf::Color color, float brightness);

sf::FloatRect computeBoundingBox(std::vector<float2> points);

sf::Color HSVtoRGB(float H, float S, float V);

#endif