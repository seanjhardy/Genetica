// environment.hpp
#pragma once

#include <SFML/Graphics.hpp>
#include <utility>
#include <vector>
#include "../modules/graphics/vertexManager.hpp"

class Environment {
public:
    Environment(const char *str, const sf::FloatRect& bounds)
            : bounds(bounds) {
        // Copy the input string into the myString array
        strncpy_s(title, str, sizeof(title) - 1);
        // Ensure null termination
        title[sizeof(title) - 1] = '\0';
    }

    virtual ~Environment() = default;

    virtual void simulate(float deltaTime) = 0;
    virtual void render(VertexManager& window) = 0;
    virtual void reset() = 0;

    [[nodiscard]] const char* getTitle() const { return title; }
    [[nodiscard]] sf::FloatRect getBounds() const { return bounds; }
    float dt = 0.0f;

protected:
    char title[20]{};
    sf::FloatRect bounds;
    int time = 0;
};