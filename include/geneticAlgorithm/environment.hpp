// environment.hpp
#ifndef ENVIRONMENT
#define ENVIRONMENT

#include "SFML/Graphics.hpp"
#include "utility"
#include "vector"
#include "modules/graphics/vertexManager.hpp"

/**
 * Environment is an abstract class which provides a wrapper for the simulation to run.
 */
class Environment {
public:
    Environment(const std::string&, const sf::FloatRect& bounds) : bounds(bounds) {
        title = std::move(title);
    }

    virtual ~Environment() = default;

    virtual void simulate(float deltaTime) = 0;
    virtual void render(VertexManager& window) = 0;
    virtual void reset() = 0;

    [[nodiscard]] std::string getTitle() const { return title; }
    [[nodiscard]] sf::FloatRect getBounds() const { return bounds; }
    float dt = 0.0f;

protected:
    std::string title;
    sf::FloatRect bounds;
    int time = 0;
};

#endif