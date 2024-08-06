// environment.hpp
#ifndef ENVIRONMENT
#define ENVIRONMENT

#include "SFML/Graphics.hpp"
#include "utility"
#include "vector"
#include <modules/graphics/vertexManager.hpp>
#include <modules/utils/print.hpp>

/**
 * Environment is an abstract class which provides a wrapper for the simulation to run.
 */
class Environment {
public:
    static const int MAX_TITLE_LENGTH = 64;
    Environment(const std::string& str, const sf::FloatRect& bounds) : bounds(bounds) {
        title = str;
    }

    virtual ~Environment() = default;

    virtual void simulate(float deltaTime) = 0;
    virtual void render(VertexManager& window) = 0;
    virtual void reset() = 0;

    [[nodiscard]] char* getTitle() const {
        char* result = new char[MAX_TITLE_LENGTH];
        // Copy the input string into the myString array
        strcpy_s(result, title.size() + 1, title.c_str());
        // Ensure null termination
        result[MAX_TITLE_LENGTH - 1] = '\0';
        return result;
    }
    [[nodiscard]] sf::FloatRect getBounds() const { return bounds; }
    float dt = 0.0f;

protected:
    std::string title;
    sf::FloatRect bounds;
    int time = 0;
};

#endif