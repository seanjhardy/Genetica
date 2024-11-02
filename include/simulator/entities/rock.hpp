#ifndef ROCK
#define ROCK
#include "SFML/Graphics.hpp"
#include "modules/physics/point.hpp"
#include "modules/utils/floatOps.hpp"
#include <modules/graphics/vertexManager.hpp>
#include "entity.hpp"

class Rock : public Entity {
public:

    Rock(float x, float y, float m = 1.0f) : Entity({x,y}) {
        mass = m;
    }

    void render(VertexManager& viewer) {
        float r = mass / 2;
        viewer.addRectangle(pos - r,
                            {pos.x + r, pos.y - r},
                            {pos.x - r, pos.y + r},
                            pos + r, sf::Color(100, 100, 100));
    }
};

#endif