#ifndef ROCK
#define ROCK
#include <SFML/Graphics.hpp>
#include <modules/verlet/point.hpp>
#include <modules/utils/floatOps.hpp>

class Rock : public Point {
public:

    Rock(float x, float y, float m = 1.0f) : Point(x, y, m) {}

    void render(VertexManager& viewer) {
        float r = mass / 2;
        viewer.addRectangle(pos - r,
                            {pos.x + r, pos.y - r},
                            {pos.x - r, pos.y + r},
                            pos + r, sf::Color(100, 100, 100));
    }
};

#endif