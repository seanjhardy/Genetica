#ifndef POINT_HPP
#define POINT_HPP

#include "modules/utils/vector_types.hpp"
#include "SFML/Graphics.hpp"

class VertexManager;

class Point {
public:
    size_t entityID{};
    double2 pos{};
    double2 prevPos{};
    double2 force{};
    float angle;
    double radius = 1.0f;

    Point() : pos(make_double2(0, 0)), prevPos(make_double2(0, 0)), force(make_double2(0, 0)) {}
    Point(const size_t id, const float x, const float y, const float r = 1.0f, float a = 0.0f) {
        entityID = id;
        pos = make_double2(x, y);
        prevPos = make_double2(x, y);
        radius = r;
        angle = a;
    }
    void setPos(float2 pos);
    void update();
    double2 getVelocity() const;
    float distanceTo(const Point& other) const;
    float angleTo(const Point& other) const;

    void rotate(const double2& origin, double angle);
    float2 getPos() const {
        return make_float2(pos.s[0], pos.s[1]);
    }

    void render(VertexManager& viewer, sf::Color colour) const;
};


#endif