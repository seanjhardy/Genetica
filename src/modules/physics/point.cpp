#include <modules/physics/point.hpp>
#include <modules/utils/vector_functions.hpp>
#include <modules/utils/operations.hpp>
#include <modules/graphics/vertexManager.hpp>
#include <modules/utils/gpu/mathUtils.hpp>

float2 Point::getVelocity() const {
    float2 d = make_float2(pos.s[0] - prevPos.s[0], pos.s[1] - prevPos.s[1]);
    float speed = magnitude(d * d);
    float dir = atan2f(d.s[1], d.s[0]);
    return make_float2(speed, dir);
}

void Point::update() {
    float2 velocity = pos - prevPos;
    float2 accel = force / radius;

    float2 newPosition = pos + velocity * 0.99f + accel;
    prevPos = pos;
    pos = newPosition;

    force = make_float2(0.0f, 0.0f);
}

void Point::setPos(float2 newPos) {
    pos = make_float2(newPos.s[0], newPos.s[1]);
    prevPos = pos;
}

float Point::distanceTo(const Point& other) const {
    return distanceBetween(getPos(), other.getPos());
}

float Point::angleTo(const Point& other) const {
    return atan2f(other.pos.s[1] - pos.s[1], other.pos.s[0] - pos.s[0]);
}

void Point::rotate(const float2& origin, float angleToRotate) {
    float2 d = pos - origin;
    float newX = origin.s[0] + cosf(angleToRotate) * d.s[0] - sinf(angleToRotate) * d.s[1];
    float newY = origin.s[1] + sinf(angleToRotate) * d.s[0] + cosf(angleToRotate) * d.s[1];
    pos = make_float2(newX, newY);
}

void Point::render(VertexManager& viewer, sf::Color colour) const {
    viewer.addCircle(getPos(), radius, colour);
}
