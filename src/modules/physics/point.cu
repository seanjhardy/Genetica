#include <modules/physics/point.hpp>
#include <modules/utils/floatOps.hpp>
#include <modules/graphics/vertexManager.hpp>

 __host__ __device__ double2 Point::getVelocity() const {
    float2 d = make_float2(pos.x - prevPos.x, pos.y - prevPos.y);
    float speed = sqrtf(sum(d*d));
    float dir = FastMath::atan2f(d.y, d.x);
    return make_double2(speed, dir);
}

__host__ __device__ void Point::update(float dt) {
    double2 velocity = pos - prevPos;
    double2 accel = force / radius;

    double2 newPosition = pos + velocity * 0.99 + accel * dt * dt;
    prevPos = pos;
    pos = newPosition;

    force = double2(0.0f, 0.0f);
}

__host__ __device__ void Point::setPos(float2 newPos) {
    pos = make_double2(newPos.x, newPos.y);
    prevPos = pos;
}

__host__ __device__ float Point::distanceTo(const Point& other) const {
    return distanceBetween(getPos(), other.getPos());
}

__host__ __device__ float Point::angleTo(const Point& other) const{
    return FastMath::atan2f(other.pos.y - pos.y, other.pos.x - pos.x);
}

__host__ __device__ void Point::rotate(const float2& origin, float angle) {
    float2 d = getPos() - origin;
    pos.x = origin.x + cosf(angle) * d.x - sinf(angle) * d.y;
    pos.y = origin.y + sinf(angle) * d.x + cosf(angle) * d.y;
}

void Point::render(VertexManager& viewer, sf::Color colour) const {
    viewer.addCircle(getPos(), radius, colour);
}

