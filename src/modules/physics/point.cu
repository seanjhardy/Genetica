#ifndef POINT_CU
#define POINT_CU

#include <modules/physics/point.hpp>
#include <modules/utils/floatOps.hpp>
#include <modules/utils/fastMath.hpp>

 __host__ __device__ float2 Point::getVelocity() const {
    float2 d = pos - prevPos;
    float speed = sqrtf(sum(d*d));
    float dir = FastMath::atan2f(d.y, d.x);
    return make_float2(speed, dir);
}

__host__ __device__ void Point::update(float dt) {
    float2 velocity = pos - prevPos;
    float2 accel = force / mass;

    float2 newPosition = pos + velocity * pow(0.99, dt) + accel * dt * dt;
    prevPos = pos;
    pos = newPosition;

    force = float2(0.0f, 0.0f);
}

__host__ __device__ void Point::setPos(float2 newPos) {
    pos = newPos;
    prevPos = newPos;
}

__host__ __device__ float Point::distanceTo(const Point& other) const {
    float2 d = pos - other.pos;
    return sqrtf(sum(d*d));
}

__host__ __device__ float Point::angleTo(const Point& other) const{
    return FastMath::atan2f(other.pos.y - pos.y, other.pos.x - pos.x);
}

__host__ __device__ void Point::rotate(const float2& origin, float angle) {
    float2 d = pos - origin;
    pos.x = origin.x + cosf(angle) * d.x - sinf(angle) * d.y;
    pos.y = origin.y + sinf(angle) * d.x + cosf(angle) * d.y;
}

#endif

