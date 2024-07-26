#ifndef POINT_CU
#define POINT_CU

#include "point.hpp"
#include "../utils/floatOps.hpp"

 __host__ __device__ float2 Point::getVelocity() const {
    float x_vel = pos.x - prevPos.x;
    float y_vel = pos.y - prevPos.y;
    float speed = sqrtf(x_vel * x_vel + y_vel * y_vel);
    float dir = atan2f(y_vel, x_vel);
    return make_float2(speed, dir);
}

__host__ __device__ void Point::update(float dt) {
    float acceleration_x = force.x / mass;
    float acceleration_y = force.y / mass;


    float2 newPosition = pos + (pos - prevPos) * pow(0.9f, dt)
                         + make_float2(acceleration_x * dt * dt,
                                       acceleration_y * dt * dt);

    prevPos = newPosition;
    pos = prevPos;

    force = float2(0.0f, 0.0f);
}

__host__ __device__ float Point::distanceTo(const Point& other) const {
    float2 d = pos - other.pos;
    return sqrtf(d.x * d.x + d.y * d.y);
}

__host__ __device__ float Point::angleTo(const Point& other) const{
    return atan2f(other.pos.y - pos.y, other.pos.x - pos.x);
}


__host__ __device__ void Point::rotate(const float2& origin, float angle) {
    float2 d = pos - origin;
    pos.x = origin.x + cosf(angle) * d.x - sinf(angle) * d.y;
    pos.y = origin.y + sinf(angle) * d.x + cosf(angle) * d.y;
}

__host__ __device__ void Point::applyForce(float2 f) {
    force = force + f;
}

#endif
