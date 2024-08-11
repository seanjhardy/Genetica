#ifndef POINT_HPP
#define POINT_HPP
#include "vector_types.h"
#include "cuda_runtime.h"
#include "SFML/Graphics.hpp"
#include "iostream"
#include <modules/graphics/vertexManager.hpp>

struct Point {
    float2 pos{};
    float2 prevPos{};
    float2 force{};
    float mass = 1.0f;

    Point() : pos{0,0}, prevPos{0,0}, force{0,0}, mass(1.0f) {}
    Point(float x, float y, float m = 1.0f) {
        pos.x = x;
        pos.y = y;
        prevPos.x = x;
        prevPos.y = y;
        mass = m;
    }
    __host__ __device__ void setPos(float2 pos);
    __host__ __device__ void update(float dt);
    __host__ __device__ float2 getVelocity() const;
    __host__ __device__ float distanceTo(const Point& other) const;
    __host__ __device__ float angleTo(const Point& other) const;

    __host__ __device__ void rotate(const float2& origin, float angle);

    void render(VertexManager& viewer, sf::Color colour) const {
        viewer.addCircle(pos, mass, colour);
    }
};

struct Connection {
    size_t a;
    size_t b;
    float distance;
};

struct AngleConstraint {
    size_t a;
    size_t b;
    size_t parentStart;
    size_t parentEnd;
    float targetAngle;
    float stiffness;
};

#endif