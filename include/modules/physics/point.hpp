#ifndef POINT_HPP
#define POINT_HPP

#include "vector_types.h"
#include "cuda_runtime.h"
#include "SFML/Graphics.hpp"

class VertexManager;

struct ParentChildLink {
    int startPoint;
    int endPoint;
    int parentStartPoint;
    int parentEndPoint;
    float2 pointOnParent;
    float targetAngle;
    float stiffness;
};

class Point {
public:
    size_t entityID{};
    double2 pos{};
    double2 prevPos{};
    double2 force{};
    double2 deltaPos{};
    int connections = 0;
    float radius = 1.0f;

    Point() : pos{0,0}, prevPos{0,0}, force{0,0}, deltaPos{0, 0} {}
    Point(const size_t id, const float x, const float y, const float r = 1.0f) {
        entityID = id;
        pos.x = x;
        pos.y = y;
        prevPos.x = x;
        prevPos.y = y;
        deltaPos.x = 0;
        deltaPos.y = 0;
        radius = r;
    }
    __host__ __device__ void setPos(float2 pos);
    __host__ __device__ void update(float dt);
    __host__ __device__ double2 getVelocity() const;
    __host__ __device__ float distanceTo(const Point& other) const;
    __host__ __device__ float angleTo(const Point& other) const;

    __host__ __device__ void rotate(const float2& origin, float angle);
    __host__ __device__ float2 getPos() const {
        return make_float2(pos.x, pos.y);
    }

    void render(VertexManager& viewer, sf::Color colour) const;
};


#endif