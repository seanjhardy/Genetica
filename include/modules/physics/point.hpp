#ifndef POINT_HPP
#define POINT_HPP
#include "vector_types.h"
#include "cuda_runtime.h"
#include "SFML/Graphics.hpp"
#include "iostream"
#include <modules/graphics/vertexManager.hpp>

struct Connection {
    int p1;
    int p2;
    float length;
};

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
    int entityID{};
    float2 pos{};
    float2 prevPos{};
    float2 force{};
    float mass = 1.0f;

    Point() : pos{0,0}, prevPos{0,0}, force{0,0}, mass(1.0f) {}
    Point(int id, float x, float y, float m = 1.0f) {
        entityID = id;
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


#endif