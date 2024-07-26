#ifndef POINT_HPP
#define POINT_HPP
#include <vector_types.h>
#include <cuda_runtime.h>
#include <SFML/Graphics.hpp>
#include <iostream>

struct Point {
    float2 pos{};
    float2 prevPos{};
    float2 force{};
    float mass = 1.0f;
    int id{};

    Point(float x, float y, float m = 1.0f) {
        pos.x = x;
        pos.y = y;
        prevPos.x = x;
        prevPos.y = y;
        mass = m;
        id = 0;
    }
    __host__ __device__ void update(float dt);
    __host__ __device__ float2 getVelocity() const;
    __host__ __device__ float distanceTo(const Point& other) const;
    __host__ __device__ float angleTo(const Point& other) const;

    __host__ __device__ void rotate(const float2& origin, float angle);
    __host__ __device__ void applyForce(float2 force);

    void render(sf::RenderWindow& viewer, sf::Color color) const {
        sf::CircleShape circle(mass);
        circle.setFillColor(color);
        circle.setPosition({pos.x - mass, pos.y - mass});
        viewer.draw(circle);
    }
};

#endif