#ifndef ENTITY
#define ENTITY

#include <vector_types.h>
#include "modules/physics/point.hpp"

class Entity : public Point {
public:
    explicit Entity(float2 pos);
    virtual ~Entity() = default;

    virtual void simulate(float dt) = 0;
    virtual void render(VertexManager& viewer) = 0;
};

#endif