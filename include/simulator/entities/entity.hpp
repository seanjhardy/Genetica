#ifndef ENTITY
#define ENTITY

#include <vector_types.h>
#include "modules/physics/point.hpp"

class Entity : public Point {
public:
    explicit Entity(float2 pos);
    virtual ~Entity() = default;
    float2 pos;
};

#endif