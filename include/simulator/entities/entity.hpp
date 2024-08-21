#ifndef ENTITY
#define ENTITY

#include <vector_types.h>
#include "modules/physics/point.hpp"

class Entity : public Point {
public:
    explicit Entity(float2 pos);
    float2 pos;
};

#endif