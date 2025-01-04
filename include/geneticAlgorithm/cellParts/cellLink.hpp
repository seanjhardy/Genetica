#ifndef CELL_LINK
#define CELL_LINK

#include <geneticAlgorithm/cellParts/cell.hpp>
#include <modules/physics/point.hpp>

class LifeForm;

class CellLink {
public:
    static constexpr float INITIAL_DISTANCE = 2.0f;

    size_t lifeFormId;
    size_t cellAId;
    size_t cellBId;
    size_t p1;
    size_t p2;
    float length;

    CellLink(size_t lifeFormId, size_t cellAId, size_t cellBId, size_t p1, size_t p2, float startLength);
};

#endif