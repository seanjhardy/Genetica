#ifndef CELL_LINK
#define CELL_LINK

#include <geneticAlgorithm/cellParts/cell.hpp>
#include <modules/physics/point.hpp>

class LifeForm;

class CellLink {
public:
    static constexpr float INITIAL_DISTANCE = 2.0f;

    int connectionIdx;
    LifeForm* lifeForm;
    Cell* cell1;
    Cell* cell2;

    CellLink(LifeForm* lifeForm, Cell* a, Cell* b, float startLength);

    void adjustSize(float distance);
    void moveCell1(Cell* newCell);
    void moveCell2(Cell* newCell);

    [[nodiscard]] float getBuildCost() const;
};

#endif