#ifndef UPDATE_CELLS
#define UPDATE_CELLS

#include <modules/physics/point.hpp>
#include "modules/cuda/structures/GPUVector.hpp"
#include <geneticAlgorithm/lifeform.hpp>

void updateCells(GPUVector<Cell>& cells,
                 GPUVector<Point>& points,
                 float dt = 1.0f);

#endif