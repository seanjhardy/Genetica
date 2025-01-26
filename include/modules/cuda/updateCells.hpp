#ifndef UPDATE_CELLS
#define UPDATE_CELLS

#include <modules/physics/point.hpp>
#include "modules/cuda/structures/GPUVector.hpp"
#include <geneticAlgorithm/lifeform.hpp>
#include <modules/utils/structures/DynamicStableVector.hpp>

void updateCells(DynamicStableVector<LifeForm>& lifeForms,
                 const GPUVector<Cell>& cells,
                 const GPUVector<Point>& points);

#endif