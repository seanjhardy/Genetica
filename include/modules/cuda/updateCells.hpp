#ifndef UPDATE_CELLS
#define UPDATE_CELLS

#include <modules/physics/point.hpp>
#include "modules/utils/GPU/mathUtils.hpp"
#include "modules/cuda/structures/GPUVector.hpp"
#include <geneticAlgorithm/lifeform.hpp>
#include <modules/utils/structures/dynamicStableVector.hpp>
#include <modules/cuda/structures/cellGrowthData.hpp>

void updateCells(dynamicStableVector<LifeForm>& lifeForms,
                 const GPUVector<Cell>& cells,
                 const GPUVector<Point>& points,
                 cellGrowthData& cellDivisionData);

#endif