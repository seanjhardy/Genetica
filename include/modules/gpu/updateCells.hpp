#ifndef UPDATE_CELLS
#define UPDATE_CELLS

#include <modules/physics/point.hpp>
#include "modules/utils/gpu/mathUtils.hpp"
#include "modules/gpu/structures/GPUVector.hpp"
#include <geneticAlgorithm/lifeform.hpp>
#include <modules/utils/structures/dynamicStableVector.hpp>
#include <modules/gpu/structures/cellGrowthData.hpp>

void updateCells(dynamicStableVector<LifeForm>& lifeForms,
  const GPUVector<Cell>& cells,
  const GPUVector<Point>& points,
  cellGrowthData& cellDivisionData);

#endif