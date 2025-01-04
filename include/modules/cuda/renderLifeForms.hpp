#ifndef RENDER_LIFE_FORMS
#define RENDER_LIFE_FORMS

#include <modules/physics/point.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <modules/cuda/structures/GPUVector.hpp>
#include <modules/cuda/structures/CGPUValue.hpp>
#include <geneticAlgorithm/cellParts/cellLink.hpp>
#include <geneticAlgorithm/lifeform.hpp>

void renderLifeForms(
  GPUVector<LifeForm>& lifeForms,
  GPUVector<Cell>& cells,
  GPUVector<CellLink>& cellLinks,
  GPUVector<Point>& points,
  CGPUValue<sf::FloatRect>& screenBounds);

#endif