#ifndef UPDATE_LIFEFORM
#define UPDATE_LIFEFORM

#include <modules/physics/point.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "modules/cuda/structures/GPUVector.hpp"
#include "modules/cuda/structures/CGPUValue.hpp"
#include <geneticAlgorithm/cellParts/cellLink.hpp>

void cloneLifeForm(GPUVector<LifeForm>& lifeForms,
                  int lifeFormId);
void killLifeForm(GPUVector<LifeForm>& lifeForms,
                   int lifeFormId);
void mutateLifeForm(GPUVector<LifeForm>& lifeForms,
                   int lifeFormId);
void energiseLifeForm(GPUVector<LifeForm>& lifeForms,
                   int lifeFormId);

LifeForm* getLifeForm(GPUVector<LifeForm>& lifeForms, int lifeFormId);

#endif