#include <cuda_runtime.h>
#include <geneticAlgorithm/systems/morphogen/morphogen.hpp>
#include "modules/utils/floatOps.hpp"
#include "modules/noise/random.hpp"

__host__ __device__ float Morphogen::sample(float2 cellPos, float cellRadius,
                                            float distFraction, const float2& point) const {
    float value = 0;
    if (type == 0) { // Linear Gradient
        float2 pos = cellPos + vec(angle) * cellRadius;
        float dx = point.x - pos.x;
        float dy = point.y - pos.y;
        float distance = dx * extra.x + dy * extra.y;
        float t = std::max(0.0f, std::min(1.0f, distance));
        value = startConcentration + t * (endConcentration - startConcentration);
    }
    if (type == 1) { // Radial Gradient
        float2 pos = cellPos + vec(angle) * cellRadius;
        float dx = point.x - pos.x;
        float dy = point.y - pos.y;
        float distance = sqrt(dx*dx + dy*dy);
        float t = std::min(1.0f, distance / extra.x);
        value = startConcentration + t * (endConcentration - startConcentration);
    }
    if (type == 2) { // Sinusoidal
        float2 pos = cellPos + vec(angle) * cellRadius * distFraction;
        float dx = point.x - pos.x;
        float dy = point.y - pos.y;
        float distance = dx * extra.x + dy * extra.y;
        value = startConcentration * sin(2 * M_PI * distance / endConcentration + extra.x);
    }
    if (type == 3) { // Random
        value = Random::random(startConcentration, endConcentration);
    }

    return fmaxf(0, value + Random::random(-0.01,0.01));
};