#include <cuda_runtime.h>
#include <geneticAlgorithm/systems/morphogen/morphogen.hpp>

__host__ __device__ float Morphogen::sample(const float2& point) const {
    if (type == 0) {
        float dx = point.x - pos.x;
        float dy = point.y - pos.y;
        float distance = dx * extra.x + dy * extra.y;
        float t = std::max(0.0f, std::min(1.0f, distance));
        return startConcentration + t * (endConcentration - startConcentration);
    }
    if (type == 1) { // Radial
        float dx = point.x - pos.x;
        float dy = point.y - pos.y;
        float distance = sqrt(dx*dx + dy*dy);
        float t = std::min(1.0f, distance / extra.x);
        return startConcentration + t * (endConcentration - startConcentration);
    }
    if (type == 2) { //Sinusoidal
        float dx = point.x - pos.x;
        float dy = point.y - pos.y;
        float distance = dx * extra.x + dy * extra.y;
        return extra.x * sin(2 * M_PI * distance / extra.y + extra.z);
    }
    return 0.0;
};