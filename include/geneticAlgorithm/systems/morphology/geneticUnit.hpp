#ifndef GENETIC_UNIT
#define GENETIC_UNIT

#include "modules/utils/print.hpp"
#include <vector_types.h>

/**
 * A genetic element that can be part of a gene regulatory network
 */
struct GeneticUnit {
    static constexpr float DISTANCE_THRESHOLD = 0.5f;
    bool sign;
    float modifier;
    float3 embedding;
};

#endif