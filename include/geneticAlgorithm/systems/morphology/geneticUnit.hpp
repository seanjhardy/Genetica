#ifndef GENETIC_UNIT
#define GENETIC_UNIT

#include "modules/utils/print.hpp"
#include "modules/utils/vector_types.hpp"

/**
 * A genetic element that can be part of a gene regulatory network
 */
struct GeneticUnit {
    static constexpr float DISTANCE_THRESHOLD = 0.5f;
    bool sign{};
    float modifier{};
    float3 embedding{};

    GeneticUnit() = default;
    GeneticUnit(bool sign, float modifier, float3 embedding)
        : sign(sign), modifier(modifier), embedding(embedding) {
    }
};

#endif