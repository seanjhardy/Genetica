#ifndef GENETIC_UNIT
#define GENETIC_UNIT

#include "modules/utils/print.hpp"
#include <array>

/**
 * A genetic element that can be part of a gene regulatory network
 */
class GeneticUnit {
public:
    static constexpr float DISTANCE_THRESHOLD = 0.5f;
    static constexpr int N = 3;

    bool sign;
    float modifier;
    std::array<float, N> embedding;

    GeneticUnit(bool sign,
                float modifier, const float* embedding)
        : sign(sign), modifier(modifier){
        for (int i = 0; i < N; i++) {
            this->embedding[i] = embedding[i];
        }
    }

    [[nodiscard]] float calculateAffinity(const GeneticUnit& other) const {
        float distance = 0.0;
        for (int i = 0; i < GeneticUnit::N; i++) {
            distance += std::pow(embedding.at(i) - other.embedding.at(i), 2);
        }
        distance = std::sqrt(distance);
        if (distance > GeneticUnit::DISTANCE_THRESHOLD) return 0.0;
        float affinitySign = (sign == other.sign) ? 1.0f : -1.0f;
        return affinitySign *
             (2.0f * std::abs(modifier * other.modifier)
                * (GeneticUnit::DISTANCE_THRESHOLD - distance)) /
             (10.0f * distance + std::abs(modifier * other.modifier));
    }
};
#endif