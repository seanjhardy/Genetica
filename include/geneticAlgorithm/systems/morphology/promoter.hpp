#ifndef PROMOTER
#define PROMOTER

#include "./geneticUnit.hpp"
#include "./gene.hpp"
#include <modules/utils/print.hpp>

/**
 * An element that promotes the level of other elements
 */
struct Promoter : GeneticUnit {
    enum class PromoterType {
        Additive,
        Multiplicative
    } promoterType;

    Promoter(PromoterType promoterType, bool sign, float modifier, float3 embedding)
        : GeneticUnit(sign, modifier, embedding), promoterType(promoterType) {}

    float calculateActivity(float* levels,
                            float* promoterFactorAffinities) {
        float activity = 0.0f;
        for (auto& [factor, level] : levels) {
            float affinity = promoterFactorAffinities.at({this, factor});
            activity += level * affinity;
        }
        return activity;
    }
};

#endif