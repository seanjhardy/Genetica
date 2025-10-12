#ifndef PROMOTER
#define PROMOTER

#include "./geneticUnit.hpp"
#include "./gene.hpp"
#include <modules/utils/print.hpp>
#include "modules/gpu/structures/GPUVector.hpp"

/**
 * An element that promotes the level of other elements
 */
struct Promoter : GeneticUnit {
    enum class PromoterType {
        Additive,
        Multiplicative
    } promoterType;

    Promoter() : promoterType(PromoterType::Additive) {}
    Promoter(PromoterType promoterType, bool sign, float modifier, float3 embedding)
        : GeneticUnit(sign, modifier, embedding), promoterType(promoterType) {
    }

    static float calculateActivity(int index,
        StaticGPUVector<float>& levels,
        StaticGPUVector<float>& promoterFactorAffinities) {
        float activity = 0.0f;
        for (int i = 0; i < levels.size(); i++) {
            int promoterFactorIndex = index * levels.size() + i;
            if (promoterFactorIndex < promoterFactorAffinities.size()) {
                float affinity = promoterFactorAffinities[promoterFactorIndex];
                activity += levels[i] * affinity;
            }
        }
        return activity;
    }
};

#endif