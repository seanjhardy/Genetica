#ifndef PROMOTER
#define PROMOTER

#include "./geneticUnit.hpp"
#include "./gene.hpp"
#include <modules/utils/print.hpp>
#include "modules/cuda/structures/GPUVector.hpp"

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

    __device__ static float calculateActivity(int index,
                            int numProducts,
                            StaticGPUVector<float>& levels,
                            StaticGPUVector<float>& promoterFactorAffinities) {
        float activity = 0.0f;
        for (int i = 0; i < numProducts; i++) {
            int promoterFactorIndex = index * numProducts + i;
            float affinity = promoterFactorAffinities[promoterFactorIndex];
            activity += levels[i] * affinity;
        }
        return activity;
    }
};

#endif