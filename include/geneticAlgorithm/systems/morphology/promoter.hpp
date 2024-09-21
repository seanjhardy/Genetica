#ifndef PROMOTER
#define PROMOTER

#include "./geneticUnit.hpp"
#include "./gene.hpp"
#include <modules/utils/print.hpp>

/**
 * An element that promotes the level of other elements
 */
class Promoter : public GeneticUnit {
public:
    enum class PromoterType {
        Additive,
        Multiplicative
    } promoterType;

    virtual ~Promoter() = default;
    Promoter(PromoterType promoterType,
             bool sign,
             float modifier,
             const float* embedding)
      : GeneticUnit(sign, modifier, embedding),
        promoterType(promoterType) {}

    float calculateActivity(std::unordered_map<Gene*, float>& levels,
                            std::map<std::pair<Promoter*, Gene*>, float>& promoterFactorAffinities) {
        float activity = 0.0f;
        for (auto& [factor, level] : levels) {
            float affinity = promoterFactorAffinities.at({this, factor});
            activity += level * affinity;
        }
        return activity;
    }
};

#endif