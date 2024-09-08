#ifndef REGULATORY_UNIT
#define REGULATORY_UNIT

#include <vector>
#include "./geneticUnit.hpp"

class RegulatoryUnit {
public:
    static constexpr float W = 10.0f;
    static constexpr float SYNTHESIS_RATE = 1.0f;
    static constexpr float DEGRADATION_RATE = 1.0f;

    std::vector<Promoter> promoters;
    std::vector<Gene> genes;

    std::unordered_map<Gene*, float> factorLevels;

    float calculateActivation(std::map<Promoter*, float> promoterActivities) {
        float additivePromoterValue = 0;
        float multiplicativePromoterValue = 0;
        for (auto& promoter : promoters) {
            if (promoter.promoterType == Promoter::PromoterType::Additive) {
                additivePromoterValue += promoterActivities[&promoter];
            } else if (promoter.promoterType == Promoter::PromoterType::Multiplicative) {
                if (multiplicativePromoterValue == 0) {
                    multiplicativePromoterValue = 1;
                }
                multiplicativePromoterValue *= promoterActivities[&promoter];
            }
        }

        float value = additivePromoterValue * multiplicativePromoterValue;
        float transformedValue = 1.0f / (1.0f + std::exp(-W * (value - 0.5)));

        for (auto& [gene, level] : factorLevels) {

            factorLevels[gene] += SYNTHESIS_RATE * transformedValue
                                  - DEGRADATION_RATE * factorLevels[gene];
        }
    }
};

#endif //REGULATORY_UNIT