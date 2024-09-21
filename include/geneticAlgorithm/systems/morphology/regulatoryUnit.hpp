#ifndef REGULATORY_UNIT
#define REGULATORY_UNIT

#include <vector>
#include "./geneticUnit.hpp"

class RegulatoryUnit {
public:
    static constexpr float W = 10.0f;

    std::vector<int> promoters{};
    std::vector<int> factors{};

    std::unordered_map<Gene*, float> calculateActivation(
      std::vector<Promoter>& grnPromoters,
        std::vector<Gene>& grnFactors,
      std::unordered_map<Gene*, float> factorLevels,
        std::map<std::pair<Promoter*, Gene*>, float> promoterFactorAffinities) {
        // Calculate activity of promoters based on input factors
        std::map<Promoter*, float> promoterActivities;
        for (auto promoterIndex : promoters) {
            Promoter* promoter = &grnPromoters[promoterIndex];
            float promoterActivity = promoter->calculateActivity(factorLevels, promoterFactorAffinities);
            promoterActivities.insert({promoter, promoterActivity});
        }

        // Combine those activities into one overall regulatory unit activity
        float additivePromoterValue = 0;
        float multiplicativePromoterValue = 1;
        for (auto promoterIndex : promoters) {
            auto promoter = grnPromoters[promoterIndex];
            if (promoter.promoterType == Promoter::PromoterType::Additive) {
                additivePromoterValue += promoterActivities[&promoter];
            } else if (promoter.promoterType == Promoter::PromoterType::Multiplicative) {
                multiplicativePromoterValue *= promoterActivities[&promoter];
            }
        }
        float value = additivePromoterValue * multiplicativePromoterValue;
        float transformedValue = 1.0f / (1.0f + std::exp(-W * (value - 0.5)));

        // Calculate the amount of each factor produced based on the unit's activity
        std::unordered_map<Gene*, float> deltaFactorLevels;
        for (auto& factorIndex : factors) {
            auto* gene = &grnFactors[factorIndex];
            // Only update internal and external products
            if (gene->factorType != Gene::FactorType::InternalProduct &&
                gene->factorType != Gene::FactorType::ExternalProduct) {
                continue;
            }
            deltaFactorLevels[gene] += transformedValue;
        }
        return deltaFactorLevels;
    }
};

#endif //REGULATORY_UNIT