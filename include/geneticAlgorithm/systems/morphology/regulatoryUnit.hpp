#ifndef REGULATORY_UNIT
#define REGULATORY_UNIT

#include <vector>
#include <modules/graphics/vertexManager.hpp>
#include <modules/utils/stringUtils.hpp>
#include "modules/cuda/structures/GPUVector.hpp"

struct RegulatoryUnit {
    static constexpr float W = 10.0f;

    staticGPUVector<int> promoters;
    staticGPUVector<int> factors;

    __host__ __device__ RegulatoryUnit() = default;

    __device__ void calculateActivation(
      staticGPUVector<Promoter>& grnPromoters,
      staticGPUVector<Gene>& grnFactors,
      staticGPUVector<float>& factorLevels,
      staticGPUVector<float>& promoterFactorAffinities) {

        // Calculate activity of promoters based on input factors
        // and combine those activities into one overall regulatory unit activity
        float additivePromoterValue = 0;
        float multiplicativePromoterValue = 1;
        for (size_t i = 0; i < promoters.size(); i++) {
            size_t promoterIndex = promoters[i];
            Promoter* promoter = &grnPromoters[promoterIndex];
            float promoterActivity = promoter->calculateActivity(promoterIndex, factorLevels, promoterFactorAffinities);

            if (promoter->promoterType == Promoter::PromoterType::Additive) {
                additivePromoterValue += promoterActivity;
            } else if (promoter->promoterType == Promoter::PromoterType::Multiplicative) {
                multiplicativePromoterValue *= promoterActivity;
            }
        }

        // Calculate the overall activity of the regulatory unit
        float value = additivePromoterValue * multiplicativePromoterValue;
        float transformedValue = 1.0f / (1.0f + std::exp(-W * (value - 0.5f)));

        // Prevent 0 values introducing a small positive term
        if (value < 0.01f) transformedValue = 0.0f;

        // Calculate the amount of each factor produced based on the unit's activity
        for (int i = 0; i < factors.size(); i++) {
            int geneIndex = factors[i];
            auto* gene = &grnFactors[geneIndex];
            // Only update internal and external products
            if (gene->factorType != Gene::FactorType::InternalProduct &&
                gene->factorType != Gene::FactorType::ExternalProduct) {
                continue;
            }
            factorLevels[geneIndex] += transformedValue;
        }
    }

    /*void render(VertexManager& vertexManager, float2 pos, float scale,
                std::vector<Promoter>& grnPromoters,
                std::vector<Gene>& grnFactors,
                std::unordered_map<Gene*, float> factorLevels) {
        pos -= make_float2(scale, scale/2);
        float promoterWidth = scale * 2.0f / promoters.size();
        for (int promoterIndex = 0; promoterIndex < promoters.size(); promoterIndex++) {
            vertexManager.addFloatRect({promoterWidth*(float)promoterIndex + pos.x, pos.y,
                                        promoterWidth - 0.005f, scale/2.0f}, sf::Color::Red);
        }

        int factorIndex = 0;
        float factorWidth = scale * 2.0f / factors.size();
        std::string factorLevel;
        for (auto factorId : factors) {
            auto& factor = grnFactors[factorId];
            factorLevel = roundToDecimalPlaces(factorLevels[&factor], 2);
            vertexManager.addFloatRect({factorWidth*(float)factorIndex + pos.x, pos.y + scale/2.0f,
                                        factorWidth  - 0.005f, scale/2.0f}, sf::Color::Green);
            factorIndex += 1;
        }
        vertexManager.addText(factorLevel,
                              {pos.x + scale,
                               pos.y + 0.1f},
                              0.0015, sf::Color::White, TextAlignment::Center, 2);
    }*/
};

#endif //REGULATORY_UNIT