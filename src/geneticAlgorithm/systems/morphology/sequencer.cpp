#include "geneticAlgorithm/sequencer.hpp"
#include "modules/utils/genomeUtils.hpp"
#include <string>
#include <utility>
#include "modules/cuda/calculateAffinities.hpp"

void sequenceGRN(LifeForm& lifeForm) {
    // Temporary vars
    std::vector<Gene> factors;
    std::vector<Promoter> promoters;
    std::vector<Effector> effectors;
    std::vector<RegulatoryUnit> regulatoryUnits;

    // Temporary variables for regulatory units
    auto regulatoryUnit = RegulatoryUnit();
    std::vector<int> regulatoryPromoters;
    std::vector<int> regulatoryFactors;

    bool readingPromoters = true;
    for (auto [id, sequence] : lifeForm.genome.hoxGenes) {
        std::string rna = std::move(sequence); // Create rna copy
        try {
            int type = (int)(readUniqueBaseRange(rna, 2) * 16) % 5;
            bool sign = readBase(rna) >= 2;
            bool active = readBase(rna) >= 1;
            if (!active) continue;

            // Uniformly distributed modifiers (also allows for large jumps genetically)
            float modifier = readUniqueBaseRange(rna, 8);

            float3 embedding = make_float3(
                readUniqueBaseRange(rna, 8),
                readUniqueBaseRange(rna, 8),
                readUniqueBaseRange(rna, 8));

            // External factors (inputs to grn)
            if (type == 0) {
                Gene::FactorType externalFactorTypes[] = {
                    Gene::FactorType::MaternalFactor,
                    Gene::FactorType::Crowding,
                    Gene::FactorType::Constant,
                    Gene::FactorType::Generation,
                    Gene::FactorType::Energy,
                    Gene::FactorType::Time,
                };
                int subType = (int)(readUniqueBaseRange(rna, 3) * 64) % 6;
                Gene::FactorType externalFactorType = externalFactorTypes[subType];

                float2 extra = {readUniqueBaseRange(rna, 8), readUniqueBaseRange(rna, 8)};
                Gene externalFactor = Gene(externalFactorType,
                                           sign, modifier, embedding, extra);
                factors.push_back(externalFactor);
            }

            // Gene effectors (outputs of grn)
            if (type == 1) {
                Effector::EffectorType subType = static_cast<Effector::EffectorType>(
                    (int)(readUniqueBaseRange(rna, 4) * 256) % (int)Effector::EffectorType::EFFECTOR_LENGTH);

                // If this grn already contains a special node
                // (e.g. maternal factor, crowding, etc.) of this type, skip
                if (std::any_of(effectors.begin(), effectors.end(),
                                [subType](const Effector& effector) {
                                    return effector.effectorType == subType;
                                })) {
                    continue;
                }

                Effector effector = Effector(subType,
                                             sign, modifier, embedding);
                effectors.push_back(effector);
            }

            // Regulatory units (nodes in grn)
            // Promoters (take in factors and produce activity levels excitatory/inhibitory additive/multiplicative)
            if (type == 2) {
                int additive = readBase(rna) >= 1;
                Promoter::PromoterType promoterType = additive
                                                          ? Promoter::PromoterType::Additive
                                                          : Promoter::PromoterType::Multiplicative;
                auto promoter = Promoter(promoterType, sign, modifier, embedding);

                if (!readingPromoters) {
                    regulatoryUnit.promoters = StaticGPUVector(regulatoryPromoters);
                    regulatoryUnit.factors = StaticGPUVector(regulatoryFactors);
                    regulatoryUnits.push_back(regulatoryUnit);
                    regulatoryUnit = RegulatoryUnit();
                    readingPromoters = true;
                }

                regulatoryPromoters.push_back(promoters.size());
                promoters.push_back(promoter);
            }

            // Genes - internal product, external product, receptor
            if (type == 3 || type == 4 || type == 5) {
                Gene::FactorType geneTypes[] = {
                    Gene::FactorType::ExternalProduct,
                    Gene::FactorType::InternalProduct,
                    Gene::FactorType::Receptor,
                };

                Gene gene = Gene(geneTypes[type - 3], sign, modifier, embedding);

                if (regulatoryPromoters.empty()) continue;
                readingPromoters = false;
                regulatoryFactors.push_back((int)factors.size());
                factors.push_back(gene);
            }
        }
        catch (RNAExhaustedException& e) {
            // "RNA exhausted"
        }
    }

    lifeForm.grn.factors = StaticGPUVector(factors);
    lifeForm.grn.promoters = StaticGPUVector(promoters);
    lifeForm.grn.effectors = StaticGPUVector(effectors);

    lifeForm.grn.regulatoryUnits = StaticGPUVector(regulatoryUnits);
    calculateAffinities(lifeForm.grn);
}
