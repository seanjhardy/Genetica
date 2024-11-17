#include "geneticAlgorithm/sequencer.hpp"
#include "modules/utils/genomeUtils.hpp"
#include <string>
#include <utility>
#include <modules/utils/GPU/GPUUtils.hpp>
#include <geneticAlgorithm/systems/morphology/updateGRN.hpp>

void sequenceGRN(LifeForm* lifeForm, const Genome& genome) {
    // Temporary vars
    std::vector<Gene> factors;
    std::vector<Promoter> promoters;
    std::vector<Effector> effectors;
    std::vector<RegulatoryUnit> regulatoryUnits;
    RegulatoryUnit regulatoryUnit;

    bool readingPromoters = true;
    for (auto [id, sequence] : genome.hoxGenes) {
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
                Promoter promoter = Promoter(promoterType, sign, modifier, embedding);

                if (!readingPromoters) {
                    regulatoryUnits.push_back(regulatoryUnit);
                    regulatoryUnit = RegulatoryUnit();
                    readingPromoters = true;
                }

                regulatoryUnit.promoters.push_back(promoters.size());
                promoters.push_back(promoter);
            }

            // Genes - internal product, external product, receptor
            if (type == 3 || type == 4 || type == 5) {
                Gene::FactorType geneTypes[] = {
                  Gene::FactorType::ExternalProduct,
                 Gene::FactorType::InternalProduct,
                 Gene::FactorType::Receptor,};

                Gene gene = Gene(geneTypes[type - 3], sign, modifier, embedding);

                if (regulatoryUnit.promoters.empty()) continue;
                readingPromoters = false;
                regulatoryUnit.factors.push_back(factors.size());
                factors.push_back(gene);
            }
        } catch (RNAExhaustedException& e) {
            // "RNA exhausted"
        }
    }

    saveGPUArray(lifeForm->grn.factors, factors);
    lifeForm->grn.numFactors = factors.size();

    saveGPUArray(lifeForm->grn.promoters, promoters);
    lifeForm->grn.numPromoters = promoters.size();

    saveGPUArray(lifeForm->grn.effectors, effectors);
    lifeForm->grn.numEffectors = effectors.size();

    saveGPUArray(lifeForm->grn.regulatoryUnits, regulatoryUnits);
    lifeForm->grn.numRegulatoryUnits = regulatoryUnits.size();

    computeAffinities(lifeForm->grn);
}
