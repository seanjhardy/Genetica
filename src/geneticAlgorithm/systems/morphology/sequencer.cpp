#include "geneticAlgorithm/sequencer.hpp"
#include "modules/utils/genomeUtils.hpp"
#include <string>
#include <utility>

void sequenceGRN(LifeForm* lifeForm, const Genome& genome) {
    lifeForm->grn.lifeForm = lifeForm;

    RegulatoryUnit regulatoryUnit;
    bool readingPromoters = true;
    for (auto [id, sequence] : genome.hoxGenes) {
        std::string rna = std::move(sequence); // Create rna copy
        try {
            int type = readBase(rna);
            bool sign = readBase(rna) >= 2;
            bool active = readBase(rna) >= 1;
            if (!active) continue;

            // Uniformly distributed modifiers (also allows for large jumps genetically)
            float modifier = readUniqueBaseRange(rna, 8);

            float embedding[GeneticUnit::N];
            for (float & i : embedding) {
                // spatial fidelity = (4 values * 20 per dimension) = 80
                // 80^(3 dimensions) = 512000 unique positions
                // Uniform distribution using readUniqueBaseRange
                i = readUniqueBaseRange(rna, 8);
            }

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

                // If this grn already contains a special node
                // (e.g. maternal factor, crowding, etc.) of this type, skip
                if (std::any_of(lifeForm->grn.factors.begin(), lifeForm->grn.factors.end(),
                                [externalFactorType](const Gene& gene) {
                                    return gene.factorType == externalFactorType;
                                })) {
                    continue;
                }
                float2 extra = {(readBase(rna) >= 2 ? -1 : 1) * readUniqueBaseRange(rna, 8),
                                (readBase(rna) >= 2 ? -1 : 1) * readUniqueBaseRange(rna, 8)};
                Gene externalFactor = Gene(externalFactorType,
                                           sign, modifier, embedding, extra);
                lifeForm->grn.factors.push_back(externalFactor);
            }

            // Gene effectors (outputs of grn)
            if (type == 1) {
                int subType = (int)(readUniqueBaseRange(rna, 3) * 64) % 7;
                Effector effector = Effector(static_cast<Effector::EffectorType>(subType),
                                             sign, modifier, embedding);
                lifeForm->grn.effectors.push_back(effector);
            }

            // Regulatory units (nodes in grn)
            // Promoters (take in factors and produce activity levels excitatory/inhibitory additive/multiplicative)
            if (type == 2) {
                int additive = readBase(rna) >= 1;
                Promoter::PromoterType promoterType = additive
                                                      ? Promoter::PromoterType::Additive
                                                      : Promoter::PromoterType::Multiplicative;
                Promoter promoter = Promoter(promoterType, sign, modifier, embedding);
                if (readingPromoters) {
                    regulatoryUnit.promoters.push_back(lifeForm->grn.promoters.size());
                } else {
                    lifeForm->grn.regulatoryUnits.push_back(regulatoryUnit);
                    regulatoryUnit = RegulatoryUnit();
                    readingPromoters = true;
                }
                lifeForm->grn.promoters.push_back(promoter);
            }

            // Genes - internal product, external product, receptor
            if (type == 3 || type == 4 || type == 5) {
                Gene::FactorType geneTypes[] = {
                  Gene::FactorType::ExternalProduct,
                 Gene::FactorType::InternalProduct,
                 Gene::FactorType::Receptor,};

                Gene gene = Gene(geneTypes[type - 3], sign, modifier, embedding);

                readingPromoters = false;
                regulatoryUnit.factors.push_back(lifeForm->grn.factors.size());
                lifeForm->grn.factors.push_back(gene);
            }
        } catch (RNAExhaustedException& e) {
            // "RNA exhausted"
        }
    }

    lifeForm->grn.precomputeAffinities();
}
