#include "geneticAlgorithm/sequencer.hpp"
#include "modules/utils/genomeUtils.hpp"
#include <modules/utils/print.hpp>
#include <string>
#include <utility>
#include <geneticAlgorithm/systems/morphology/geneRegulatoryNetwork.hpp>
#include <modules/utils/genomeUtils.hpp>
#include <geneticAlgorithm/genome.hpp>

GeneRegulatoryNetwork sequenceGRN(const Genome& genome) {
    GeneRegulatoryNetwork grn;

    std::vector<Promoter> promoters;
    std::vector<Gene> genes;
    RegulatoryUnit regulatoryUnit;
    bool readingPromoters = true;
    for (auto [id, sequence] : genome.hoxGenes) {
        std::string rna = std::move(sequence); // Create rna copy
        try {
            int type = readBase(rna);
            bool sign = readBase(rna) >= 2;
            bool active = readBase(rna) >= 2;
            // Uniformly distributed modifiers (also allows for large jumps genetically)
            float modifier = readUniqueBaseRange(rna, 20);

            float embedding[GeneticUnit::N];
            for (float & i : embedding) {
                // spatial fidelity = (4 values * 20 per dimension) = 80
                // 80^(3 dimensions) = 512000 unique positions
                // Uniform distribution using readUniqueBaseRange
                i = readUniqueBaseRange(rna, 20);
            }

            // External factors (inputs to grn)
            if (type == 0) {
                Gene::FactorType externalFactorTypes[] = {
                  Gene::FactorType::ExternalFactorP1,
                  Gene::FactorType::ExternalFactorP2,
                  Gene::FactorType::ExternalFactorP3,
                  Gene::FactorType::Constant,
                  Gene::FactorType::Congestion,
                  Gene::FactorType::Time,
                  Gene::FactorType::Generation,
                  Gene::FactorType::Energy
                };
                int subType = readUniqueBaseRange(rna, 2) * 8;
                Gene::FactorType externalFactorType = externalFactorTypes[subType];
                Gene externalFactor = Gene(externalFactorType,
                                           active, sign, modifier, embedding);
                grn.elements.push_back(externalFactor);
            }
            // Gene effectors (outputs of grn)
            if (type == 1) {
                int subType = (int)(readUniqueBaseRange(rna, 6) * 4096) % 7;
                Effector effector = Effector(static_cast<Effector::EffectorType>(subType),
                                             active, sign, modifier, embedding);
                grn.elements.push_back(effector);
            }

            // Regulatory units (nodes in grn)
            // Promoters (take in morphogens and produce activity levels exitatory/inhibitory)
            if (type == 2) {
                int additive = readBase(rna) >= 1;
                Promoter::PromoterType promoterType = additive
                                                      ? Promoter::PromoterType::Additive
                                                      : Promoter::PromoterType::Multiplicative;
                Promoter promoter = Promoter(promoterType, active,
                                             sign, modifier, embedding);
                if (readingPromoters) {
                    regulatoryUnit.promoters.push_back(promoter);
                } else {
                    grn.regulatoryUnits.push_back(regulatoryUnit);
                    regulatoryUnit = RegulatoryUnit();
                    readingPromoters = true;
                }
                grn.elements.push_back(promoter);
            }

            // Genes - internal product, external product, receptor
            if (type == 3 || type == 4 || type == 5) {
                Gene::FactorType geneTypes[] = {
                  Gene::FactorType::ExternalProduct,
                 Gene::FactorType::InternalProduct,
                 Gene::FactorType::Receptor};

                Gene gene = Gene(geneTypes[type - 3], active,
                                 sign, modifier, embedding);

                readingPromoters = false;
                regulatoryUnit.genes.push_back(gene);
                grn.elements.push_back(gene);
            }
        } catch (RNAExhaustedException& e) {
            // "RNA exhausted"
        }
    }

    grn.precomputeAffinities();
    return grn;
}
