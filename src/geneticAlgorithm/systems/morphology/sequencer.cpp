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
            float modifier = readBaseRange(rna, 20);

            float embedding[GeneticUnit::N];
            for (float & i : embedding) {
                // spatial fidelity = (4*20 per dimension) = 80 * 80 * 75 = 512000 unique positions
                // Most of these will lie near 0.5 as the sample approximates a normal distribution
                i = readBaseRange(rna, 20);
            }
            // Maternal factors
            if (type == 0) {
                ExternalFactor externalFactor = ExternalFactor(
                                           active, sign, modifier, embedding);
                grn.elements.push_back(externalFactor);
            }
            // Gene effectors
            if (type == 1) {
                int base = readBase(rna);
                int base2 = readBase(rna);
                auto effectorType = static_cast<Effector::EffectorType>(
                  (base + base2) % 6
                );
                Effector effector = Effector(effectorType,
                                             active, sign, modifier, embedding);
                grn.elements.push_back(effector);
            }

            // Construct regulatory units
            if (type == 2) {
                int additive = readBase(rna) >= 2;
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

            // Genes - internal product, external product
            if (type == 3 || type == 4 || type == 5) {
                Gene::ProductType geneType[] = {
                  Gene::ProductType::ExternalProduct,
                 Gene::ProductType::InternalProduct,
                 Gene::ProductType::Receptor};

                Gene gene = Gene(geneType[type - 3], active,
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
