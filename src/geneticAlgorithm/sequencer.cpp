#include "geneticAlgorithm/sequencer.hpp"
#include <modules/utils/print.hpp>
#include <geneticAlgorithm/systems/morphology/sequencer.hpp>
#include <geneticAlgorithm/genome.hpp>

void sequence(LifeForm* lifeForm, const Genome& genome) {
    // Read hox genes
    GeneRegulatoryNetwork grn = sequenceGRN(genome);
    lifeForm->grn = grn;
}