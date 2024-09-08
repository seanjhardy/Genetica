#include "geneticAlgorithm/sequencer.hpp"
#include <modules/utils/print.hpp>
#include <geneticAlgorithm/systems/morphology/sequencer.hpp>

void sequence(LifeForm* lifeForm, const Genome& genome) {
    // Read hox genes
    GeneRegulatoryNetwork grn = sequenceGRN(genome);
    grn.lifeForm = lifeForm;
    lifeForm->grn = grn;
}