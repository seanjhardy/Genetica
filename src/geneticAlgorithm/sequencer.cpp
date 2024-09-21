#include "geneticAlgorithm/sequencer.hpp"
#include <geneticAlgorithm/systems/morphology/sequencer.hpp>

void sequence(LifeForm* lifeForm, const Genome& genome) {
    // Read hox genes
    sequenceGRN(lifeForm, genome);

    // Create head cell
    Cell* head = new Cell(lifeForm, nullptr, {lifeForm->pos.x, lifeForm->pos.y}, 5.0f);
    // Add initial products to cell
    for (auto& product: lifeForm->grn.factors) {
        head->products[&product] = 0.0f;
    }
    lifeForm->head = head;
    lifeForm->addCell(head);
}