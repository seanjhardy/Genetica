#include "geneticAlgorithm/sequencer.hpp"
#include <geneticAlgorithm/systems/morphology/sequencer.hpp>
#include <modules/utils/GPU/GPUUtils.hpp>

void sequence(LifeForm* lifeForm, const Genome& genome) {
    // Read hox genes
    sequenceGRN(lifeForm, genome);

    // Create head cell
    Cell* head = new Cell(lifeForm, nullptr, {lifeForm->pos.x, lifeForm->pos.y}, 5.0f);
    head->rotation = Random::random(0, M_PI_2);

    // Add initial products to cell
    int numProducts = lifeForm->grn.numFactors;
    std::vector<float> products(numProducts);
    for (int i = 0; i < numProducts; ++i) {
        products[i] = 0.0f;
    }
    saveGPUArray(head->products, products);

    lifeForm->head = head;
    lifeForm->addCell(head);
}
