#include "geneticAlgorithm/sequencer.hpp"
#include <geneticAlgorithm/systems/morphology/sequencer.hpp>
#include <simulator/simulator.hpp>

void sequence(LifeForm& lifeForm, const Genome& genome, const float2& pos) {
    // Read hox genes
    sequenceGRN(lifeForm, genome);

    // Create head cell
    auto head = Cell(lifeForm, nullptr, pos, 5.0f);
    head.rotation = Random::random(0.0f, M_PI_2);

    // Add initial products to cell
    size_t numProducts = lifeForm.grn.factors.size();
    std::vector<float> products(numProducts);
    for (int i = 0; i < numProducts; ++i) {
        products[i] = 0.0f;
    }
    head.products = StaticGPUVector(products);
    int headIndex = Simulator::get().getEnv().addCell(head);
    lifeForm.cells.push(headIndex);
}
