#include "geneticAlgorithm/sequencer.hpp"
#include <geneticAlgorithm/systems/morphology/sequencer.hpp>
#include <simulator/simulator.hpp>

void sequence(LifeForm& lifeForm, const Genome& genome, const float2& pos) {
    // Read hox genes
    sequenceGRN(lifeForm, genome);

    // Create head cell
    float size = Random::random(20) + 0.05;
    auto head = Cell(lifeForm.idx, pos, size);
    head.rotation = Random::random(0.0f, M_PI_2);

    // Add initial products to cell
    size_t numProducts = lifeForm.grn.factors.size();
    std::vector<float> products(numProducts);
    for (int i = 0; i < numProducts; ++i) {
        products[i] = 0.0f;
    }
    head.products = StaticGPUVector(products);
    head.idx = Simulator::get().getEnv().nextCellIdx();
    Simulator::get().getEnv().addCell(head);
    lifeForm.cells.push(head.idx);
}
