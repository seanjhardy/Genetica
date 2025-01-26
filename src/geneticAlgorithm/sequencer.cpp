#include "geneticAlgorithm/sequencer.hpp"
#include <geneticAlgorithm/systems/morphology/sequencer.hpp>
#include <simulator/simulator.hpp>

void sequence(LifeForm& lifeForm, const Genome& genome, const float2& pos) {
    // Read hox genes
    //sequenceGRN(lifeForm, genome);

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
    head.idx = Simulator::get().getEnv().addCell(head);
    lifeForm.cells.push(head.idx);

    /*int size2 = Random::random(20) + 0.05;
    auto cell2 = Cell(lifeForm.idx, nullptr, pos + make_float2(0.5, 0.5), size2);
    int cell2Index = Simulator::get().getEnv().addCell(cell2);
    lifeForm.cells.push(cell2Index);

    auto cellLink = CellLink(lifeForm.idx, headIndex, cell2Index, head.pointIdx, cell2.pointIdx, 30.0f);
    Simulator::get().getEnv().addCellLink(cellLink);*/
}
