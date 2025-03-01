#include "geneticAlgorithm/sequencer.hpp"
#include <geneticAlgorithm/systems/morphology/sequencer.hpp>
#include <simulator/simulator.hpp>

void sequence(LifeForm& lifeForm, const Genome& genome, const float2& pos) {
    // Read hox genes
    //sequenceGRN(lifeForm, genome);

    // Create head cell
    float size = Random::random(20) + 0.05;
    auto cell = Cell(lifeForm.idx, pos, size);
    cell.rotation = Random::random(0.0f, M_PI_2);

    // Add initial products to cell
    size_t numProducts = lifeForm.grn.factors.size();
    std::vector<float> products(1);
    /*for (int i = 0; i < numProducts; ++i) {
        products[i] = 0.0f;
    }*/
    cell.products = StaticGPUVector(products);
    cell.idx = Simulator::get().getEnv().nextCellIdx();
    Simulator::get().getEnv().addCell(cell);
    printf("Pushing cell: products=%p size=%llu capacity=%llu\n",
           cell.products.data(), cell.products.size(), cell.products.capacity());

    lifeForm.cells.push(cell.idx);
    printf("Base cells address: %p\n", Simulator::get().getEnv().getCells().data());
    printf("Cell idx: %llu\n", cell.idx);
    printf("Calculated pointer: %p\n", Simulator::get().getEnv().getCells().data() + cell.idx);
    lifeForm.cellPointers.push(Simulator::get().getEnv().getCells().data() + cell.idx);
    Cell verify_cell;
    cudaMemcpy(&verify_cell, Simulator::get().getEnv().getCells().data() + Simulator::get().getEnv().getCells().size() - 1, sizeof(Cell), cudaMemcpyDeviceToHost);
    printf("After push: products=%p size=%llu capacity=%llu\n",
           verify_cell.products.data(), verify_cell.products.size(), verify_cell.products.capacity());
}
