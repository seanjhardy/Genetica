#include "geneticAlgorithm/sequencer.hpp"
#include <geneticAlgorithm/systems/morphology/sequencer.hpp>
#include <simulator/simulator.hpp>

void sequence(LifeForm& lifeForm, const float2& pos) {
    // Read hox genes
    sequenceGRN(lifeForm);
    //TODO: Sequence connectome

    // Create initial cell
    float size = Random::random(20) + 0.05;
    float angle = Random::random(0.0f, M_PI_2);
    Point point = Point(lifeForm.idx, pos.x, pos.y, size * 0.5f, angle);
    auto cell = Cell(lifeForm.idx, point);
    cell.energy = 100.0f;
    cell.targetRadius = point.radius;

    // Add initial products to cell
    size_t numProducts = lifeForm.grn.factors.size();
    std::vector<float> products(numProducts);
    for (int i = 0; i < numProducts; ++i) {
        products[i] = 0.0f;
    }
    //Random cuda function to log errors
    cell.products = StaticGPUVector(products);
    cell.idx = Simulator::get().getEnv().nextCellIdx();
    Simulator::get().getEnv().addCell(cell);

    /*lifeForm.cellIdxs.push_back(cell.idx);
    lifeForm.grn.cellDistances = StaticGPUVector<
        float>((lifeForm.cellIdxs.size() * (lifeForm.cellIdxs.size() - 1)) / 2);*/
}
