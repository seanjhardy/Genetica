#include <modules/cuda/updateCells.hpp>
#include <modules/utils/structures/DynamicStableVector.hpp>

struct LfUpdateData {
    bool dividing = false;
    float energyChange = 0;
    size_t motherIdx = 0;
};
#

__global__ void updateCell(GPUVector<Cell> cells, GPUVector<Point> points, StaticGPUVector<LfUpdateData> lfUpdateData) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cells.size()) return;
    Cell &cell = cells[index];
    if (cell.pointIdx > points.size()) return;
    const Point &point = points[cell.pointIdx];
    auto* lfData = lfUpdateData + cell.lifeFormIdx;

    const float cellEnergyCost = 0.00001f * M_PI * point.radius * point.radius + cell.energyUse;
    atomicAdd(&lfUpdateData[cell.lifeFormIdx].energyChange, -cellEnergyCost);
    cell.energyUse = 0.0f;

    // Send new cell data to life form
    if (!lfData->dividing && cell.dividing) {
        cell.dividing = false;
        lfData->motherIdx = cell.idx;
        lfData->dividing = true;

        cell.numDivisions += 1;
    };
}

void updateCells(DynamicStableVector<LifeForm>& lifeForms,
                 const GPUVector<Cell>& cells,
                 const GPUVector<Point>& points) {
    int threadsPerBlock = 512;
    int numCellBlocks((cells.size() + threadsPerBlock - 1) / threadsPerBlock);

    if (cells.size() == 0) {
        return;
    }

    /*auto lfUpdateData = StaticGPUVector<LfUpdateData>(lifeForms.size());
    updateCell<<<numCellBlocks, threadsPerBlock>>>(cells, points, lfUpdateData);
    std::vector<LfUpdateData> lfUpdateDataHost = lfUpdateData.toHost();

    for (int i = 0; i < lifeForms.size(); i++) {
        if (lifeForms.freeList_.contains(i)) continue;
        lifeForms[i].energy += lfUpdateDataHost[i].energyChange;

        if (lfUpdateDataHost[i].dividing && lfUpdateDataHost[i].motherIdx < cells.size()) {
            Cell mother = cells.itemToHost(lfUpdateDataHost[i].motherIdx);
            Point p = points.itemToHost(mother.pointIdx);
            lifeForms[i].addCell(lfUpdateDataHost[i].motherIdx, mother, p);
        }
    };
    lfUpdateData.destroy();*/
}