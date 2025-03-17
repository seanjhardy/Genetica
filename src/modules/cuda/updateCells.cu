#include <modules/cuda/updateCells.hpp>
#include <modules/utils/structures/DynamicStableVector.hpp>
#include <simulator/simulator.hpp>

struct LfUpdateData {
    int dividing = 0;
    float energyChange = 0;
    size_t motherIdx = 0;
};
#

__global__ void updateCell(GPUVector<Cell> cells, GPUVector<Point> points, StaticGPUVector<LfUpdateData> lfUpdateData, size_t step) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cells.size()) return;
    Cell &cell = cells[index];

    if (cell.pointIdx > points.size()) return;
    Point &point = points[cell.pointIdx];
    auto* lfData = &lfUpdateData[cell.lifeFormIdx];
    float cellEnergyCost = 0.00001f * M_PI * point.radius * point.radius + cell.energyUse;
    if (abs(point.radius - cell.targetRadius) > 1.0f) {
        cellEnergyCost += 0;
        point.radius = sqrt((M_PI * point.radius * point.radius + 1.0f) / M_PI);
    }
    atomicAdd(&lfUpdateData[cell.lifeFormIdx].energyChange, -cellEnergyCost);
    cell.energyUse = 0.0f;

    if (cell.dividing && (step - cell.lastDivideTime) > 10 && point.radius >= cell.targetRadius * 0.98) {
        if (atomicCAS(&lfData->dividing, 0, 1) == 0) {
            cell.dividing = false;
            lfData->motherIdx = cell.idx;

            cell.numDivisions += 1;
            cell.lastDivideTime = step;
            //point.radius *= 0.7f;
            printf("lifeFormIdx: %llu, motherIdx: %llu\n", cell.lifeFormIdx, cell.idx);
        }
   };
}

void updateCells(DynamicStableVector<LifeForm>& lifeForms,
                 const GPUVector<Cell>& cells,
                 const GPUVector<Point>& points) {
    // Only run every 20 simulation steps
    if (Simulator::get().getStep() % 10 != 0) {
        return;
    }

    int threadsPerBlock = 512;
    int numCellBlocks((cells.size() + threadsPerBlock - 1) / threadsPerBlock);

    if (cells.size() == 0) {
        return;
    }

    auto lfUpdateData = StaticGPUVector<LfUpdateData>(lifeForms.size());
    updateCell<<<numCellBlocks, threadsPerBlock>>>(cells, points, lfUpdateData, Simulator::get().getStep());
    std::vector<LfUpdateData> lfUpdateDataHost = lfUpdateData.toHost();

    for (int i = 0; i < lifeForms.size(); i++) {
        if (lifeForms.freeList_.contains(i)) continue;
        lifeForms[i].energy += lfUpdateDataHost[i].energyChange;
        if (lfUpdateDataHost[i].dividing) {
            Cell mother = cells.itemToHost(lfUpdateDataHost[i].motherIdx);
            Point p = points.itemToHost(mother.pointIdx);
            lifeForms[i].addCell(lfUpdateDataHost[i].motherIdx, mother, p);
        }
    };
    lfUpdateData.destroy();
}