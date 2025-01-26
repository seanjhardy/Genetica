#include <modules/cuda/updateCells.hpp>
#include <modules/utils/structures/DynamicStableVector.hpp>

__global__ void updateCell(GPUVector<Cell> cells, GPUVector<Point> points) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cells.size()) return;

    Cell &cell = cells[index];
    const Point &point = points[cell.pointIdx];

    /*const float cellEnergyCost = M_PI * point.radius * point.radius + cell.energyUse;
    atomicAdd(&lfUpdateData[cell.lifeFormIdx].energyChange, -cellEnergyCost);
    cell.energyUse = 0.0f;

    // Send new cell data to life form
    if (cell.dividing && !lfUpdateData[cell.lifeFormIdx].cellAdded) {
        lfUpdateData[cell.lifeFormIdx].newCell = LifeForm::LfUpdateData::NEW_CELL(
         cell.idx,
            cell.pointIdx,
            point.getPos(),
            point.radius,
            cell.rotation,
            cell.divisionRotation,
            cell.generation + 1,
            cell.hue,
            cell.saturation,
            cell.luminosity);
        lfUpdateData[cell.lifeFormIdx].cellAdded = true;
        cell.dividing = false;
    };*/
}

void updateCells(DynamicStableVector<LifeForm>& lifeForms,
                 const GPUVector<Cell>& cells,
                 const GPUVector<Point>& points) {
    int threadsPerBlock = 512;
    int numCellBlocks((cells.size() + threadsPerBlock - 1) / threadsPerBlock);
    if (cells.size() == 0) {
        return;
    }

    auto lfUpdateData = StaticGPUVector<LifeForm::LfUpdateData>(lifeForms.size());
    updateCell<<<numCellBlocks, threadsPerBlock>>>(cells, points);
    std::vector<LifeForm::LfUpdateData> lfUpdateDataHost = lfUpdateData.toHost();

    for (int i = 0; i < lifeForms.size(); i++) {
        if (lifeForms.freeList_.contains(i)) continue;
        lifeForms[i].energy += lfUpdateDataHost[i].energyChange;
        print(lfUpdateDataHost[i].cellAdded, lfUpdateDataHost[i].newCell.pos, lfUpdateDataHost[i].newCell.radius,
                lfUpdateDataHost[i].newCell.rotation, lfUpdateDataHost[i].newCell.divisionRotation,
                lfUpdateDataHost[i].newCell.generation, lfUpdateDataHost[i].newCell.hue, lfUpdateDataHost[i].newCell.saturation,
                lfUpdateDataHost[i].newCell.luminosity);
        if (lfUpdateDataHost[i].cellAdded) {
            //lifeForms[i].addCell(lfUpdateDataHost[i].newCell);
        }
        //lfUpdateDataHost[i].newCell.products.destroy();
    };
    lfUpdateData.destroy();
}