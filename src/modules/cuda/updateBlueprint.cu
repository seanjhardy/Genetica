#include <modules/cuda/updateBlueprint.hpp>
#include "../physics/constraints.cu"

__global__ void updateBlueprint(
    StaticGPUVector<Cell>& cells, StaticGPUVector<size_t>& cellIdxs,
    GPUVector<Point>& points) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cellIdxs.size()) return;

    int cellIdx = cellIdxs[index];
    Cell& cell = cells[cellIdx];
    Point& point = points[cell.blueprintPointIdx];

    point.update();
}

__global__ void updateBlueprintSquared(
    StaticGPUVector<Cell>& cells, StaticGPUVector<size_t>& cellIdxs,
    StaticGPUVector<CellLink>& cellLinks, StaticGPUVector<size_t>& cellLinkIdxs,
    GPUVector<Point>& points
) {
    size_t indexA = blockIdx.x * blockDim.x + threadIdx.x;
    size_t indexB = blockIdx.y * blockDim.y + threadIdx.y;

    if (indexA >= cellIdxs.size() || indexB >= cellIdxs.size() || indexA >= indexB) return;

    int cellIdxA = cellIdxs[indexA];
    int cellIdxB = cellIdxs[indexB];
    Cell& cellA = cells[cellIdxA];
    Cell& cellB = cells[cellIdxB];
    Point& pointA = points[cellA.blueprintPointIdx];
    Point& pointB = points[cellB.blueprintPointIdx];

    constrainMinDistance(pointA, pointB, pointA.radius + pointB.radius);
}

void updateBlueprint(
    LifeForm& lifeForm,
    GPUVector<Point>& points,
    GPUVector<Cell>& cells,
    GPUVector<CellLink>& cellLinks) {
    auto cellIdxs = static_cast<StaticGPUVector<size_t>>(lifeForm.cellIdxs);
    auto cellLinkIdxs = static_cast<StaticGPUVector<size_t>>(lifeForm.linkIdxs);
    auto staticCells = static_cast<StaticGPUVector<Cell>>(cells);
    auto staticCellLinks = static_cast<StaticGPUVector<CellLink>>(cellLinks);

    float blockSize = 256;
    int numBlocks = (cells.size() + blockSize - 1) / blockSize;
    updateBlueprint<<<numBlocks, blockSize>>>(staticCells, cellIdxs, points);

    dim3 threadsPerBlock(16, 16);
    dim3 numCellCellBlocks((cells.size() + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (cells.size() + threadsPerBlock.y - 1) / threadsPerBlock.y);
    updateBlueprintSquared<<<numCellCellBlocks, threadsPerBlock>>>(staticCells, cellIdxs, staticCellLinks, cellLinkIdxs,
                                                                   points);
}
