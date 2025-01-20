#include <modules/cuda/updateCells.hpp>

__global__ void updateCell(GPUVector<Cell> cells, GPUVector<Point> points) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cells.size()) return;

    Cell &cell = cells[index];
}

void updateCells(GPUVector<Cell>& cells,
                 GPUVector<Point>& points,
                 float dt) {
    int threadsPerBlock = 256;
    int numCellBlocks((cells.size() + threadsPerBlock - 1) / threadsPerBlock);
    if (cells.size() == 0) {
        return;
    }
    updateCell<<<numCellBlocks, threadsPerBlock>>>(cells, points);
}