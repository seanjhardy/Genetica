#include <modules/cuda/updateCells.hpp>

__global__ void updateCell(GPUVector<Cell>& cells, GPUVector<Point>& points) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cells.size()) return;

    Cell &cell = cells[index];
}

void updateCells(GPUVector<LifeForm>& lifeForms,
                 GPUVector<Cell>& cells,
                 GPUVector<Point>& points,
                 float dt) {
    dim3 threadsPerBlock(256, 256);
    dim3 numCellBlocks((cells.size() + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (cells.size() + threadsPerBlock.y - 1) / threadsPerBlock.y);
    updateCell<<<numCellBlocks, threadsPerBlock>>>(cells, points);
}