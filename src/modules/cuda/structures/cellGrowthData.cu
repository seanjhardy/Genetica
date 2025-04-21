#include <modules/cuda/structures/cellGrowthData.hpp>

cellGrowthData::cellGrowthData() {
    cudaMalloc(&dividingCellIdxs, MAX_DIVISION_EVENTS * sizeof(size_t));
    cudaMalloc(&numDivisions, sizeof(int));
}

void cellGrowthData::setEnabled(bool enabled) {
    this->enabled = enabled;
}

void cellGrowthData::reset() {
    if (!enabled) return;
    cudaMemset(numDivisions, 0, sizeof(int));
}

__device__ bool cellGrowthData::push(size_t cellIdx) {
    if (!enabled) return false;
    int eventIdx = atomicAdd(numDivisions, 1);
    if (eventIdx < MAX_DIVISION_EVENTS) {
        dividingCellIdxs[eventIdx] = cellIdx;
        return true;
    }
    return false;
}

int cellGrowthData::getNumDivisions() const {
    if (!enabled) return 0;
    int count = 0;
    cudaMemcpy(&count, numDivisions, sizeof(int), cudaMemcpyDeviceToHost);
    return count;
}

// Retrieve dividing cell indices (only when needed)
void cellGrowthData::getDividingCellIndices(size_t* hostIndices, int count) {
    cudaMemcpy(hostIndices, dividingCellIdxs, count * sizeof(size_t), cudaMemcpyDeviceToHost);
}
