#include <vector>
#include <cuda_runtime.h>

template <typename T>
void saveGPUArray(T* d_data, const std::vector<T>& h_data);

#include "../../../../src/modules/utils/GPU/GPUUtils.tpp"