#ifndef GPU_VECTOR_H
#define GPU_VECTOR_H

#include <vector>
#include "StaticGPUVector.hpp"

template<typename T>
class GPUVector : public StaticGPUVector<T> {
    size_t* overflow_ = nullptr;
    size_t overflow_size_ = 0;
    size_t overflow_capacity_ = 0;

public:
    __host__ __device__ GPUVector() = default;
    __host__ __device__ explicit GPUVector(const std::vector<T>& hostData);
    __host__ __device__ void destroy();

    __host__ __device__ size_t push(const T& value);
    __host__ __device__ void remove(size_t index);
    __host__ __device__ void update(size_t i, T value);
    __host__ __device__ size_t getNextIndex();
};

#include "../../../../src/modules/cuda/structures/GPUVector.tpp"

#endif // GPU_VECTOR_H