#ifndef STATIC_GPU_VECTOR_H
#define STATIC_GPU_VECTOR_H

#include <vector>
#include <cuda_runtime.h>

template<typename T>
class StaticGPUVector {
protected:
    T* data_ = nullptr;        // Device data
    size_t size_ = 0;     // Current size
    size_t capacity_ = 0; // Current capacity

    __host__ __device__ void reallocateDevice(size_t new_capacity);
public:
    // Constructors
    __host__ StaticGPUVector() = default;
    __host__ explicit StaticGPUVector(size_t initial_capacity);
    __host__ explicit StaticGPUVector(const std::vector<T>& host_vector);

    __host__ __device__ T& operator[](size_t index);
    __host__ __device__ T& operator[](size_t index) const;
    __host__ __device__ T* operator+(size_t index) { return data_ + index; }
    __host__ __device__ T* operator+(size_t index) const { return data_ + index; }

    __host__ __device__ size_t size() const { return size_; }
    __host__ __device__ size_t capacity() const { return capacity_; }
    __host__ __device__ StaticGPUVector copy() const;
    __host__ std::vector<T> toHost() const;
    __host__ T itemToHost(size_t index) const;
    __host__ __device__ void resize(size_t new_size);

    __host__ __device__ T* data() { return data_; }
    __host__ void destroy();

    __host__ __device__ T* begin() { return data_; }
    __host__ __device__ T* end() { return data_ + size_; }
};

#include "../../../../src/modules/cuda/structures/staticGPUVector.tpp"

#endif // GPU_VECTOR_H