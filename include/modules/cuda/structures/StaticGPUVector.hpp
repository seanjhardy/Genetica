#ifndef STATIC_GPU_VECTOR_H
#define STATIC_GPU_VECTOR_H

#include <vector>

template<typename T>
class StaticGPUVector {
protected:
    T* data_ = nullptr;        // Device data
    size_t size_ = 0;     // Current size
    size_t capacity_ = 0; // Current capacity

    __host__ __device__ void reallocateDevice(size_t new_capacity);
public:
    // Constructors
    __host__ __device__ StaticGPUVector() = default;
    __host__ __device__ explicit StaticGPUVector(size_t initial_capacity);
    __host__ __device__ explicit StaticGPUVector(const std::vector<T>& host_vector);

    __host__ __device__ T& operator[](size_t index);
    __host__ __device__ T& operator[](size_t index) const;
    __host__ __device__ T* operator+(size_t index) { return data_ + index; }
    __host__ __device__ T* operator+(size_t index) const { return data_ + index; }

    __host__ __device__ [[nodiscard]] size_t size() const { return size_; }
    __host__ __device__ [[nodiscard]] size_t capacity() const { return capacity_; }
    __host__ __device__ void resize(size_t new_size);

    __host__ __device__ T* data() { return data_; }
    __host__ __device__ void destroy();
};

#include "../../../../src/modules/cuda/structures/StaticGPUVector.tpp"

#endif // GPU_VECTOR_H