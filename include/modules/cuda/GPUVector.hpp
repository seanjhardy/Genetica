#ifndef GPU_VECTOR_H
#define GPU_VECTOR_H

#include "vector"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

template<typename T>
class GPUVector {
private:
    T* d_data;        // Device data
    std::vector<T> h_data;  // Host mirror
    size_t size_;     // Current size
    size_t capacity_ = 0; // Current capacity

    void reallocateDevice(size_t new_capacity);

public:
    GPUVector() : d_data(nullptr), size_(0), capacity_(0) {}
    explicit GPUVector(size_t initial_capacity);
    explicit GPUVector(const std::vector<T>& host_vector);
    ~GPUVector();

    // Disable copy constructor and assignment operator
    GPUVector(const GPUVector&) = delete;
    GPUVector& operator=(const GPUVector&) = delete;

    // Move constructor and assignment operator
    GPUVector(GPUVector&& other) noexcept;
    GPUVector& operator=(GPUVector&& other) noexcept;

    void push_back(const T& value);
    T* back();
    void pop_back();
    T& operator[](size_t index);

    [[nodiscard]] size_t size() const { return size_; }
    [[nodiscard]] size_t capacity() const { return capacity_; }
    void resize(size_t new_size);
    void reserve(size_t new_capacity);
    void clear();

    void syncToHost();
    void syncToDevice();

    T* deviceData() { return d_data; }
    const T* deviceData() const { return d_data; }
    const std::vector<T>& hostData() const { return h_data; }
};

#include "../../../src/modules/cuda/GPUVector.tpp"

#endif // GPU_VECTOR_H