#ifndef STATIC_GPU_VECTOR_H
#define STATIC_GPU_VECTOR_H

#include <vector>
#include <OpenCL/opencl.h>
#include "../OpenCLManager.hpp"

template<typename T>
class StaticGPUVector {
protected:
    cl_mem data_ = nullptr;        // OpenCL buffer
    size_t size_ = 0;             // Current size
    size_t capacity_ = 0;         // Current capacity

    void reallocateDevice(size_t new_capacity);
public:
    // Constructors
    StaticGPUVector() = default;
    explicit StaticGPUVector(size_t initial_capacity);
    explicit StaticGPUVector(const std::vector<T>& host_vector);

    // Destructor
    ~StaticGPUVector() { destroy(); }

    // Array access operators - syntactic sugar for getItem/setItem
    T operator[](size_t index) const;

    // Direct access methods
    T getItem(size_t index) const;
    void setItem(size_t index, const T& value);

    // Buffer access for kernel arguments
    cl_mem getBuffer() const { return data_; }
    cl_mem* getBufferPtr() const { return &data_; }

    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    StaticGPUVector copy() const;
    std::vector<T> toHost() const;
    T itemToHost(size_t index) const;
    void resize(size_t new_size);

    void destroy();

    // Iterator-like access (returns buffer for kernel use)
    cl_mem begin() { return data_; }
    cl_mem end() { return data_; } // OpenCL buffers don't have end concept
};

#include "../../../../src/modules/gpu/structures/staticGPUVector.tpp"

#endif // GPU_VECTOR_H