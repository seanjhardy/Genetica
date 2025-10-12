#ifndef GPU_VECTOR_H
#define GPU_VECTOR_H

#include <vector>
#include "staticGPUVector.hpp"

template<typename T>
class GPUVector : public StaticGPUVector<T> {
    cl_mem free_list_ = nullptr;        // OpenCL buffer for free list
    size_t free_size_ = 0;
    size_t free_capacity_ = 0;

    void reallocateFreeList(size_t new_capacity);

public:
    GPUVector() = default;
    explicit GPUVector(const int capacity) : StaticGPUVector<T>(capacity) {}
    explicit GPUVector(const std::vector<T>& hostData);

    // Destructor
    ~GPUVector() { destroy(); }

    void destroy();

    size_t push(const T& value);
    void remove(size_t index);
    void update(size_t i, const T& value);
    size_t getNextIndex();

    // OpenCL-specific methods
    cl_mem getFreeListBuffer() const { return free_list_; }
    size_t getFreeSize() const { return free_size_; }
};

#include "../../../../src/modules/gpu/structures/GPUVector.tpp"

#endif // GPU_VECTOR_H