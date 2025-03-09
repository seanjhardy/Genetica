#ifndef CGPU_VECTOR_H
#define CGPU_VECTOR_H

#include "vector"
#include "GPUVector.hpp"

template<typename T>
class CGPUVector: public GPUVector<T> {
    std::vector<T> h_data;

public:

    explicit CGPUVector(size_t initial_capacity);
    explicit CGPUVector(const std::vector<T>& host_vector);

    void push(const T& value);
    void remove(int index);
    T& operator[](size_t index);

    [[nodiscard]] size_t size() const { return this->size_; }
    [[nodiscard]] size_t capacity() const { return this->capacity_; }
    void clear();

    T* deviceData() { return this->data_; }
    std::vector<T>& hostData() { return h_data; }
};

#include "../../../../src/modules/cuda/structures/CGPUVector.tpp"

#endif // GPU_VECTOR_H