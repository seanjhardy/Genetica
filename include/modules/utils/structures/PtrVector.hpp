#ifndef PTR_VECTOR_H
#define PTR_VECTOR_H

#include <vector>
#include "StaticPtrVector.hpp"

template<typename T>
class PtrVector : public StaticPtrVector<T> {
    size_t* overflow_ = nullptr;
    size_t overflow_size_ = 0;
    size_t overflow_capacity_ = 0;

public:
    PtrVector() = default;
    explicit PtrVector(const std::vector<T>& hostData);
    void clear();

    size_t push(const T& value);
    size_t push(const T&& value);
    void remove(size_t index);
    void update(size_t i, T value);
    size_t getNextIndex();

    T* begin() { return this->data_; }
    T* end() { return this->data_ + this->size_; }
    const T* begin() const { return this->data_; }
    const T* end() const { return this->data_ + this->size_; }

};

#include "../../../../src/modules/utils/structures/PtrVector.tpp"

#endif // GPU_VECTOR_H