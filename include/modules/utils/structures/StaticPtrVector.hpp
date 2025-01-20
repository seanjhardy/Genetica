#ifndef STATIC_PTR_VECTOR_H
#define STATIC_PTR_VECTOR_H

#include <vector>

template<typename T>
class StaticPtrVector {
protected:
    T* data_ = nullptr;        // Device data
    size_t size_ = 0;     // Current size
    size_t capacity_ = 0; // Current capacity

    void reallocateHost(size_t new_capacity);
public:
    // Constructors
    StaticPtrVector() = default;
    explicit StaticPtrVector(size_t initial_capacity);
    explicit StaticPtrVector(const std::vector<T>& host_vector);

    T& operator[](size_t index);
    T& operator[](size_t index) const;
    T* operator+(size_t index) { return data_ + index; }
    T* operator+(size_t index) const { return data_ + index; }

    [[nodiscard]] size_t size() const { return size_; }
    [[nodiscard]] size_t capacity() const { return capacity_; }
    void resize(size_t new_size);

    T* data() { return data_; }
    void clear();
};

#include "../../../../src/modules/utils/structures/StaticPtrVector.tpp"

#endif // GPU_VECTOR_H