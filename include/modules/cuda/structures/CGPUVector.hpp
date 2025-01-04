#ifndef CGPU_VECTOR_H
#define CGPU_VECTOR_H

#include "vector"

template<typename T>
class CGPUVector {
    T* d_data;        // Device data
    std::vector<T> h_data;  // Host mirror
    size_t size_;     // Current size
    size_t capacity_ = 0; // Current capacity

    void reallocateDevice(size_t new_capacity);

public:
    CGPUVector() : d_data(nullptr), size_(0), capacity_(0) {}
    explicit CGPUVector(size_t initial_capacity);
    explicit CGPUVector(const std::vector<T>& host_vector);
    ~CGPUVector();

    // Disable copy constructor and assignment operator
    CGPUVector(const CGPUVector&) = delete;
    CGPUVector& operator=(const CGPUVector&) = delete;

    // Move constructor and assignment operator
    CGPUVector(CGPUVector&& other) noexcept;
    CGPUVector& operator=(CGPUVector&& other) noexcept;

    void push_back(const T& value);
    void remove(int index);
    T* back();
    void pop_back();
    T& operator[](size_t index);
    void update(size_t i, T value);

    [[nodiscard]] size_t size() const { return size_; }
    [[nodiscard]] size_t capacity() const { return capacity_; }
    void resize(size_t new_size);
    void reserve(size_t new_capacity);
    void clear();

    void syncToHost();
    void syncToDevice();

    T* deviceData() { return d_data; }
    const std::vector<T>& hostData() const { return h_data; }
};

#include "../../../../src/modules/cuda/structures/CGPUVector.tpp"

#endif // GPU_VECTOR_H