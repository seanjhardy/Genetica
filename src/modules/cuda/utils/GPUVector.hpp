// GPUVector.h

#ifndef GPU_VECTOR_H
#define GPU_VECTOR_H

#include <vector>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

template<typename T>
class GPUVector {
private:
    T* d_data;        // Device data
    std::vector<T> h_data;  // Host mirror
    size_t size_;     // Current size
    size_t capacity_ = 0; // Current capacity

    void reallocateDevice(size_t new_capacity) {
        T* new_d_data;
        cudaMalloc(&new_d_data, new_capacity * sizeof(T));
        if (d_data) {
            cudaMemcpy(new_d_data, d_data, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaFree(d_data);
        }
        d_data = new_d_data;
        capacity_ = new_capacity;
    }

public:
    GPUVector() : d_data(nullptr), size_(0), capacity_(0) {}

    explicit GPUVector(size_t initial_capacity) : size_(0), capacity_(0) {
        reserve(initial_capacity);
        //cudaMalloc(&d_data, initial_capacity * sizeof(T));
    }

    explicit GPUVector(const std::vector<T>& host_vector) : h_data(host_vector), size_(host_vector.size()), capacity_(host_vector.size()) {
        cudaMalloc(&d_data, capacity_ * sizeof(T));
        cudaMemcpy(d_data, h_data.data(), size_ * sizeof(T), cudaMemcpyHostToDevice);
    }

    ~GPUVector() {
        if (d_data) {
            cudaFree(d_data);
        }
    }

    // Disable copy constructor and assignment operator
    GPUVector(const GPUVector&) = delete;
    GPUVector& operator=(const GPUVector&) = delete;

    // Move constructor and assignment operator
    GPUVector(GPUVector&& other) noexcept
            : d_data(other.d_data), h_data(std::move(other.h_data)), size_(other.size_), capacity_(other.capacity_) {
        other.d_data = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    GPUVector& operator=(GPUVector&& other) noexcept {
        if (this != &other) {
            if (d_data) {
                cudaFree(d_data);
            }
            d_data = other.d_data;
            h_data = std::move(other.h_data);
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.d_data = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    void push_back(const T& value) {
        if (size_ == capacity_) {
            size_t new_capacity = capacity_ == 0 ? 1 : capacity_ * 2;
            reallocateDevice(new_capacity);
        }
        h_data.push_back(value);
        cudaMemcpy(d_data + size_, &value, sizeof(T), cudaMemcpyHostToDevice);
        ++size_;
    }

    T* back() {
        return &h_data.back();
    }

    void pop_back() {
        if (size_ > 0) {
            --size_;
            h_data.pop_back();
        }
    }

    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }

    void resize(size_t new_size) {
        if (new_size > capacity_) {
            reallocateDevice(new_size);
        }
        size_ = new_size;
        h_data.resize(new_size);
    }

    void reserve(size_t new_capacity) {
        if (new_capacity > capacity_) {
            h_data.reserve(new_capacity);
            reallocateDevice(new_capacity);
        }
    }

    void clear() {
        size_ = 0;
        h_data.clear();
    }

    void syncToHost() {
        cudaMemcpy(h_data.data(), d_data, size_ * sizeof(T), cudaMemcpyDeviceToHost);
    }

    void syncToDevice() {
        cudaMemcpy(d_data, h_data.data(), size_ * sizeof(T), cudaMemcpyHostToDevice);
    }

    T* deviceData() { return d_data; }
    const T* deviceData() const { return d_data; }
    const std::vector<T>& hostData() const { return h_data; }
};

#endif // GPU_VECTOR_H