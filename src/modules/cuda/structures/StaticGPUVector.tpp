#pragma once

template<typename T>
StaticGPUVector<T>::StaticGPUVector(size_t capacity) : size_(0), capacity_(capacity) {
    reallocateDevice(capacity);
}

template<typename T>
StaticGPUVector<T>::StaticGPUVector(const std::vector<T>& h_data){
    size_ = h_data.size();
    capacity_ = h_data.size();
    cudaFree(data_);
    cudaMalloc(&data_, h_data.size() * sizeof(T));
    cudaMemcpy(data_, h_data.data(), h_data.size() * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
void StaticGPUVector<T>::reallocateDevice(size_t new_capacity) {
    T* d_data_old = data_;
    cudaMalloc(&data_, new_capacity * sizeof(T));
    cudaMemcpy(data_, d_data_old, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaFree(d_data_old);
    capacity_ = new_capacity;
}

template<typename T>
StaticGPUVector<T>::~StaticGPUVector() {
    if (data_) {
        cudaFree(data_);
        size_ = 0;
        capacity_ = 0;
    }
}

template <typename T>
__host__ StaticGPUVector<T>::StaticGPUVector(const StaticGPUVector& other)
  : size_(other.size_), capacity_(other.capacity_) {
    if (capacity_ > 0) {
        reallocateDevice(capacity_);
    } else {
        data_ = nullptr;
    }
}

template <typename T>
__host__ StaticGPUVector<T>& StaticGPUVector<T>::operator=(const StaticGPUVector& other) {
    if (this != &other) {
        if (data_) {
            cudaFree(data_);
        }
        size_ = other.size_;
        capacity_ = other.capacity_;
        if (capacity_ > 0) {
            reallocateDevice(capacity_);
        } else {
            data_ = nullptr;
        }
    }
    return *this;
}

template<typename T>
StaticGPUVector<T>& StaticGPUVector<T>::operator=(StaticGPUVector<T>&& other) noexcept {
    if (this != &other) {
        if (data_) {
            cudaFree(data_);
        }
        data_ = other.data_;
        size_ = other.size_;
        capacity_ = other.capacity_;
        other.size_ = 0;
        other.capacity_ = 0;
    }
    return *this;
}

template<typename T>
StaticGPUVector<T>::StaticGPUVector(StaticGPUVector<T>&& other) noexcept
  : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
}

template<typename T>
T& StaticGPUVector<T>::operator[](size_t index) {
    return data_[index];
}

template<typename T>
T& StaticGPUVector<T>::operator[](size_t index) const{
    return data_[index];
}

template <typename T>
void StaticGPUVector<T>::clear() {
    size_ = 0;
}

template<typename T>
void StaticGPUVector<T>::resize(size_t new_size) {
    if (new_size > capacity_) {
        reallocateDevice(new_size);
    }
    size_ = new_size;
}
