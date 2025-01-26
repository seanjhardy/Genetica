#pragma once
#include <modules/utils/print.hpp>
#include <modules/cuda/logging.hpp>

template<typename T>
StaticGPUVector<T>::StaticGPUVector(size_t capacity) {
    size_ = capacity;
    printf("created");
    reallocateDevice(capacity);
}

template<typename T>
StaticGPUVector<T>::StaticGPUVector(const std::vector<T>& h_data){
    size_ = h_data.size();
    capacity_ = h_data.size();
    if (data_ != nullptr) {
        cudaLog(cudaFree(data_));
        data_ = nullptr;
    }
    cudaLog(cudaMalloc(&data_, h_data.size() * sizeof(T)));
    cudaLog(cudaMemcpy(data_, h_data.data(), h_data.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void StaticGPUVector<T>::reallocateDevice(size_t new_capacity) {
    if (new_capacity == 0) {
        data_ = nullptr;
        size_ = 0;
        capacity_ = 0;
        return;
    }
    printf("capacity: %d, %d \n", capacity_, new_capacity);
    T* d_data_old = data_;
    cudaLog(cudaMalloc(&data_, new_capacity * sizeof(T)));
    if (d_data_old != nullptr) {
        cudaLog(cudaMemcpy(data_, d_data_old, size_ * sizeof(T), cudaMemcpyDeviceToDevice));
        cudaLog(cudaFree(d_data_old));
    }
    capacity_ = new_capacity;
}

template<typename T>
void StaticGPUVector<T>::destroy() {
    printf("destroy");
    if (data_ != nullptr) {
        cudaLog(cudaFree(data_));
        data_ = nullptr;
        size_ = 0;
        capacity_ = 0;
    }
}

template<typename T>
StaticGPUVector<T> StaticGPUVector<T>::copy() const {
    StaticGPUVector copy(size_);
    if (size_ > 0) {
        cudaLog(cudaMemcpy(copy.data_, data_, size_ * sizeof(T), cudaMemcpyDeviceToDevice));
    }
    return copy;
}

template<typename T>
std::vector<T> StaticGPUVector<T>::toHost() const {
    std::vector<T> host_data(size_);
    if (size_ > 0) {
        cudaLog(cudaMemcpy(host_data.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
    }
    return host_data;
}

template<typename T>
T& StaticGPUVector<T>::operator[](size_t index) {
    return data_[index];
}

template<typename T>
T& StaticGPUVector<T>::operator[](size_t index) const {
    return data_[index];
}

template<typename T>
void StaticGPUVector<T>::resize(size_t new_size) {
    if (new_size > capacity_) {
        reallocateDevice(new_size);
    }
    size_ = new_size;
}
