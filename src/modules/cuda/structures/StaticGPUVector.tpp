#pragma once
#include <modules/utils/print.hpp>
#include <modules/cuda/logging.hpp>

template<typename T>
StaticGPUVector<T>::StaticGPUVector(size_t capacity) {
    size_ = capacity;
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
    cudaLog(cudaMemset(data_, 0, h_data.size() * sizeof(T)));
    cudaLog(cudaMemcpy(data_, h_data.data(), h_data.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void StaticGPUVector<T>::reallocateDevice(size_t new_capacity) {
    if (new_capacity == 0) {
        if (data_ != nullptr) {
            cudaLog(cudaFree(data_));
            data_ = nullptr;
        }
        size_ = 0;
        capacity_ = 0;  // Changed from 1 to 0
        return;
    }

    // Allocate new buffer
    T* new_data = nullptr;
    cudaLog(cudaMalloc(&new_data, new_capacity * sizeof(T)));

    // Only zero the NEW portion of memory
    if (size_ < new_capacity) {
        size_t new_bytes = (new_capacity - size_) * sizeof(T);
        cudaLog(cudaMemset(new_data + size_, 0, new_bytes));
    }

    // Copy old data if it exists
    if (data_ != nullptr && size_ > 0) {
        cudaLog(cudaMemcpy(new_data, data_, size_ * sizeof(T), cudaMemcpyDeviceToDevice));
        cudaLog(cudaFree(data_));
    }

    data_ = new_data;
    capacity_ = new_capacity;
}

template<typename T>
void StaticGPUVector<T>::destroy() {
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
T StaticGPUVector<T>::itemToHost(size_t index) const {
    T host_data;
    if (index < size_) {
        cudaLog(cudaMemcpy(&host_data, data_ + index, sizeof(T), cudaMemcpyDeviceToHost));
    }
    return host_data;
}

template<typename T>
T& StaticGPUVector<T>::operator[](size_t index) {
    if (data_ == nullptr) {
        printf("Warning: Accessing null data pointer\n");
    }
    return data_[index];
}

template<typename T>
T& StaticGPUVector<T>::operator[](size_t index) const {
    if (data_ == nullptr) {
        printf("Warning: Accessing null data pointer\n");
    }
    return data_[index];
}

template<typename T>
void StaticGPUVector<T>::resize(size_t new_size) {
    if (new_size > capacity_) {
        reallocateDevice(new_size);
    }
    size_ = new_size;
}
