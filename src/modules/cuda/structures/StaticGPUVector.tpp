#pragma once
#include <modules/utils/print.hpp>
#include <modules/cuda/logging.hpp>

template<typename T>
StaticGPUVector<T>::StaticGPUVector(size_t capacity) {
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
    if (data_ != nullptr) {
        cudaLog(cudaFree(data_));
        data_ = nullptr;
        size_ = 0;
        capacity_ = 0;
    }
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
