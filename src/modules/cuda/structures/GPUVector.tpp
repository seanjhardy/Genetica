#pragma once

#include <geneticAlgorithm/cellParts/cell.hpp>
#include <modules/cuda/logging.hpp>

template <typename T>
GPUVector<T>::GPUVector(const std::vector<T>& hostData): StaticGPUVector<T>(hostData) {
    free_size_ = 0;
    free_capacity_ = 0;
}

template <typename T>
void GPUVector<T>::destroy() {
    StaticGPUVector<T>::destroy();
    if (free_list_ != nullptr) {
        cudaLog(cudaFree(free_list_));
        free_list_ = nullptr;
    }
    free_size_ = 0;
    free_capacity_ = 0;
}


template <typename T>
__host__ __device__ size_t GPUVector<T>::push(const T& value) {
    // If we have empty spaces in the middle of the vector, add the value there
    if (free_size_ > 0) {
        int hostIndex;
        cudaLog(cudaMemcpy(&hostIndex, free_list_ + free_size_ - 1, sizeof(int), cudaMemcpyDeviceToHost));
        cudaLog(cudaMemcpy(this->data_ + hostIndex, &value, sizeof(T), cudaMemcpyHostToDevice));
        --free_size_;
        return hostIndex;
    }
    if (this->size_ == this->capacity_) {
        size_t new_capacity = this->capacity_ == 0 ? 1 : this->capacity_ * 2;
        this->reallocateDevice(new_capacity);
    }
    cudaLog(cudaMemcpy(this->data_ + this->size_, &value, sizeof(T), cudaMemcpyHostToDevice));

    ++this->size_;
    return this->size_ - 1;
}

template <typename T>
__host__ __device__ size_t GPUVector<T>::getNextIndex() {
    if (free_size_ > 0) {
        int hostIndex;
        cudaLog(cudaMemcpy(&hostIndex, free_list_ + free_size_ - 1, sizeof(int), cudaMemcpyDeviceToHost));
        return hostIndex;
    }
    return this->size_;
}


template <typename T>
__host__ __device__ void GPUVector<T>::remove(size_t index) {
    if (free_size_ == free_capacity_) {
        size_t new_overflow_capacity_ = free_capacity_ == 0 ? 1 : free_capacity_ * 2;
        size_t* new_overflow;
        cudaLog(cudaMalloc(&new_overflow, new_overflow_capacity_ * sizeof(int)));
        cudaLog(cudaMemcpy(new_overflow, free_list_, free_capacity_ * sizeof(int), cudaMemcpyDeviceToDevice));
        if (free_list_ != nullptr) {
            cudaLog(cudaFree(free_list_));
        }
        free_list_ = new_overflow;
        free_capacity_ = new_overflow_capacity_;
    }
    cudaLog(cudaMemcpy(free_list_ + free_size_, &index, sizeof(int), cudaMemcpyDeviceToDevice));
    free_size_++;
}

template <typename T>
__host__ __device__ void GPUVector<T>::update(size_t i, T value) {
    cudaLog(cudaMemcpy(this->data_ + i, &value, sizeof(T), cudaMemcpyDeviceToDevice));
}
