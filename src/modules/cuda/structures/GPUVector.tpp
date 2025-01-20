#pragma once
#include <modules/cuda/logging.hpp>

template<typename T>
GPUVector<T>::GPUVector(const std::vector<T>& hostData): StaticGPUVector<T>(hostData) {
    overflow_size_ = 0;
    overflow_capacity_ = 0;
}

template<typename T>
void GPUVector<T>::destroy() {
    StaticGPUVector<T>::destroy();
    if (overflow_ != nullptr) {
        cudaLog(cudaFree(overflow_));
        overflow_ = nullptr;
    }
    overflow_size_ = 0;
    overflow_capacity_ = 0;
}

template<typename T>
size_t GPUVector<T>::push(const T& value) {
    // If we have empty spaces in the middle of the vector, add the value there
    if (overflow_size_ > 0) {
        int hostIndex;
        cudaLog(cudaMemcpy(&hostIndex, overflow_ + overflow_size_ - 1, sizeof(int), cudaMemcpyDeviceToHost));
        cudaLog(cudaMemcpy(this->data_ + hostIndex, &value, sizeof(T), cudaMemcpyHostToDevice));
        --overflow_size_;
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

template<typename T>
size_t GPUVector<T>::getNextIndex(){
    if (overflow_size_ > 0) {
        int hostIndex;
        cudaLog(cudaMemcpy(&hostIndex, overflow_ + overflow_size_ - 1, sizeof(int), cudaMemcpyDeviceToHost));
        return hostIndex;
    }
    return this->size_;
}


template<typename T>
void GPUVector<T>::remove(size_t index) {
    if (overflow_size_ == overflow_capacity_) {
        size_t new_overflow_capacity_ = overflow_capacity_ == 0 ? 1 : overflow_capacity_ * 2;
        size_t* new_overflow;
        cudaLog(cudaMalloc(&new_overflow, new_overflow_capacity_ * sizeof(int)));
        cudaLog(cudaMemcpy(new_overflow, overflow_, overflow_capacity_ * sizeof(int), cudaMemcpyDeviceToDevice));
        if (overflow_ != nullptr) {
            cudaLog(cudaFree(overflow_));
        }
        overflow_ = new_overflow;
        overflow_capacity_ = new_overflow_capacity_;
    }
    cudaLog(cudaMemcpy(overflow_ + overflow_size_, &index, sizeof(int), cudaMemcpyDeviceToDevice));
    overflow_size_++;
}

template<typename T>
void GPUVector<T>::update(size_t i, T value) {
    cudaLog(cudaMemcpy(this->data_ + i, &value, sizeof(T), cudaMemcpyDeviceToDevice));
}
