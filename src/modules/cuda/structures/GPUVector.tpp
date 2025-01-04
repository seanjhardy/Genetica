#pragma once

template<typename T>
GPUVector<T>::~GPUVector() {
    if (this->data_) {
        cudaFree(this->data_);
        cudaFree(overflow_);
        this->size_ = 0;
        this->capacity_ = 0;
        overflow_size_ = 0;
        overflow_capacity_ = 0;
    }
}

template<typename T>
size_t GPUVector<T>::push(const T& value) {
    // If we have empty spaces in the middle of the vector, add the value there
    if (overflow_size_ > 0) {
        int* index;
        cudaMalloc(&index, sizeof(int));
        cudaMemcpy(index, overflow_ + overflow_size_ - 1, sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->data_ + *index, &value, sizeof(T), cudaMemcpyDeviceToDevice);
        --overflow_size_;
        return *index;
    }
    if (this->size_ == this->capacity_) {
        size_t new_capacity = this->capacity_ == 0 ? 1 : this->capacity_ * 2;
        this->reallocateDevice(new_capacity);
    }
    cudaMemcpy(this->data_ + this->size_, &value, sizeof(T), cudaMemcpyDeviceToDevice);
    ++this->size_;
    return this->size_ - 1;
}

template<typename T>
size_t GPUVector<T>::getNextIndex(){
    if (overflow_size_ > 0) {
        int* index;
        cudaMalloc(&index, sizeof(int));
        cudaMemcpy(index, overflow_ + overflow_size_ - 1, sizeof(int), cudaMemcpyDeviceToDevice);
        return *index;
    }
    return this->size_;
}


template<typename T>
void GPUVector<T>::remove(size_t index) {
    if (overflow_size_ == overflow_capacity_) {
        size_t new_overflow_capacity_ = overflow_capacity_ == 0 ? 1 : overflow_capacity_ * 2;
        size_t* new_overflow;
        cudaMalloc(&new_overflow, new_overflow_capacity_ * sizeof(int));
        cudaMemcpy(new_overflow, overflow_, overflow_capacity_ * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaFree(overflow_);
        overflow_ = new_overflow;
        overflow_capacity_ = new_overflow_capacity_;
    }
    cudaMemcpy(overflow_ + overflow_size_, &index, sizeof(int), cudaMemcpyDeviceToDevice);
}

template<typename T>
void GPUVector<T>::update(size_t i, T value) {
    cudaMemcpy(this->data_ + i, &value, sizeof(T), cudaMemcpyDeviceToDevice);
}
