#pragma once

#include "../../../include/modules/gpu/structures/GPUVector.hpp"
#include <stdexcept>

template <typename T>
GPUVector<T>::GPUVector(const std::vector<T>& hostData) : StaticGPUVector<T>(hostData) {
    free_size_ = 0;
    free_capacity_ = 0;
}

template <typename T>
void GPUVector<T>::destroy() {
    StaticGPUVector<T>::destroy();
    if (free_list_ != nullptr) {
        clReleaseMemObject(free_list_);
        free_list_ = nullptr;
    }
    free_size_ = 0;
    free_capacity_ = 0;
}

template <typename T>
void GPUVector<T>::reallocateFreeList(size_t new_capacity) {
    if (new_capacity == 0) {
        if (free_list_ != nullptr) {
            clReleaseMemObject(free_list_);
            free_list_ = nullptr;
        }
        free_capacity_ = 0;
        return;
    }

    cl_int err;
    cl_context context = OpenCLManager::getContext();
    cl_command_queue queue = OpenCLManager::getQueue();

    // Create new free list buffer
    cl_mem new_free_list = clCreateBuffer(context, CL_MEM_READ_WRITE,
        new_capacity * sizeof(size_t), nullptr, &err);
    clCheckError(err, "clCreateBuffer (free list)");

    // Copy old data if it exists
    if (free_list_ != nullptr && free_size_ > 0) {
        err = clEnqueueCopyBuffer(queue, free_list_, new_free_list, 0, 0,
            free_size_ * sizeof(size_t), 0, nullptr, nullptr);
        clCheckError(err, "clEnqueueCopyBuffer (free list)");
        clFinish(queue);

        clReleaseMemObject(free_list_);
    }

    free_list_ = new_free_list;
    free_capacity_ = new_capacity;
}

template <typename T>
size_t GPUVector<T>::push(const T& value) {
    // If we have empty spaces in the middle of the vector, add the value there
    if (free_size_ > 0) {
        size_t hostIndex;
        cl_int err;
        cl_command_queue queue = OpenCLManager::getQueue();

        // Read the last free index
        err = clEnqueueReadBuffer(queue, free_list_, CL_TRUE,
            (free_size_ - 1) * sizeof(size_t), sizeof(size_t),
            &hostIndex, 0, nullptr, nullptr);
        clCheckError(err, "clEnqueueReadBuffer (free list)");

        // Set the value at that index
        this->setItem(hostIndex, value);
        --free_size_;
        return hostIndex;
    }

    // No free slots, append to end
    if (this->size_ == this->capacity_) {
        size_t new_capacity = this->capacity_ == 0 ? 1 : this->capacity_ * 2;
        this->reallocateDevice(new_capacity);
    }

    this->setItem(this->size_, value);
    ++this->size_;
    return this->size_ - 1;
}

template <typename T>
size_t GPUVector<T>::getNextIndex() {
    if (free_size_ > 0) {
        size_t hostIndex;
        cl_int err;
        cl_command_queue queue = OpenCLManager::getQueue();

        // Read the last free index
        err = clEnqueueReadBuffer(queue, free_list_, CL_TRUE,
            (free_size_ - 1) * sizeof(size_t), sizeof(size_t),
            &hostIndex, 0, nullptr, nullptr);
        clCheckError(err, "clEnqueueReadBuffer (getNextIndex)");
        return hostIndex;
    }
    return this->size_;
}

template <typename T>
void GPUVector<T>::remove(size_t index) {
    // Reallocate free list if needed
    if (free_size_ == free_capacity_) {
        size_t new_capacity = free_capacity_ == 0 ? 1 : free_capacity_ * 2;
        reallocateFreeList(new_capacity);
    }

    cl_int err;
    cl_command_queue queue = OpenCLManager::getQueue();

    // Add index to free list
    err = clEnqueueWriteBuffer(queue, free_list_, CL_TRUE,
        free_size_ * sizeof(size_t), sizeof(size_t),
        &index, 0, nullptr, nullptr);
    clCheckError(err, "clEnqueueWriteBuffer (remove)");
    free_size_++;
}

template <typename T>
void GPUVector<T>::update(size_t i, const T& value) {
    if (i >= this->size_) {
        throw std::out_of_range("Index out of range");
    }
    this->setItem(i, value);
}
