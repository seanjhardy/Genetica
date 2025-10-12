#pragma once

#include "../../../include/modules/gpu/structures/staticGPUVector.hpp"
#include <stdexcept>

template <typename T>
StaticGPUVector<T>::StaticGPUVector(size_t capacity) {
    size_ = capacity;
    capacity_ = capacity;
    reallocateDevice(capacity);
}

template <typename T>
StaticGPUVector<T>::StaticGPUVector(const std::vector<T>& h_data) {
    size_ = h_data.size();
    capacity_ = h_data.size();

    if (data_ != nullptr) {
        clReleaseMemObject(data_);
        data_ = nullptr;
    }

    cl_int err;
    cl_context context = OpenCLManager::getContext();
    cl_command_queue queue = OpenCLManager::getQueue();

    data_ = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        h_data.size() * sizeof(T), (void*)h_data.data(), &err);
    clCheckError(err, "clCreateBuffer");
}

template <typename T>
void StaticGPUVector<T>::reallocateDevice(size_t new_capacity) {
    if (new_capacity == 0) {
        if (data_ != nullptr) {
            clReleaseMemObject(data_);
            data_ = nullptr;
        }
        size_ = 0;
        capacity_ = 0;
        return;
    }

    cl_int err;
    cl_context context = OpenCLManager::getContext();
    cl_command_queue queue = OpenCLManager::getQueue();

    // Create new buffer
    cl_mem new_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
        new_capacity * sizeof(T), nullptr, &err);
    clCheckError(err, "clCreateBuffer (reallocate)");

    // Copy old data if it exists
    if (data_ != nullptr && size_ > 0) {
        err = clEnqueueCopyBuffer(queue, data_, new_buffer, 0, 0,
            size_ * sizeof(T), 0, nullptr, nullptr);
        clCheckError(err, "clEnqueueCopyBuffer");

        // Wait for copy to complete
        clFinish(queue);

        // Release old buffer
        clReleaseMemObject(data_);
    }

    data_ = new_buffer;
    capacity_ = new_capacity;
}

template <typename T>
void StaticGPUVector<T>::destroy() {
    if (data_ != nullptr) {
        clReleaseMemObject(data_);
        data_ = nullptr;
        size_ = 0;
        capacity_ = 0;
    }
}

template <typename T>
StaticGPUVector<T> StaticGPUVector<T>::copy() const {
    StaticGPUVector copy(size_);
    if (size_ > 0) {
        cl_int err;
        cl_command_queue queue = OpenCLManager::getQueue();

        err = clEnqueueCopyBuffer(queue, data_, copy.data_, 0, 0,
            size_ * sizeof(T), 0, nullptr, nullptr);
        clCheckError(err, "clEnqueueCopyBuffer (copy)");
        clFinish(queue);
    }
    return copy;
}

template <typename T>
std::vector<T> StaticGPUVector<T>::toHost() const {
    std::vector<T> host_data(size_);
    if (size_ > 0) {
        cl_int err;
        cl_command_queue queue = OpenCLManager::getQueue();

        err = clEnqueueReadBuffer(queue, data_, CL_TRUE, 0,
            size_ * sizeof(T), host_data.data(), 0, nullptr, nullptr);
        clCheckError(err, "clEnqueueReadBuffer");
    }
    return host_data;
}

template <typename T>
T StaticGPUVector<T>::itemToHost(size_t index) const {
    T host_data;
    if (index < size_) {
        cl_int err;
        cl_command_queue queue = OpenCLManager::getQueue();

        err = clEnqueueReadBuffer(queue, data_, CL_TRUE, index * sizeof(T),
            sizeof(T), &host_data, 0, nullptr, nullptr);
        clCheckError(err, "clEnqueueReadBuffer (item)");
    }
    return host_data;
}

template <typename T>
T StaticGPUVector<T>::getItem(size_t index) const {
    if (index >= size_) {
        throw std::out_of_range("Index out of range");
    }
    return itemToHost(index);
}

template <typename T>
void StaticGPUVector<T>::setItem(size_t index, const T& value) {
    if (index >= size_) {
        throw std::out_of_range("Index out of range");
    }

    cl_int err;
    cl_command_queue queue = OpenCLManager::getQueue();

    err = clEnqueueWriteBuffer(queue, data_, CL_TRUE, index * sizeof(T),
        sizeof(T), &value, 0, nullptr, nullptr);
    clCheckError(err, "clEnqueueWriteBuffer (setItem)");
}

template <typename T>
void StaticGPUVector<T>::resize(size_t new_size) {
    if (new_size > capacity_) {
        reallocateDevice(new_size);
    }
    size_ = new_size;
}

template <typename T>
T StaticGPUVector<T>::operator[](size_t index) const {
    if (index >= size_) {
        throw std::out_of_range("Index out of range");
    }
    return getItem(index);
}
