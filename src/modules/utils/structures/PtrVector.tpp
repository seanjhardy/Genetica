#pragma once
#include <modules/cuda/logging.hpp>
#include <cstring>
#include <vector>


template <typename T>
PtrVector<T>::PtrVector(const std::vector<T>& hostData)
    : StaticPtrVector<T>(hostData), overflow_(nullptr), overflow_size_(0), overflow_capacity_(0) {}

template <typename T>
void PtrVector<T>::clear() {
    StaticPtrVector<T>::clear();
    delete[] overflow_;
    overflow_ = nullptr;
    overflow_size_ = 0;
    overflow_capacity_ = 0;
}

template <typename T>
size_t PtrVector<T>::push(const T& value) {
    // If we have empty spaces in the middle of the vector, add the value there
    if (overflow_size_ > 0) {
        size_t hostIndex = overflow_[--overflow_size_];
        this->data_[hostIndex] = value;
        return hostIndex;
    }

    if (this->size_ == this->capacity_) {
        size_t new_capacity = this->capacity_ == 0 ? 1 : this->capacity_ * 2;
        this->reallocateHost(new_capacity);
    }

    this->data_[this->size_] = value;
    return this->size_++;
}


template <typename T>
size_t PtrVector<T>::push(const T&& value) {
    // If we have empty spaces in the middle of the vector, add the value there
    if (overflow_size_ > 0) {
        size_t hostIndex = overflow_[--overflow_size_];
        this->data_[hostIndex] = std::move(value);
        return hostIndex;
    }

    if (this->size_ == this->capacity_) {
        size_t new_capacity = this->capacity_ == 0 ? 1 : this->capacity_ * 2;
        this->reallocateHost(new_capacity);
    }

    this->data_[this->size_] = value;
    return this->size_++;
}

template <typename T>
size_t PtrVector<T>::getNextIndex() {
    if (overflow_size_ > 0) {
        return overflow_[overflow_size_ - 1];
    }
    return this->size_;
}

template <typename T>
void PtrVector<T>::remove(size_t index) {
    if (overflow_size_ == overflow_capacity_) {
        size_t new_overflow_capacity = overflow_capacity_ == 0 ? 1 : overflow_capacity_ * 2;
        size_t* new_overflow = new size_t[new_overflow_capacity];
        std::memcpy(new_overflow, overflow_, overflow_capacity_ * sizeof(size_t));
        delete[] overflow_;
        overflow_ = new_overflow;
        overflow_capacity_ = new_overflow_capacity;
    }

    overflow_[overflow_size_++] = index;
}

template <typename T>
void PtrVector<T>::update(size_t i, T value) {
    if (i >= this->size_) {
        throw std::out_of_range("Index out of range");
    }
    this->data_[i] = value;
}
