#pragma once

template <typename T>
StaticPtrVector<T>::StaticPtrVector(size_t capacity)
    : data_(nullptr), size_(0), capacity_(0) {
    reallocateHost(capacity);
}

template <typename T>
StaticPtrVector<T>::StaticPtrVector(const std::vector<T>& h_data)
    : data_(nullptr), size_(0), capacity_(0) {
    size_ = h_data.size();
    capacity_ = h_data.size();
    if (data_ != nullptr) {
        delete[] data_;
        data_ = nullptr;
    }
    data_ = new T[h_data.size()];
    std::memcpy(data_, h_data.data(), h_data.size() * sizeof(T));
}

template <typename T>
void StaticPtrVector<T>::reallocateHost(size_t new_capacity) {
    if (new_capacity == 0) {
        delete[] data_;
        data_ = nullptr;
        size_ = 0;
        capacity_ = 0;
        return;
    }

    T* old_data = data_;
    data_ = new T[new_capacity];
    if (old_data != nullptr) {
        std::memcpy(data_, old_data, size_ * sizeof(T));
        delete[] old_data;
    }
    capacity_ = new_capacity;
}

template <typename T>
void StaticPtrVector<T>::clear() {
    delete[] data_;
    data_ = nullptr;
    size_ = 0;
    capacity_ = 0;
}

template <typename T>
T& StaticPtrVector<T>::operator[](size_t index) {
    if (index >= size_) {
        throw std::out_of_range("Index out of range");
    }
    return data_[index];
}

template <typename T>
T& StaticPtrVector<T>::operator[](size_t index) const {
    if (index >= size_) {
        throw std::out_of_range("Index out of range");
    }
    return data_[index];
}

template <typename T>
void StaticPtrVector<T>::resize(size_t new_size) {
    if (new_size > capacity_) {
        reallocateHost(new_size);
    }
    size_ = new_size;
}
