#pragma once

template<typename T>
CGPUVector<T>::CGPUVector(size_t capacity) : GPUVector<T>(capacity) {
    h_data.reserve(capacity);
}

template<typename T>
CGPUVector<T>::CGPUVector(const std::vector<T>& host_vector): GPUVector<T>(host_vector) {
    h_data = host_vector;
}

template<typename T>
void CGPUVector<T>::push(const T& value) {
    GPUVector<T>::push(value);
    h_data.push_back(value);
}

template<typename T>
void CGPUVector<T>::remove(int index) {
    h_data.erase(h_data.begin() + index);
    GPUVector<T>::remove(index);
}


template<typename T>
T& CGPUVector<T>::operator[](size_t index) {
    return h_data[index];
}

template <typename T>
void CGPUVector<T>::clear() {
    h_data.clear();
    GPUVector<T>::clear();
}
