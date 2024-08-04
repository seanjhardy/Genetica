template<typename T>
GPUVector<T>::GPUVector(size_t capacity) : size_(0), capacity_(capacity) {
    reserve(capacity);
}

template<typename T>
GPUVector<T>::GPUVector(const std::vector<T>& host_vector) : h_data(host_vector), size_(host_vector.size()), capacity_(host_vector.size()) {
    cudaMalloc(&d_data, capacity_ * sizeof(T));
    cudaMemcpy(d_data, h_data.data(), size_ * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
GPUVector<T>::GPUVector(GPUVector&& other) noexcept
        : d_data(other.d_data), h_data(std::move(other.h_data)), size_(other.size_), capacity_(other.capacity_) {
    other.d_data = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
}

template<typename T>
void GPUVector<T>::reallocateDevice(size_t new_capacity) {
    T* new_d_data;
    cudaMalloc(&new_d_data, new_capacity * sizeof(T));
    if (d_data) {
        cudaMemcpy(new_d_data, d_data, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
        cudaFree(d_data);
    }
    d_data = new_d_data;
    capacity_ = new_capacity;
}

template<typename T>
GPUVector<T>::~GPUVector() {
    if (d_data) {
        cudaFree(d_data);
    }
}


template<typename T>
GPUVector<T>& GPUVector<T>::operator=(GPUVector&& other) noexcept {
    if (this != &other) {
        if (d_data) {
            cudaFree(d_data);
        }
        d_data = other.d_data;
        h_data = std::move(other.h_data);
        size_ = other.size_;
        capacity_ = other.capacity_;
        other.d_data = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }
    return *this;
}

template<typename T>
void GPUVector<T>::push_back(const T& value) {
    if (size_ == capacity_) {
        size_t new_capacity = capacity_ == 0 ? 1 : capacity_ * 2;
        reallocateDevice(new_capacity);
    }
    h_data.push_back(value);
    cudaMemcpy(d_data + size_, &value, sizeof(T), cudaMemcpyHostToDevice);
    ++size_;
}

template<typename T>
T* GPUVector<T>::back() {
    return &h_data.back();
}

template<typename T>
void GPUVector<T>::pop_back() {
    if (size_ > 0) {
        --size_;
        h_data.pop_back();
    }
}

template <typename T>
void GPUVector<T>::clear() {
    size_ = 0;
    h_data.clear();
}

template<typename T>
void GPUVector<T>::resize(size_t new_size) {
    if (new_size > capacity_) {
        reallocateDevice(new_size);
    }
    size_ = new_size;
    h_data.resize(new_size);
}

template<typename T>
void GPUVector<T>::reserve(size_t new_capacity) {
    if (new_capacity > capacity_) {
        h_data.reserve(new_capacity);
        reallocateDevice(new_capacity);
    }
}

template<typename T>
void GPUVector<T>::syncToHost() {
    cudaMemcpy(h_data.data(), d_data, size_ * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
void GPUVector<T>::syncToDevice() {
    cudaMemcpy(d_data, h_data.data(), size_ * sizeof(T), cudaMemcpyHostToDevice);
}
