template<typename T>
GPUValue<T>::GPUValue(T& value) {
    h_data = value;
    T* d_ptr;
    cudaMalloc(&d_ptr, sizeof(T));
    cudaMemcpy(d_ptr, &value, sizeof(T), cudaMemcpyHostToDevice);
    d_data = d_ptr;
}

template<typename T>
GPUValue<T>::~GPUValue() {
    if (d_data) {
        cudaFree(d_data);
    }
}

template<typename T>
GPUValue<T>& GPUValue<T>::operator=(T value) {
    h_data = value;
    cudaMemcpy(d_data, &value, sizeof(T), cudaMemcpyHostToDevice);
    return *this;
}


template<typename T>
void GPUValue<T>::syncToHost() {
    cudaMemcpy(h_data.data(), d_data, sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
void GPUValue<T>::syncToDevice() {
    cudaMemcpy(d_data, h_data.data(), sizeof(T), cudaMemcpyHostToDevice);
}
