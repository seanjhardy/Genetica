template<typename T>
CGPUValue<T>::CGPUValue(T& value) {
    h_data = value;
    T* d_ptr;
    cudaMalloc(&d_ptr, sizeof(T));
    cudaMemcpy(d_ptr, &value, sizeof(T), cudaMemcpyHostToDevice);
    d_data = d_ptr;
}

template<typename T>
CGPUValue<T>::~CGPUValue() {
    if (d_data) {
        cudaFree(d_data);
    }
}

template<typename T>
CGPUValue<T>& CGPUValue<T>::operator=(T value) {
    h_data = value;
    cudaMemcpy(d_data, &value, sizeof(T), cudaMemcpyHostToDevice);
    return *this;
}


template<typename T>
void CGPUValue<T>::syncToHost() {
    cudaMemcpy(h_data.data_(), d_data, sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
void CGPUValue<T>::syncToDevice() {
    cudaMemcpy(d_data, h_data.data_(), sizeof(T), cudaMemcpyHostToDevice);
}
