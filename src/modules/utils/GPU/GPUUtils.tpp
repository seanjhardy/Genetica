template <typename T>
void saveGPUArray(T* d_data, const std::vector<T>& h_data) {
    if (d_data) {
        cudaFree(d_data);
    }
    cudaMalloc(&d_data, h_data.size() * sizeof(T));
    cudaMemcpy(d_data, h_data.data(), h_data.size() * sizeof(T), cudaMemcpyHostToDevice);
}