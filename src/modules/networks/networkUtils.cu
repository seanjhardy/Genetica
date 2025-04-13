#include <modules/networks/networkUtils.hpp>

// CUDA kernel for element-wise tanh activation
__global__ void tanh_activation(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = tanhf(data[idx]);
    }
}

// CUDA kernel to add bias and perform element-wise addition
__global__ void add_bias_and_vectors(float* output, const float* input1, const float* input2,
                                     const float* bias, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        // Add bias to each row
        output[idx] = input1[idx] + input2[idx] + bias[col];
    }
}
