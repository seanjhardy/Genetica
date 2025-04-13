#ifndef NETWORK_UTILS
#define NETWORK_UTILS

// CUDA kernel for element-wise tanh activation
__global__ void tanh_activation(float* data, int size) ;

// CUDA kernel to add bias and perform element-wise addition
__global__ void add_bias_and_vectors(float* output, const float* input1, const float* input2,
                                     const float* bias, int rows, int cols);

#endif