/*#ifndef CUDA_RNN_H
#define CUDA_RNN_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <modules/gpu/logging.hpp>

// cuBLAS error checking macro
#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

struct RNNConfig {
    int input_size;
    int hidden_size;
    int output_size;
    int batch_size;
};

class RNN {
    RNNConfig config;
    cublasHandle_t handle;

    // Network parameters (weights and biases)
    float *W_inputHidden; // Weight matrix: input -> hidden
    float *W_hiddenHidden; // Weight matrix: hidden -> hidden (recurrent)
    float *W_hiddenOutput; // Weight matrix: hidden -> output
    float *d_b_h;  // Hidden bias
    float *d_b_o;  // Output bias

    // Activations and states
    float *d_input;       // Input data
    float *d_hidden_state; // Hidden state
    float *d_hidden_new;   // New hidden state (temporary)
    float *d_output;       // Output

public:
    RNN(const RNNConfig& config) : config(config) {
        // Initialize cuBLAS
        CUBLAS_CHECK(cublasCreate(&handle));

        // Allocate memory for weights and biases
        cudaLog(cudaMalloc(&W_inputHidden, config.input_size * config.hidden_size * sizeof(float)));
        cudaLog(cudaMalloc(&W_hiddenHidden, config.hidden_size * config.hidden_size * sizeof(float)));
        cudaLog(cudaMalloc(&W_hiddenOutput, config.hidden_size * config.output_size * sizeof(float)));
        cudaLog(cudaMalloc(&d_b_h, config.hidden_size * sizeof(float)));
        cudaLog(cudaMalloc(&d_b_o, config.output_size * sizeof(float)));

        // Allocate memory for activations and states
        cudaLog(cudaMalloc(&d_input, config.batch_size * config.input_size * sizeof(float)));
        cudaLog(cudaMalloc(&d_hidden_state, config.batch_size * config.hidden_size * sizeof(float)));
        cudaLog(cudaMalloc(&d_hidden_new, config.batch_size * config.hidden_size * sizeof(float)));
        cudaLog(cudaMalloc(&d_output, config.batch_size * config.output_size * sizeof(float)));
    }

    ~RNN() {
        // Free memory
        cudaLog(cudaFree(W_inputHidden));
        cudaLog(cudaFree(W_hiddenHidden));
        cudaLog(cudaFree(W_hiddenOutput));
        cudaLog(cudaFree(d_b_h));
        cudaLog(cudaFree(d_b_o));
        cudaLog(cudaFree(d_input));
        cudaLog(cudaFree(d_hidden_state));
        cudaLog(cudaFree(d_hidden_new));
        cudaLog(cudaFree(d_output));

        // Destroy cuBLAS handle
        CUBLAS_CHECK(cublasDestroy(handle));
    }

    // Initialize weights with pre-defined values
    void setWeights(const float* W_ih, const float* W_hh, const float* W_ho,
                    const float* b_h, const float* b_o) const;

    // Set input data
    void setInput(const float* input_data) const;

    // Get output data
    void getOutput(float* output_data) const;

    // Get hidden state
    void getHiddenState(float* hidden_data) const;

    static void forward(cublasHandle_t handle, const RNNConfig& config,
               float* d_input, float* d_hidden_state, float* d_hidden_new,
               float* d_output, const float* d_W_ih, const float* d_W_hh,
               const float* d_W_ho, const float* d_b_h, const float* d_b_o);
};


#endif // CUDA_RNN_H/*