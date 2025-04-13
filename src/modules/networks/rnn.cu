/*#include <modules/networks/rnn.hpp>
#include <modules/networks/networkUtils.hpp>

// Initialize weights with pre-defined values
void RNN::setWeights(const float* W_ih, const float* W_hh, const float* W_ho,
                const float* b_h, const float* b_o) const {
    cudaLog(cudaMemcpy(W_inputHidden, W_ih, config.input_size * config.hidden_size * sizeof(float),
                     cudaMemcpyHostToDevice));
    cudaLog(cudaMemcpy(W_hiddenHidden, W_hh, config.hidden_size * config.hidden_size * sizeof(float),
                     cudaMemcpyHostToDevice));
    cudaLog(cudaMemcpy(W_hiddenOutput, W_ho, config.hidden_size * config.output_size * sizeof(float),
                     cudaMemcpyHostToDevice));
    cudaLog(cudaMemcpy(d_b_h, b_h, config.hidden_size * sizeof(float),
                     cudaMemcpyHostToDevice));
    cudaLog(cudaMemcpy(d_b_o, b_o, config.output_size * sizeof(float),
                     cudaMemcpyHostToDevice));

    // Initialize hidden state to zeros
    cudaLog(cudaMemset(d_hidden_state, 0, config.batch_size * config.hidden_size * sizeof(float)));
}

// Set input data
void RNN::setInput(const float* input_data) const {
    cudaLog(cudaMemcpy(d_input, input_data,
                     config.batch_size * config.input_size * sizeof(float),
                     cudaMemcpyHostToDevice));
}

// Get output data
void RNN::getOutput(float* output_data) const {
    cudaLog(cudaMemcpy(output_data, d_output,
                     config.batch_size * config.output_size * sizeof(float),
                     cudaMemcpyDeviceToHost));
}

// Get hidden state
void RNN::getHiddenState(float* hidden_data) const {
    cudaLog(cudaMemcpy(hidden_data, d_hidden_state,
                     config.batch_size * config.hidden_size * sizeof(float),
                     cudaMemcpyDeviceToHost));
}

// Perform a single RNN step
void RNN::forward(cublasHandle_t handle, const RNNConfig& config,
               float* d_input, float* d_hidden_state, float* d_hidden_new,
               float* d_output, const float* d_W_ih, const float* d_W_hh,
               const float* d_W_ho, const float* d_b_h, const float* d_b_o) {

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Step 1: Compute input contribution to hidden state: d_hidden_new = d_input * d_W_ih
    // Note: cuBLAS uses column-major order, so we compute d_W_ih^T * d_input^T
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        config.hidden_size, config.batch_size, config.input_size,
                        &alpha,
                        d_W_ih, config.hidden_size,
                        d_input, config.input_size,
                        &beta,
                        d_hidden_new, config.hidden_size));

    // Step 2: Add recurrent contribution: d_hidden_new += d_hidden_state * d_W_hh
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        config.hidden_size, config.batch_size, config.hidden_size,
                        &alpha,
                        d_W_hh, config.hidden_size,
                        d_hidden_state, config.hidden_size,
                        &alpha,  // Note: beta = 1.0 to add to existing value
                        d_hidden_new, config.hidden_size));

    // Step 3: Add bias and apply tanh activation
    dim3 block(16, 16);
    dim3 grid((config.hidden_size + block.x - 1) / block.x,
              (config.batch_size + block.y - 1) / block.y);

    add_bias_and_vectors<<<grid, block>>>(d_hidden_new, d_hidden_new,
                                          d_hidden_state, d_b_h,
                                          config.batch_size, config.hidden_size);

    int hidden_size = config.batch_size * config.hidden_size;
    int block_size = 256;
    int grid_size = (hidden_size + block_size - 1) / block_size;
    tanh_activation<<<grid_size, block_size>>>(d_hidden_new, hidden_size);

    // Step 4: Copy new hidden state to current hidden state
    cudaLog(cudaMemcpy(d_hidden_state, d_hidden_new,
                    config.batch_size * config.hidden_size * sizeof(float),
                    cudaMemcpyDeviceToDevice));

    // Step 5: Compute output: d_output = d_hidden_state * d_W_ho
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        config.output_size, config.batch_size, config.hidden_size,
                        &alpha,
                        d_W_ho, config.output_size,
                        d_hidden_state, config.hidden_size,
                        &beta,
                        d_output, config.output_size));

    // Step 6: Add output bias
    dim3 output_block(16, 16);
    dim3 output_grid((config.output_size + output_block.x - 1) / output_block.x,
                    (config.batch_size + output_block.y - 1) / output_block.y);

    // Use a simpler kernel for just adding bias (no second vector)
    add_bias_and_vectors<<<output_grid, output_block>>>(d_output, d_output,
                                                     d_output, d_b_o,
                                                     config.batch_size, config.output_size);
}*/