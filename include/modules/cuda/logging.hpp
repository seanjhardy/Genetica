#ifndef CUDA_LOGGING
#define CUDA_LOGGING

#include <cuda_runtime.h>

#define cudaLog(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define LOGGING true

__host__ __device__ inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (!LOGGING) return;
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPU Log: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#endif