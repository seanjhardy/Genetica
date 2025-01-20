#ifndef CUDA_LOGGING
#define CUDA_LOGGING

#include <cuda_runtime.h>
#include <iostream>

#define cudaLog(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPU Log: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#endif