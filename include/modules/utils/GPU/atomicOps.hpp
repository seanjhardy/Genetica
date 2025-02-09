#include <cuda_runtime.h>

__device__ double atomicAddDouble(double* address, double val);
