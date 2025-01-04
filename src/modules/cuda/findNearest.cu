#include <modules/cuda/findNearest.hpp>
#include <cuda_runtime.h>
#include "modules/utils/print.hpp"

// Atomic function for updating a float value using atomicCAS
__device__ float atomicMinFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;  // Treat the address as an integer
    int old_as_int = *addr_as_int;  // Load the old value as an integer
    float old = __int_as_float(old_as_int);  // Convert it back to float

    while (value < old) {
        int assumed_as_int = old_as_int;
        old_as_int = atomicCAS(addr_as_int, assumed_as_int, __float_as_int(value));
        old = __int_as_float(old_as_int);  // Update old with the new value

        if (old == __int_as_float(assumed_as_int)) {
            // If CAS was successful, break the loop
            break;
        }
    }
    return old;
}

__global__ void findNearestKernel(GPUVector<Point>& points,
                                  float x, float y, float minDistance, int* closestIndex, float* distance) {
    // Allocate shared memory for thread's local minimum distance and index
    __shared__ float local_min_dist;
    __shared__ int local_min_idx;

    // Initialize shared variables with max values
    if (threadIdx.x == 0) {
        local_min_dist = minDistance * minDistance;
        local_min_idx = -1;
    }
    __syncthreads();

    // Compute the squared distance for each point in parallel
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < points.size()) {
        float dx = points[i].pos.x - x;
        float dy = points[i].pos.y - y;
        float dist_sq = dx * dx + dy * dy;

        // Use atomic operations to update the closest point
        atomicMinFloat(&local_min_dist, dist_sq);
        if (dist_sq == local_min_dist) {
            atomicExch(&local_min_idx, i);
        }
    }
    __syncthreads();

    // Write the final result back to global memory
    if (threadIdx.x == 0) {
        if (local_min_idx != -1) {
            *closestIndex = local_min_idx;
            *distance = sqrt(local_min_dist);
        } else {
            *closestIndex = -1.0f;
            *distance = -1;
        }
    }
}

std::pair<int, float> findNearest(GPUVector<Point> &points, float x, float y, float minDistance) {
    int numPoints = (int)points.size();

    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;

    int* d_closest_idx;
    float* d_closest_dist;
    cudaMalloc(&d_closest_idx, sizeof(int));
    cudaMalloc(&d_closest_dist, sizeof(float));

    findNearestKernel<<<numBlocks, blockSize>>>(points, x, y, minDistance,
                                                d_closest_idx, d_closest_dist);

    // Copy results back to host
    int h_closest_idx;
    float h_closest_dist;
    cudaMemcpy(&h_closest_idx, d_closest_idx, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_closest_dist, d_closest_dist, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_closest_idx);
    cudaFree(d_closest_dist);

    return std::make_pair(h_closest_idx, h_closest_dist);
}