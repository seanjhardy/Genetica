#include "cuda_runtime.h"

struct MeshData {
    float2* d_vertices;
    int* d_vertexCount;

    explicit MeshData(size_t maxMeshSize) {
        // Allocate device memory
        cudaMalloc(&d_vertices, maxMeshSize * sizeof(float2));
        cudaMalloc(&d_vertexCount, sizeof(int));
    }
};