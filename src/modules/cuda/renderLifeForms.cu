#include <modules/cuda/renderLifeForms.hpp>
#include <modules/cuda/structures/MeshData.hpp>
#include <modules/cuda/structures/ViewParams.hpp>
#include <modules/utils/GPU/rendering.hpp>

// Returns angle between two points relative to center
__device__ float getAngle(float2 center, float2 p1, float2 p2) {
    float2 v1 = make_float2(p1.x - center.x, p1.y - center.y);
    float2 v2 = make_float2(p2.x - center.x, p2.y - center.y);

    float angle = atan2(v2.y, v2.x) - atan2(v1.y, v1.x);
    return angle < 0 ? angle + 2*M_PI : angle;
}

__global__ void renderLifeFormKernel(GPUVector<LifeForm>& lifeForms,
                                     GPUVector<Cell>& cells,
                                     GPUVector<CellLink>& cellLinks,
                                     GPUVector<Point>& points,
                                     ViewParams* viewParams) {
    float2* d_vertices;
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= lifeForms.size()) return;

    LifeForm &lifeForm = lifeForms[index];
}

void renderLifeForms(GPUVector<LifeForm>& lifeForms,
                     GPUVector<Cell>& cells,
                     GPUVector<CellLink>& cellLinks,
                     GPUVector<Point>& points,
                     ViewParams& viewParams) {
    // Transfer render data to device
    ViewParams* d_viewParams;
    cudaMalloc(&d_viewParams, sizeof(ViewParams));
    cudaMemcpy(d_viewParams, &viewParams, sizeof(ViewParams), cudaMemcpyHostToDevice);

    // Create mesh data store for each creature
    MeshData* lifeFormMeshes;
    cudaMalloc(&lifeFormMeshes, lifeForms.size() * sizeof(MeshData));

    for (int i = 0; i < lifeForms.size(); i++) {
        LifeForm &lifeForm = lifeForms[i];
        auto meshData = MeshData(lifeForm.cells.size() * 8);
        cudaMemcpy(&lifeFormMeshes[i], &meshData, sizeof(MeshData), cudaMemcpyHostToDevice);
    }

    int blockSize = 256; // Number of threads per block
    int numBlocks = (lifeForms.size() + blockSize - 1) / blockSize;
    renderLifeFormKernel<<<numBlocks, blockSize>>>(lifeForms, cells, cellLinks, points,
                                                   d_viewParams);


    MeshData* h_meshes = new MeshData[lifeForms.size()];

}