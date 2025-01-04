#ifndef GPU_GRAPHICS_MANAGER
#define GPU_GRAPHICS_MANAGER

#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

class GPUGraphicsManager {
    GLuint vao;
    GLuint vbo;
    cudaGraphicsResource* cudaVBO;

    GPUPolygonRenderer() {
        // Initialize OpenGL resources
        glGenBuffers(1, &vbo);
        glGenVertexArrays(1, &vao);

        // Register buffer with CUDA - this is key for zero-copy
        cudaGraphicsGLRegiusterBuffer(&cudaVBO, vbo,
                                     cudaGraphicsMapFlagsWriteDiscard);
    }

    // Tessellation helper
    __host__ __device__ void tessellatePolygon(const float* points, size_t numPoints,
                           std::vector<float>& vertices,
                           std::vector<unsigned int>& indices);
};

#endif