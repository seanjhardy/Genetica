#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <vector>
/*
class GPUDirectRenderer {
private:
    // OpenGL buffer objects
    static GLuint vbo;
    static cudaGraphicsResource_t cuda_vbo_resource;

    // Buffer size management
    static size_t current_buffer_size;
    static size_t vertices_count;
    static constexpr size_t INITIAL_BUFFER_SIZE = 1024;

    // Mutex for thread synchronization
    static cuda::std::mutex* d_mutex;

public:
    static GPUDirectRenderer() : current_buffer_size(INITIAL_BUFFER_SIZE), vertices_count(0) {
        // Initialize OpenGL buffer
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, INITIAL_BUFFER_SIZE * sizeof(float2), nullptr, GL_DYNAMIC_DRAW);

        // Register buffer with CUDA
        cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);

        // Create device mutex
        cuda::std::mutex* h_mutex;
        cudaMalloc(&d_mutex, sizeof(cuda::std::mutex));
        cudaMemcpy(d_mutex, h_mutex, sizeof(cuda::std::mutex), cudaMemcpyHostToDevice);
    }

    static ~GPUDirectRenderer() {
        cudaGraphicsUnregisterResource(cuda_vbo_resource);
        glDeleteBuffers(1, &vbo);
        cudaFree(d_mutex);
    }

    // Called from CUDA kernel to add vertices
    static __device__ void addVertices(const float2* vertices, size_t count) {
        d_mutex->lock();

        // Map buffer for writing
        float2* mapped_ptr;
        size_t mapped_size;

        cudaGraphicsMapResources(1, &cuda_vbo_resource);
        cudaGraphicsResourceGetMappedPointer((void**)&mapped_ptr, &mapped_size, cuda_vbo_resource);

        // Resize buffer if needed
        if (vertices_count + count > current_buffer_size) {
            size_t new_size = current_buffer_size * 2;
            while (new_size < vertices_count + count) {
                new_size *= 2;
            }

            // Create new buffer
            GLuint new_vbo;
            glGenBuffers(1, &new_vbo);
            glBindBuffer(GL_ARRAY_BUFFER, new_vbo);
            glBufferData(GL_ARRAY_BUFFER, new_size * sizeof(float2), nullptr, GL_DYNAMIC_DRAW);

            // Copy existing data
            if (vertices_count > 0) {
                cudaMemcpy(mapped_ptr, vertices, vertices_count * sizeof(float2), cudaMemcpyDeviceToDevice);
            }

            // Clean up old buffer
            cudaGraphicsUnregisterResource(cuda_vbo_resource);
            glDeleteBuffers(1, &vbo);

            // Update state
            vbo = new_vbo;
            current_buffer_size = new_size;
            cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);
        }

        // Copy new vertices
        cudaMemcpy(mapped_ptr + vertices_count, vertices, count * sizeof(float2), cudaMemcpyDeviceToDevice);
        vertices_count += count;

        // Unmap buffer
        cudaGraphicsUnmapResources(1, &cuda_vbo_resource);

        d_mutex->unlock();
    }

    // Called from CPU to render accumulated vertices
    static void render() {
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glDrawArrays(GL_POINTS, 0, vertices_count);
        glDisableVertexAttribArray(0);

        // Reset vertex count for next frame
        vertices_count = 0;
    }
};*/