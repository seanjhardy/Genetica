#include "include/modules/gpu/structures/staticGPUVector.hpp"
#include "include/modules/gpu/OpenCLManager.hpp"
#include <iostream>
#include <vector>
#include <OpenCL/opencl.h>

// Example OpenCL kernel source code
const char* kernel_source = R"(
__kernel void vector_add(__global float* a, __global float* b, __global float* result, int n) {
    int i = get_global_id(0);
    if (i < n) {
        result[i] = a[i] + b[i];
    }
}
)";

int main() {
  try {
    // Initialize OpenCL
    OpenCLManager::init();

    // Create input vectors
    std::vector<float> a_data = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
    std::vector<float> b_data = { 10.0f, 20.0f, 30.0f, 40.0f, 50.0f };

    // Create GPU vectors
    StaticGPUVector<float> gpu_a(a_data);
    StaticGPUVector<float> gpu_b(b_data);
    StaticGPUVector<float> gpu_result(5); // Result vector

    std::cout << "Created GPU vectors for kernel computation" << std::endl;

    // Create OpenCL program and kernel
    cl_context context = OpenCLManager::getContext();
    cl_command_queue queue = OpenCLManager::getQueue();
    cl_device_id device = OpenCLManager::getDevice();

    // Create program
    cl_int err;
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, nullptr, &err);
    clCheckError(err, "clCreateProgramWithSource");

    // Build program
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    clCheckError(err, "clBuildProgram");

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
    clCheckError(err, "clCreateKernel");

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), gpu_a.getBufferPtr());
    clCheckError(err, "clSetKernelArg (a)");

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), gpu_b.getBufferPtr());
    clCheckError(err, "clSetKernelArg (b)");

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), gpu_result.getBufferPtr());
    clCheckError(err, "clSetKernelArg (result)");

    int n = 5;
    err = clSetKernelArg(kernel, 3, sizeof(int), &n);
    clCheckError(err, "clSetKernelArg (n)");

    // Execute kernel
    size_t global_size = 5;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    clCheckError(err, "clEnqueueNDRangeKernel");

    // Wait for completion
    clFinish(queue);

    // Get results
    std::vector<float> result = gpu_result.toHost();

    std::cout << "Kernel computation results: ";
    for (float val : result) {
      std::cout << val << " ";
    }
    std::cout << std::endl;

    // Cleanup
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    std::cout << "Kernel example completed successfully!" << std::endl;

  }
  catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  // Cleanup
  OpenCLManager::cleanup();
  return 0;
}
