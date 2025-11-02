#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <stdexcept>

// OpenCL headers
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// Helper function to read kernel file
std::string readFile(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  std::string content((std::istreambuf_iterator<char>(file)),
    std::istreambuf_iterator<char>());
  return content;
}

// Helper function to check OpenCL errors
void checkError(cl_int error, const std::string& message) {
  if (error != CL_SUCCESS) {
    throw std::runtime_error(message + " (Error code: " + std::to_string(error) + ")");
  }
}

int main() {
  try {
    std::cout << "=== OpenCL 3x3 Kernel Example ===" << std::endl;

    cl_int error;

    // 1. Get platform
    cl_platform_id platform;
    error = clGetPlatformIDs(1, &platform, nullptr);
    checkError(error, "Failed to get platform");

    // Get platform name
    char platformName[128];
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, 128, platformName, nullptr);
    std::cout << "Platform: " << platformName << std::endl;

    // 2. Get device (try GPU, then CPU, then any available device)
    cl_device_id device;
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (error != CL_SUCCESS) {
      std::cout << "GPU not available, trying CPU..." << std::endl;
      error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
    }
    if (error != CL_SUCCESS) {
      std::cout << "CPU not available, trying any device..." << std::endl;
      error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, nullptr);
    }
    checkError(error, "Failed to get device");

    // Get device name
    char deviceName[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, 128, deviceName, nullptr);
    std::cout << "Device: " << deviceName << std::endl;

    // 3. Create context
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
    checkError(error, "Failed to create context");
    std::cout << "Created OpenCL context" << std::endl;

    // 4. Create command queue
#ifdef CL_VERSION_2_0
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &error);
#else
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &error);
#endif
    checkError(error, "Failed to create command queue");
    std::cout << "Created command queue" << std::endl;

    // 5. Load and compile kernel
    std::string kernelSource = readFile("opencl_example_kernel.cl");
    const char* kernelSourcePtr = kernelSource.c_str();
    size_t kernelSourceSize = kernelSource.size();

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSourcePtr,
      &kernelSourceSize, &error);
    checkError(error, "Failed to create program");

    error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (error != CL_SUCCESS) {
      // Get build log
      size_t logSize;
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
      std::vector<char> log(logSize);
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
      std::cerr << "Build log:\n" << log.data() << std::endl;
      throw std::runtime_error("Failed to build program");
    }
    std::cout << "Compiled OpenCL kernel" << std::endl;

    // 6. Create kernel
    cl_kernel kernel = clCreateKernel(program, "compute_grid", &error);
    checkError(error, "Failed to create kernel");
    std::cout << "Created kernel 'compute_grid'" << std::endl;

    // 7. Create buffers
    struct Constants {
      float multiplier_x = 2.0f;
      float multiplier_y = 3.0f;
    } constants;

    // Constants buffer (input)
    cl_mem constantsBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sizeof(Constants), &constants, &error);
    checkError(error, "Failed to create constants buffer");
    std::cout << "Created constants buffer: x=" << constants.multiplier_x
      << ", y=" << constants.multiplier_y << std::endl;

    // Results buffer (output)
    const size_t resultCount = 9;
    const size_t resultBufferSize = resultCount * sizeof(float);
    cl_mem resultsBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
      resultBufferSize, nullptr, &error);
    checkError(error, "Failed to create results buffer");
    std::cout << "Created results buffer for " << resultCount << " elements" << std::endl;

    // 8. Set kernel arguments
    error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &constantsBuffer);
    checkError(error, "Failed to set kernel arg 0");
    error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &resultsBuffer);
    checkError(error, "Failed to set kernel arg 1");
    std::cout << "Set kernel arguments" << std::endl;

    // 9. Execute kernel
    std::cout << "Dispatching 3x3 kernel..." << std::endl;
    size_t globalWorkSize = 9; // 3x3 = 9 work items
    error = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize,
      nullptr, 0, nullptr, nullptr);
    checkError(error, "Failed to enqueue kernel");

    // Wait for kernel to finish
    error = clFinish(queue);
    checkError(error, "Failed to finish queue");
    std::cout << "Kernel execution complete" << std::endl;

    // 10. Read results back to CPU
    std::vector<float> results(resultCount);
    error = clEnqueueReadBuffer(queue, resultsBuffer, CL_TRUE, 0, resultBufferSize,
      results.data(), 0, nullptr, nullptr);
    checkError(error, "Failed to read results buffer");

    // 11. Display results
    std::cout << "\nResults from 3x3 kernel:" << std::endl;
    for (int y = 0; y < 3; ++y) {
      for (int x = 0; x < 3; ++x) {
        int index = y * 3 + x;
        float expected = x * constants.multiplier_x + y * constants.multiplier_y;
        std::cout << "Position (" << x << "," << y << "): result=" << results[index]
          << ", expected=" << expected << std::endl;
      }
    }

    // 12. Cleanup
    clReleaseMemObject(constantsBuffer);
    clReleaseMemObject(resultsBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    std::cout << "\nOpenCL example completed successfully!" << std::endl;

  }
  catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}

