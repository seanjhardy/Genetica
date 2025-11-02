#ifndef OPENCL_MANAGER_H
#define OPENCL_MANAGER_H

#include <OpenCL/opencl.h>
#include <stdexcept>
#include <string>
#include <map>

class OpenCLManager {
private:
  static cl_context context_;
  static cl_command_queue queue_;
  static cl_device_id device_;
  static bool initialized_;
  static std::map<std::string, cl_program> programs_;
  static std::map<std::string, cl_kernel> kernels_;

  // Helper to load source from file
  static std::string loadKernelSource(const std::string& filename);

public:
  // Initialize OpenCL context and command queue
  static void init();

  // Cleanup OpenCL resources
  static void cleanup();

  // Load and compile a kernel file
  static cl_program loadProgram(const std::string& filename);

  // Get or create a kernel from a loaded program
  static cl_kernel getKernel(const std::string& programName, const std::string& kernelName);

  // Helper to run a simple 1D kernel
  static void runKernel1D(cl_kernel kernel, size_t globalSize, size_t localSize = 0);

  // Helper to run a 2D kernel
  static void runKernel2D(cl_kernel kernel, size_t globalSizeX, size_t globalSizeY,
    size_t localSizeX = 0, size_t localSizeY = 0);

  // Flush command queue (ensure commands are submitted, non-blocking)
  static void flush();

  // Finish command queue (block until all commands complete)
  static void finish();

  // Get OpenCL context
  static cl_context getContext() {
    if (!initialized_) {
      throw std::runtime_error("OpenCL not initialized. Call OpenCLManager::init() first.");
    }
    return context_;
  }

  // Get OpenCL command queue
  static cl_command_queue getQueue() {
    if (!initialized_) {
      throw std::runtime_error("OpenCL not initialized. Call OpenCLManager::init() first.");
    }
    return queue_;
  }

  // Get OpenCL device
  static cl_device_id getDevice() {
    if (!initialized_) {
      throw std::runtime_error("OpenCL not initialized. Call OpenCLManager::init() first.");
    }
    return device_;
  }

  // Check if OpenCL is initialized
  static bool isInitialized() { return initialized_; }
};

// OpenCL error checking macro
#define CL_CHECK_ERROR(call) \
    do { \
        cl_int err = call; \
        if (err != CL_SUCCESS) { \
            throw std::runtime_error("OpenCL error: " + std::to_string(err) + " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__)); \
        } \
    } while(0)

// OpenCL error checking function
inline void clCheckError(cl_int err, const std::string& operation) {
  if (err != CL_SUCCESS) {
    throw std::runtime_error("OpenCL error in " + operation + ": " + std::to_string(err));
  }
}

#endif // OPENCL_MANAGER_H
