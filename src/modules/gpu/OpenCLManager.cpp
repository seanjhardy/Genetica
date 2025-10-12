#include "../../../include/modules/gpu/OpenCLManager.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

// Static member definitions
cl_context OpenCLManager::context_ = nullptr;
cl_command_queue OpenCLManager::queue_ = nullptr;
cl_device_id OpenCLManager::device_ = nullptr;
bool OpenCLManager::initialized_ = false;
std::map<std::string, cl_program> OpenCLManager::programs_;
std::map<std::string, cl_kernel> OpenCLManager::kernels_;

void OpenCLManager::init() {
  if (initialized_) {
    return; // Already initialized
  }

  cl_int err;
  cl_platform_id platform;
  cl_uint num_platforms;
  cl_uint num_devices;

  // Get platform
  err = clGetPlatformIDs(1, &platform, &num_platforms);
  clCheckError(err, "clGetPlatformIDs");

  if (num_platforms == 0) {
    throw std::runtime_error("No OpenCL platforms found");
  }

  // Get device
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_, &num_devices);
  if (err != CL_SUCCESS || num_devices == 0) {
    // Fallback to CPU if no GPU available
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device_, &num_devices);
    clCheckError(err, "clGetDeviceIDs (CPU fallback)");

    if (num_devices == 0) {
      throw std::runtime_error("No OpenCL devices found");
    }
  }

  // Create context
  context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
  clCheckError(err, "clCreateContext");

  // Create command queue
  queue_ = clCreateCommandQueue(context_, device_, 0, &err);
  clCheckError(err, "clCreateCommandQueue");

  initialized_ = true;

  std::cout << "OpenCL initialized successfully" << std::endl;
}

void OpenCLManager::cleanup() {
  if (!initialized_) {
    return;
  }

  // Release all kernels
  for (auto& pair : kernels_) {
    clReleaseKernel(pair.second);
  }
  kernels_.clear();

  // Release all programs
  for (auto& pair : programs_) {
    clReleaseProgram(pair.second);
  }
  programs_.clear();

  if (queue_) {
    clReleaseCommandQueue(queue_);
    queue_ = nullptr;
  }

  if (context_) {
    clReleaseContext(context_);
    context_ = nullptr;
  }

  device_ = nullptr;
  initialized_ = false;

  std::cout << "OpenCL cleaned up" << std::endl;
}

std::string OpenCLManager::loadKernelSource(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open kernel file: " + filename);
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

cl_program OpenCLManager::loadProgram(const std::string& filename) {
  if (!initialized_) {
    throw std::runtime_error("OpenCL not initialized");
  }

  // Check if already loaded
  auto it = programs_.find(filename);
  if (it != programs_.end()) {
    return it->second;
  }

  // Load source
  std::string source = loadKernelSource(filename);
  const char* sourcePtr = source.c_str();
  size_t sourceSize = source.length();

  cl_int err;
  cl_program program = clCreateProgramWithSource(context_, 1, &sourcePtr, &sourceSize, &err);
  clCheckError(err, "clCreateProgramWithSource");

  // Build program
  err = clBuildProgram(program, 1, &device_, nullptr, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    // Get build log
    size_t logSize;
    clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
    std::string buildLog(logSize, '\0');
    clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], nullptr);

    std::cerr << "OpenCL build error:\n" << buildLog << std::endl;
    clReleaseProgram(program);
    throw std::runtime_error("Failed to build OpenCL program");
  }

  programs_[filename] = program;
  std::cout << "Loaded OpenCL program: " << filename << std::endl;
  return program;
}

cl_kernel OpenCLManager::getKernel(const std::string& programName, const std::string& kernelName) {
  if (!initialized_) {
    throw std::runtime_error("OpenCL not initialized");
  }

  std::string key = programName + "::" + kernelName;

  // Check if already created
  auto it = kernels_.find(key);
  if (it != kernels_.end()) {
    return it->second;
  }

  // Get program
  auto progIt = programs_.find(programName);
  if (progIt == programs_.end()) {
    throw std::runtime_error("Program not loaded: " + programName);
  }

  cl_int err;
  cl_kernel kernel = clCreateKernel(progIt->second, kernelName.c_str(), &err);
  clCheckError(err, "clCreateKernel: " + kernelName);

  kernels_[key] = kernel;
  return kernel;
}

void OpenCLManager::runKernel1D(cl_kernel kernel, size_t globalSize, size_t localSize) {
  if (!initialized_) {
    throw std::runtime_error("OpenCL not initialized");
  }

  cl_int err;
  if (localSize == 0) {
    // Let OpenCL choose local size
    err = clEnqueueNDRangeKernel(queue_, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
  }
  else {
    err = clEnqueueNDRangeKernel(queue_, kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, nullptr);
  }
  clCheckError(err, "clEnqueueNDRangeKernel (1D)");

  clFinish(queue_);
}

void OpenCLManager::runKernel2D(cl_kernel kernel, size_t globalSizeX, size_t globalSizeY,
  size_t localSizeX, size_t localSizeY) {
  if (!initialized_) {
    throw std::runtime_error("OpenCL not initialized");
  }

  size_t globalSizes[2] = { globalSizeX, globalSizeY };
  cl_int err;

  if (localSizeX == 0 || localSizeY == 0) {
    // Let OpenCL choose local size
    err = clEnqueueNDRangeKernel(queue_, kernel, 2, nullptr, globalSizes, nullptr, 0, nullptr, nullptr);
  }
  else {
    size_t localSizes[2] = { localSizeX, localSizeY };
    err = clEnqueueNDRangeKernel(queue_, kernel, 2, nullptr, globalSizes, localSizes, 0, nullptr, nullptr);
  }
  clCheckError(err, "clEnqueueNDRangeKernel (2D)");

  clFinish(queue_);
}
