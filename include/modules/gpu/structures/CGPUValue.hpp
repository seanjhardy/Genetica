#ifndef CGPU_VALUE_H
#define CGPU_VALUE_H

#include <OpenCL/opencl.h>
#include "../OpenCLManager.hpp"

// Simple wrapper for a single value that can be stored on both CPU and GPU
template<typename T>
class CGPUValue {
private:
  T hostValue_;
  cl_mem deviceBuffer_ = nullptr;
  bool isDirty_ = true; // Track if device needs update

public:
  // Constructor
  CGPUValue() : hostValue_{} {}
  explicit CGPUValue(const T& value) : hostValue_(value) {}

  // Destructor
  ~CGPUValue() {
    if (deviceBuffer_ != nullptr) {
      clReleaseMemObject(deviceBuffer_);
    }
  }

  // Copy constructor
  CGPUValue(const CGPUValue& other) : hostValue_(other.hostValue_), isDirty_(true) {}

  // Assignment operator
  CGPUValue& operator=(const T& value) {
    hostValue_ = value;
    isDirty_ = true;
    return *this;
  }

  // Get host value
  T hostData() const {
    return hostValue_;
  }

  T* hostDataPtr() {
    return &hostValue_;
  }

  // Get device buffer (creates if needed)
  cl_mem deviceData() {
    if (deviceBuffer_ == nullptr || isDirty_) {
      syncToDevice();
    }
    return deviceBuffer_;
  }

  // Sync host to device
  void syncToDevice() {
    cl_int err;
    cl_context context = OpenCLManager::getContext();
    cl_command_queue queue = OpenCLManager::getQueue();

    if (deviceBuffer_ == nullptr) {
      deviceBuffer_ = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(T), &hostValue_, &err);
      clCheckError(err, "clCreateBuffer (CGPUValue)");
    }
    else {
      err = clEnqueueWriteBuffer(queue, deviceBuffer_, CL_TRUE, 0,
        sizeof(T), &hostValue_, 0, nullptr, nullptr);
      clCheckError(err, "clEnqueueWriteBuffer (CGPUValue)");
    }
    isDirty_ = false;
  }

  // Sync device to host
  void syncToHost() {
    if (deviceBuffer_ != nullptr) {
      cl_int err;
      cl_command_queue queue = OpenCLManager::getQueue();
      err = clEnqueueReadBuffer(queue, deviceBuffer_, CL_TRUE, 0,
        sizeof(T), &hostValue_, 0, nullptr, nullptr);
      clCheckError(err, "clEnqueueReadBuffer (CGPUValue)");
    }
  }
};

#endif // CGPU_VALUE_H

