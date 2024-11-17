// GPUVector.h

#ifndef GPU_VALUE_H
#define GPU_VALUE_H

#include "vector"
#include "cuda_runtime.h"

template<typename T>
class CGPUValue {
private:
    T* d_data;
    T h_data;

public:
    // Disable copy constructor and assignment operator
    CGPUValue(const CGPUValue&) = delete;
    CGPUValue& operator=(const CGPUValue&) = delete;

    // Move constructor and assignment operator
    explicit CGPUValue(T& value);
    CGPUValue& operator=(T value);

    //Destructor
    ~CGPUValue();

    T* deviceData() { return d_data; }

    void syncToHost();
    void syncToDevice();

    T& hostData() { return h_data; }
    T* hostDataPtr() { return &h_data; }
};

#include "../../../src/modules/cuda/CGPUValue.tpp"

#endif