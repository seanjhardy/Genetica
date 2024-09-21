// GPUVector.h

#ifndef GPU_VALUE_H
#define GPU_VALUE_H

#include "vector"
#include "cuda_runtime.h"

template<typename T>
class GPUValue {
private:
    T* d_data;
    T h_data;

public:
    // Disable copy constructor and assignment operator
    GPUValue(const GPUValue&) = delete;
    GPUValue& operator=(const GPUValue&) = delete;

    // Move constructor and assignment operator
    explicit GPUValue(T& value);
    GPUValue& operator=(T value);

    //Destructor
    ~GPUValue();

    T* deviceData() { return d_data; }

    void syncToHost();
    void syncToDevice();

    T& hostData() { return h_data; }
    T* hostDataPtr() { return &h_data; }
};

#include "../../../src/modules/cuda/GPUValue.tpp"

#endif