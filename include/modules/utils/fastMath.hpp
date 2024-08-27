#ifndef FAST_MATH
#define FAST_MATH

#include <cmath>
#include <iostream>
#include <cuda_runtime.h>

#ifndef M_PI_2
    #define M_PI_2 6.283185f
#endif

class FastMath {
public:
    static const int TABLE_SIZE = 360;
    static float cosTable[TABLE_SIZE];
    static float sinTable[TABLE_SIZE];

    static void init();

    static __host__ float cos(float angle);
    static __host__ float sin(float angle);
    static __host__ __device__ float atan2f(float y, float x);

private:
    static __host__ float getValue(float angle, const float* table);
};

#endif