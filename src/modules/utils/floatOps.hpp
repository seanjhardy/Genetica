#ifndef FLOAT_OPS
#define FLOAT_OPS

#include <vector_types.h>
#include <vector_functions.h>

inline __host__ __device__ float2 operator+(const float2 &a, const float2 &b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ float2 operator-(const float2 &a, const float2 &b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ float2 operator*(const float2 &a, const float &b) {
    return make_float2(a.x * b, a.y * b);
}

inline __host__ __device__ float2 operator*(const float &b, const float2 &a) {
    return make_float2(a.x * b, a.y * b);
}

inline __host__ __device__ bool operator==(const float2 &a, const float2 &b) {
    return (a.x == b.x) && (a.y == b.y);
}
#endif