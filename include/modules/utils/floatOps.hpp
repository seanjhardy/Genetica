#ifndef FLOAT_OPS
#define FLOAT_OPS

#include "vector_types.h"
#include "vector_functions.h"

inline __host__ __device__ float2 vec(float direction) {
    return make_float2(cos(direction), sin(direction));
}

inline __host__ __device__ float2 operator+(const float2 &a, const float2 &b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ float2 operator+(const float2 &a, const float &b) {
    return make_float2(a.x + b, a.y + b);
}

inline __host__ __device__ float2 operator-(const float2 &a, const float2 &b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ float2 operator-(const float2 &a, const float &b) {
    return make_float2(a.x - b, a.y - b);
}

inline __host__ __device__ float2 operator*(const float2 &a, const float &b) {
    return make_float2(a.x * b, a.y * b);
}

inline __host__ __device__ float2 operator*(const float &b, const float2 &a) {
    return make_float2(a.x * b, a.y * b);
}

inline __host__ __device__ float2 operator/(const float2 &a, const float &b) {
    return make_float2(a.x / b, a.y / b);
}


inline __host__ __device__ float2 operator+=(float2 &a, const float2 &b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}

inline __host__ __device__ float2 operator-=(float2 &a, const float2 &b) {
    a.x -= b.x;
    a.y -= b.y;
    return a;
}


inline __host__ __device__ float2 operator*=(float2 &a, const float &b) {
    a.x *= b;
    a.y *= b;
    return a;
}
inline __host__ __device__ float2 operator/=(float2 &a, const float &b) {
    a.x /= b;
    a.y /= b;
    return a;
}

inline __host__ __device__ bool operator==(const float2 &a, const float2 &b) {
    return (a.x == b.x) && (a.y == b.y);
}

inline __host__ __device__ float2 rotate(float2 point, float angle) {
    float x = point.x * cos(angle) - point.y * sin(angle);
    float y = point.x * sin(angle) + point.y * cos(angle);
    return make_float2(x, y);
}

inline __host__ __device__ float diff(float2 p1, float2 p2) {
    return abs(p1.x - p2.x) + abs(p1.y - p2.y);
}


#endif