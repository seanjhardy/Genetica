#ifndef FLOAT_OPS
#define FLOAT_OPS

#include "vector_types.h"
#include "vector_functions.h"
#include <modules/utils/GPU/fastMath.hpp>
#include <cuda_runtime.h>

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

inline __host__ __device__ float2 operator*(const float2 &a, const float2 &b) {
    return make_float2(a.x * b.x, a.y * b.y);
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
    return a.x == b.x && a.y == b.y;
}




inline __host__ __device__ double2 operator+(const double2 &a, const double2 &b) {
    return make_double2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ double2 operator+(const double2 &a, const float &b) {
    return make_double2(a.x + b, a.y + b);
}

inline __host__ __device__ double2 operator+(const double2 &a, const float2 &b) {
    return make_double2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ double2 operator-(const double2 &a, const double2 &b) {
    return make_double2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ double2 operator-(const double2 &a, const float &b) {
    return make_double2(a.x - b, a.y - b);
}

inline __host__ __device__ double2 operator*(const double2 &a, const float &b) {
    return make_double2(a.x * b, a.y * b);
}

inline __host__ __device__ double2 operator*(const double2 &a, const double2 &b) {
    return make_double2(a.x * b.x, a.y * b.y);
}

inline __host__ __device__ double2 operator*(const float &b, const double2 &a) {
    return make_double2(a.x * b, a.y * b);
}

inline __host__ __device__ double2 operator/(const double2 &a, const float &b) {
    return make_double2(a.x / b, a.y / b);
}

inline __host__ __device__ double2 operator+=(double2 &a, const double2 &b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}

inline __host__ __device__ double2 operator-=(double2 &a, const double2 &b) {
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

inline __host__ __device__ double2 operator*=(double2 &a, const float &b) {
    a.x *= b;
    a.y *= b;
    return a;
}
inline __host__ __device__ double2 operator/=(double2 &a, const float &b) {
    a.x /= b;
    a.y /= b;
    return a;
}

inline __host__ __device__ bool operator==(const double2 &a, const double2 &b) {
    return a.x == b.x && a.y == b.y;
}

inline __host__ __device__ float2 to_float2(const double2 &a) {
    return make_float2(a.x, a.y);
}

inline __host__ __device__ uint3 operator+(const uint3 &a, const uint3 &b) {
    return {static_cast<uint8_t>(a.x + b.x), static_cast<uint8_t>(a.y + b.y), static_cast<uint8_t>(a.z + b.z)};
}

inline __host__ __device__ uint4 operator*(const uint3 &a, const float &b) {
    return {static_cast<uint8_t>(a.x * b), static_cast<uint8_t>(a.y * b), static_cast<uint8_t>(a.z * b)};
}


inline __host__ __device__ float3 operator+(const float3 &a, const float3 &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline __host__ __device__ float3 operator-(const float3 &a, const float3 &b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
inline __host__ __device__ float3 operator*(const float3 &a, const float3 &b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

inline __host__ __device__ float3 operator*(const float3 &a, const float &b) {
    return {a.x * b, a.y * b, a.z * b};
}

inline __host__ sf::FloatRect operator+=(sf::FloatRect &a, sf::FloatRect &b) {
    a.left += b.left;
    a.top += b.top;
    a.width += b.width;
    a.height += b.height;
    return a;
}

#endif