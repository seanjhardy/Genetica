#ifndef FLOAT_OPS
#define FLOAT_OPS

#include "vector_types.h"
#include "vector_functions.h"
#include <modules/utils/GPU/fastMath.hpp>
#include <cuda_runtime.h>

inline __host__ __device__ float2 vec(float direction) {
    return {cosf(direction), sinf(direction)};
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


inline __host__ __device__ float2 rotate(float2 point, float angle) {
    float2 d = vec(angle);
    float x = point.x * d.x - point.y * d.y;
    float y = point.x * d.y + point.y * d.x;
    return make_float2(x, y);
}

inline __host__ __device__ double2 rotate(double2 point, float angle) {
    float2 d = vec(angle);
    double x = point.x * d.x - point.y * d.y;
    double y = point.x * d.y + point.y * d.x;
    return make_double2(x, y);
}


inline __host__ __device__ float2 rotateOrigin(float2 point, float2 origin, float angle) {
    float2 d = vec(angle);
    float x = (point.x - origin.x) * d.x - (point.y - origin.y) * d.y;
    float y = (point.x - origin.x) * d.y + (point.y - origin.y) * d.x;
    return make_float2(x, y);
}

inline __host__ __device__ float diff(float2 p1, float2 p2) {
    return abs(p1.x - p2.x) + abs(p1.y - p2.y);
}

inline __host__ __device__ float sum(float2 p1) {
    return p1.x + p1.y;
}
inline __host__ __device__ float sum(double2 p1) {
    return p1.x + p1.y;
}

inline __host__ __device__ float sum(float3 p1) {
    return p1.x + p1.y + p1.z;
}

inline __host__ __device__ float distanceBetween(float2 p1, float2 p2) {
    float2 d = p1 - p2;
    return sqrt(sum(d*d));
}
inline __host__ __device__ float distanceBetween(double2 p1, double2 p2) {
    double2 d = p1 - p2;
    return sqrt(sum(d*d));
}

inline __host__ __device__ float distanceBetween(float3 p1, float3 p2) {
    float3 d = p1 - p2;
    return sqrt(sum(d*d));
}

inline __host__ __device__ float dir(float2 p1, float2 p2) {
    float2 d = p1 - p2;
    return std::atan2f(d.y, d.x);
}

inline __host__ sf::FloatRect operator+=(sf::FloatRect &a, sf::FloatRect &b) {
    a.left += b.left;
    a.top += b.top;
    a.width += b.width;
    a.height += b.height;
    return a;
}

inline __host__ __device__ float magnitude(float2 p1) {
    return sqrtf(p1.x * p1.x + p1.y * p1.y);
}

inline __host__ __device__ double magnitude(double2 p1) {
    return sqrt(p1.x * p1.x + p1.y * p1.y);
}

inline __host__ __device__ float magnitude(float3 p1) {
    return sqrt(p1.x * p1.x + p1.y * p1.y + p1.z * p1.z);
}

#endif