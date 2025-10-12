#ifndef VECTOR_TYPES_HPP
#define VECTOR_TYPES_HPP

// OpenCL vector type definitions
// Pure OpenCL implementation without CUDA compatibility

#include <OpenCL/opencl.h>
#include <cmath>

// OpenCL vector types - using OpenCL's native vector types directly
typedef cl_float2 float2;
typedef cl_float3 float3;
typedef cl_double2 double2;
typedef cl_double3 double3;
typedef cl_int2 int2;
typedef cl_int3 int3;
typedef cl_uint2 uint2;
typedef cl_uint3 uint3;


// Constructor functions for OpenCL vector types
inline float2 make_float2(float x, float y) {
  float2 v;
  v.s[0] = x;
  v.s[1] = y;
  return v;
}

inline float3 make_float3(float x, float y, float z) {
  float3 v;
  v.s[0] = x;
  v.s[1] = y;
  v.s[2] = z;
  return v;
}


inline double2 make_double2(double x, double y) {
  double2 v;
  v.s[0] = x;
  v.s[1] = y;
  return v;
}

inline double3 make_double3(double x, double y, double z) {
  double3 v;
  v.s[0] = x;
  v.s[1] = y;
  v.s[2] = z;
  return v;
}


inline int2 make_int2(int x, int y) {
  int2 v;
  v.s[0] = x;
  v.s[1] = y;
  return v;
}

inline int3 make_int3(int x, int y, int z) {
  int3 v;
  v.s[0] = x;
  v.s[1] = y;
  v.s[2] = z;
  return v;
}

inline uint2 make_uint2(unsigned int x, unsigned int y) {
  uint2 v;
  v.s[0] = x;
  v.s[1] = y;
  return v;
}

inline uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z) {
  uint3 v;
  v.s[0] = x;
  v.s[1] = y;
  v.s[2] = z;
  return v;
}

// Additional utility functions
inline float2 make_float2(const double2& d) {
  return make_float2(static_cast<float>(d.s[0]), static_cast<float>(d.s[1]));
}

inline double2 make_double2(const float2& f) {
  return make_double2(static_cast<double>(f.s[0]), static_cast<double>(f.s[1]));
}

// OpenCL vector functions
// Dot product
inline float dot(float2 a, float2 b) {
  return a.s[0] * b.s[0] + a.s[1] * b.s[1];
}

inline float dot(float3 a, float3 b) {
  return a.s[0] * b.s[0] + a.s[1] * b.s[1] + a.s[2] * b.s[2];
}

inline double dot(double2 a, double2 b) {
  return a.s[0] * b.s[0] + a.s[1] * b.s[1];
}

inline double dot(double3 a, double3 b) {
  return a.s[0] * b.s[0] + a.s[1] * b.s[1] + a.s[2] * b.s[2];
}


// Length/magnitude
inline float length(float2 v) {
  return sqrtf(dot(v, v));
}

inline float length(float3 v) {
  return sqrtf(dot(v, v));
}
inline double length(double2 v) {
  return sqrt(dot(v, v));
}

inline double length(double3 v) {
  return sqrt(dot(v, v));
}


// Normalize
inline float2 normalize(float2 v) {
  float len = length(v);
  if (len > 0.0f) {
    return make_float2(v.s[0] / len, v.s[1] / len);
  }
  return make_float2(0.0f, 0.0f);
}

inline float3 normalize(float3 v) {
  float len = length(v);
  if (len > 0.0f) {
    return make_float3(v.s[0] / len, v.s[1] / len, v.s[2] / len);
  }
  return make_float3(0.0f, 0.0f, 0.0f);
}

inline double2 normalize(double2 v) {
  double len = length(v);
  if (len > 0.0) {
    return make_double2(v.s[0] / len, v.s[1] / len);
  }
  return make_double2(0.0, 0.0);
}

inline double3 normalize(double3 v) {
  double len = length(v);
  if (len > 0.0) {
    return make_double3(v.s[0] / len, v.s[1] / len, v.s[2] / len);
  }
  return make_double3(0.0, 0.0, 0.0);
}

// Cross product (for 3D vectors)
inline float3 cross(float3 a, float3 b) {
  return make_float3(
    a.s[1] * b.s[2] - a.s[2] * b.s[1],
    a.s[2] * b.s[0] - a.s[0] * b.s[2],
    a.s[0] * b.s[1] - a.s[1] * b.s[0]
  );
}

inline double3 cross(double3 a, double3 b) {
  return make_double3(
    a.s[1] * b.s[2] - a.s[2] * b.s[1],
    a.s[2] * b.s[0] - a.s[0] * b.s[2],
    a.s[0] * b.s[1] - a.s[1] * b.s[0]
  );
}

// Distance
inline float distance(float2 a, float2 b) {
  return length(make_float2(a.s[0] - b.s[0], a.s[1] - b.s[1]));
}

inline float distance(float3 a, float3 b) {
  return length(make_float3(a.s[0] - b.s[0], a.s[1] - b.s[1], a.s[2] - b.s[2]));
}

inline double distance(double2 a, double2 b) {
  return length(make_double2(a.s[0] - b.s[0], a.s[1] - b.s[1]));
}

inline double distance(double3 a, double3 b) {
  return length(make_double3(a.s[0] - b.s[0], a.s[1] - b.s[1], a.s[2] - b.s[2]));
}


#endif // VECTOR_TYPES_HPP