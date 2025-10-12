#ifndef VECTOR_FUNCTIONS_HPP
#define VECTOR_FUNCTIONS_HPP

// OpenCL-compatible vector functions to replace CUDA vector_functions.h
// This provides the same interface as CUDA vector functions but uses OpenCL-compatible implementations

#include "vector_types.hpp"
#include "SFML/Graphics.hpp"
#include <cmath>


// Additional utility functions that might be needed
inline float2 fabsf(float2 v) {
  return { fabsf(v.s[0]), fabsf(v.s[1]) };
}

inline float3 fabsf(float3 v) {
  return { fabsf(v.s[0]), fabsf(v.s[1]), fabsf(v.s[2]) };
}


inline double2 fabs(double2 v) {
  return { fabs(v.s[0]), fabs(v.s[1]) };
}

inline double3 fabs(double3 v) {
  return { fabs(v.s[0]), fabs(v.s[1]), fabs(v.s[2]) };
}

// Min/max functions
inline float2 fminf(float2 a, float2 b) {
  return { fminf(a.s[0], b.s[0]), fminf(a.s[1], b.s[1]) };
}

inline float3 fminf(float3 a, float3 b) {
  return { fminf(a.s[0], b.s[0]), fminf(a.s[1], b.s[1]), fminf(a.s[2], b.s[2]) };
}

inline float2 fmaxf(float2 a, float2 b) {
  return { fmaxf(a.s[0], b.s[0]), fmaxf(a.s[1], b.s[1]) };
}

inline float3 fmaxf(float3 a, float3 b) {
  return { fmaxf(a.s[0], b.s[0]), fmaxf(a.s[1], b.s[1]), fmaxf(a.s[2], b.s[2]) };
}


inline double2 fmin(double2 a, double2 b) {
  return { fmin(a.s[0], b.s[0]), fmin(a.s[1], b.s[1]) };
}

inline double3 fmin(double3 a, double3 b) {
  return { fmin(a.s[0], b.s[0]), fmin(a.s[1], b.s[1]), fmin(a.s[2], b.s[2]) };
}


inline double2 fmax(double2 a, double2 b) {
  return { fmax(a.s[0], b.s[0]), fmax(a.s[1], b.s[1]) };
}

inline double3 fmax(double3 a, double3 b) {
  return { fmax(a.s[0], b.s[0]), fmax(a.s[1], b.s[1]), fmax(a.s[2], b.s[2]) };
}

// Clamp functions
inline float2 clamp(float2 v, float min_val, float max_val) {
  return fminf(fmaxf(v, make_float2(min_val, min_val)), make_float2(max_val, max_val));
}

inline float3 clamp(float3 v, float min_val, float max_val) {
  return fminf(fmaxf(v, make_float3(min_val, min_val, min_val)), make_float3(max_val, max_val, max_val));
}


inline double2 clamp(double2 v, double min_val, double max_val) {
  return fmin(fmax(v, make_double2(min_val, min_val)), make_double2(max_val, max_val));
}

inline double3 clamp(double3 v, double min_val, double max_val) {
  return fmin(fmax(v, make_double3(min_val, min_val, min_val)), make_double3(max_val, max_val, max_val));
}

// Trigonometric functions
inline float2 sinf(float2 v) {
  return { sinf(v.s[0]), sinf(v.s[1]) };
}

inline float3 sinf(float3 v) {
  return { sinf(v.s[0]), sinf(v.s[1]), sinf(v.s[2]) };
}

inline float2 cosf(float2 v) {
  return { cosf(v.s[0]), cosf(v.s[1]) };
}

inline float3 cosf(float3 v) {
  return { cosf(v.s[0]), cosf(v.s[1]), cosf(v.s[2]) };
}

inline float2 tanf(float2 v) {
  return { tanf(v.s[0]), tanf(v.s[1]) };
}

inline float3 tanf(float3 v) {
  return { tanf(v.s[0]), tanf(v.s[1]), tanf(v.s[2]) };
}

// Exponential and logarithmic functions
inline float2 expf(float2 v) {
  return { expf(v.s[0]), expf(v.s[1]) };
}

inline float3 expf(float3 v) {
  return { expf(v.s[0]), expf(v.s[1]), expf(v.s[2]) };
}


inline float2 logf(float2 v) {
  return { logf(v.s[0]), logf(v.s[1]) };
}

inline float3 logf(float3 v) {
  return { logf(v.s[0]), logf(v.s[1]), logf(v.s[2]) };
}

inline float2 sqrtf(float2 v) {
  return { sqrtf(v.s[0]), sqrtf(v.s[1]) };
}

inline float3 sqrtf(float3 v) {
  return { sqrtf(v.s[0]), sqrtf(v.s[1]), sqrtf(v.s[2]) };
}

inline double2 sqrt(double2 v) {
  return { sqrt(v.s[0]), sqrt(v.s[1]) };
}

inline double3 sqrt(double3 v) {
  return { sqrt(v.s[0]), sqrt(v.s[1]), sqrt(v.s[2]) };
}


#endif // VECTOR_FUNCTIONS_HPP

