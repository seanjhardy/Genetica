#ifndef FLOAT_OPS
#define FLOAT_OPS

#include "modules/utils/vector_types.hpp"
#include "modules/utils/vector_functions.hpp"

inline float2 operator+(const float2& a, const float2& b) {
    return make_float2(a.s[0] + b.s[0], a.s[1] + b.s[1]);
}

inline float2 operator+(const float2& a, const float& b) {
    return make_float2(a.s[0] + b, a.s[1] + b);
}

inline float2 operator-(const float2& a, const float2& b) {
    return make_float2(a.s[0] - b.s[0], a.s[1] - b.s[1]);
}

inline float2 operator-(const float2& a, const float& b) {
    return make_float2(a.s[0] - b, a.s[1] - b);
}

inline float2 operator*(const float2& a, const float& b) {
    return make_float2(a.s[0] * b, a.s[1] * b);
}

inline float2 operator*(const float2& a, const float2& b) {
    return make_float2(a.s[0] * b.s[0], a.s[1] * b.s[1]);
}

inline float2 operator*(const float& b, const float2& a) {
    return make_float2(a.s[0] * b, a.s[1] * b);
}

inline float2 operator/(const float2& a, const float& b) {
    return make_float2(a.s[0] / b, a.s[1] / b);
}

inline float2 operator+=(float2& a, const float2& b) {
    a.s[0] += b.s[0];
    a.s[1] += b.s[1];
    return a;
}

inline float2 operator-=(float2& a, const float2& b) {
    a.s[0] -= b.s[0];
    a.s[1] -= b.s[1];
    return a;
}

inline float2 operator*=(float2& a, const float& b) {
    a.s[0] *= b;
    a.s[1] *= b;
    return a;
}
inline float2 operator/=(float2& a, const float& b) {
    a.s[0] /= b;
    a.s[1] /= b;
    return a;
}
inline bool operator==(const float2& a, const float2& b) {
    return a.s[0] == b.s[0] && a.s[1] == b.s[1];
}




inline double2 operator+(const double2& a, const double2& b) {
    return make_double2(a.s[0] + b.s[0], a.s[1] + b.s[1]);
}

inline double2 operator+(const double2& a, const float& b) {
    return make_double2(a.s[0] + b, a.s[1] + b);
}

inline double2 operator+(const double2& a, const float2& b) {
    return make_double2(a.s[0] + b.s[0], a.s[1] + b.s[1]);
}

inline double2 operator-(const double2& a, const double2& b) {
    return make_double2(a.s[0] - b.s[0], a.s[1] - b.s[1]);
}

inline double2 operator-(const double2& a, const float& b) {
    return make_double2(a.s[0] - b, a.s[1] - b);
}

inline double2 operator*(const double2& a, const float& b) {
    return make_double2(a.s[0] * b, a.s[1] * b);
}

inline double2 operator*(const double2& a, const double2& b) {
    return make_double2(a.s[0] * b.s[0], a.s[1] * b.s[1]);
}

inline double2 operator*(const float& b, const double2& a) {
    return make_double2(a.s[0] * b, a.s[1] * b);
}

inline double2 operator/(const double2& a, const float& b) {
    return make_double2(a.s[0] / b, a.s[1] / b);
}

inline double2 operator+=(double2& a, const double2& b) {
    a.s[0] += b.s[0];
    a.s[1] += b.s[1];
    return a;
}

inline double2 operator-=(double2& a, const double2& b) {
    a.s[0] -= b.s[0];
    a.s[1] -= b.s[1];
    return a;
}

inline double2 operator*=(double2& a, const float& b) {
    a.s[0] *= b;
    a.s[1] *= b;
    return a;
}
inline double2 operator/=(double2& a, const float& b) {
    a.s[0] /= b;
    a.s[1] /= b;
    return a;
}

inline bool operator==(const double2& a, const double2& b) {
    return a.s[0] == b.s[0] && a.s[1] == b.s[1];
}

inline float2 to_float2(const double2& a) {
    return make_float2(a.s[0], a.s[1]);
}

inline uint3 operator+(const uint3& a, const uint3& b) {
    return make_uint3(a.s[0] + b.s[0], a.s[1] + b.s[1], a.s[2] + b.s[2]);
}


inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.s[0] + b.s[0], a.s[1] + b.s[1], a.s[2] + b.s[2]);
}

inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.s[0] - b.s[0], a.s[1] - b.s[1], a.s[2] - b.s[2]);
}
inline float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.s[0] * b.s[0], a.s[1] * b.s[1], a.s[2] * b.s[2]);
}

inline float3 operator*(const float3& a, const float& b) {
    return make_float3(a.s[0] * b, a.s[1] * b, a.s[2] * b);
}

inline sf::FloatRect operator+=(sf::FloatRect& a, sf::FloatRect& b) {
    a.left += b.left;
    a.top += b.top;
    a.width += b.width;
    a.height += b.height;
    return a;
}

#endif