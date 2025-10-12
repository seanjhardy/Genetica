#ifndef MATHUTILS_HPP
#define MATHUTILS_HPP

#include "modules/physics/point.hpp"

#ifndef M_PI_HALF
#define M_PI_HALF 1.57079632679
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_PI_2
#define M_PI_2 6.28318530718
#endif
#ifndef M_PI_4
#define M_PI_4 0.78539816339
#endif
#ifndef SQRT_2
#define SQRT_2 1.41421356237
#endif


float2 vec(float direction);
float dir(float2 p1, float2 p2);
float2 rotate(float2 point, float angle);
double2 rotate(double2 point, float angle);
float2 rotateOrigin(float2 point, float2 origin, float angle);
float diff(float2 p1, float2 p2);
float sum(float2 p1);
float sum(double2 p1);
float sum(float3 p1);
float distanceBetween(float2 p1, float2 p2);
float distanceBetween(double2 p1, double2 p2);
float distanceBetween(float3 p1, float3 p2);
float angleBetween(float2 p1, float2 p2);
float magnitude(float2 p1);
double magnitude(double2 p1);
float magnitude(float3 p1);

std::vector<float2> findPerpendicularPoints(const Point& point1, const Point& point2, float r1, float r2);

float getVelocity(const Point& point);

sf::Color interpolate(sf::Color c1, sf::Color c2, float x);

inline   float clamp(float min_val, float x, float max_val) {
    return std::max(min_val, std::min(x, max_val));
}

float normAngle(float angle);

float angleDiff(float angle1, float angle2, bool norm = true);

float clockwiseAngleDiff(const float2& p1, const float2& p2);

std::vector<std::pair<int, int>> bezier(int x0, int y0, int x1, int y1, int x2, int y2, int num_points = 10);

float smoothAngle(float angle1, float angle2, float tolerance = 90);

float2 getPointOnSegment(float length, float r1, float r2, float angle);

std::vector<float> geometricProgression(int n, float r);

#endif // MATHUTILS_HPP