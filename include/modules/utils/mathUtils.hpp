#ifndef MATHUTILS_HPP
#define MATHUTILS_HPP

#include <modules/verlet/point.hpp>
#include <modules/utils/floatOps.hpp>

std::vector<float2> findPerpendicularPoints(const Point& point1, const Point& point2, float r1, float r2);

float getVelocity(const Point& point);

sf::Color interpolate(sf::Color c1, sf::Color c2, float x);
float clamp(float min_val, float x, float max_val);

float normAngle(float angle);

float angleDiff(float angle1, float angle2, bool norm = true);

float clockwiseAngleDiff(const float2& p1, const float2& p2);

std::vector<std::pair<int, int>> bezier(int x0, int y0, int x1, int y1, int x2, int y2, int num_points = 10);

float smoothAngle(float angle1, float angle2, float tolerance = 90);

float2 getPointOnSegment(float length, float r1, float r2, float angle);

std::vector<float> geometricProgression(int n, float r);

#endif // MATHUTILS_HPP