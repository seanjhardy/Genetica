#include "cuda_runtime.h"

// Helper functions for Bézier curves
__device__ float2 bezierPoint(float2 p0, float2 p1, float2 p2, float2 p3, float t) {
    float mt = 1 - t;
    float mt2 = mt * mt;
    float mt3 = mt2 * mt;
    float t2 = t * t;
    float t3 = t2 * t;

    return make_float2(
      mt3 * p0.x + 3 * mt2 * t * p1.x + 3 * mt * t2 * p2.x + t3 * p3.x,
    mt3 * p0.y + 3 * mt2 * t * p1.y + 3 * mt * t2 * p2.y + t3 * p3.y
    );
};

__device__ void generateCircle(
  const Point& point,
  float2* vertices,
  int& vertexCount,
  int numPoints,
  const ViewParams& view
) {
    float angleStep = 2.0f * M_PI / numPoints;
    for (int i = 0; i < numPoints; i++) {
        float angle = i * angleStep;
        vertices[vertexCount++] = make_float2(
          point.pos.x + point.radius * cosf(angle),
          point.pos.y + point.radius * sinf(angle)
        );
    }
}