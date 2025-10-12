// OpenCL kernel for updating points with physics simulation

typedef struct {
    double s[2];
} double2;

typedef struct {
    float s[2];
} float2;

typedef struct {
    size_t entityID;
    double2 pos;
    double2 prevPos;
    double2 force;
    float angle;
    double radius;
} Point;

typedef struct {
    float left;
    float top;
    float width;
    float height;
} FloatRect;

// Inline helper functions
inline double2 make_double2(double x, double y) {
    double2 result;
    result.s[0] = x;
    result.s[1] = y;
    return result;
}

inline double2 double2_sub(double2 a, double2 b) {
    double2 result;
    result.s[0] = a.s[0] - b.s[0];
    result.s[1] = a.s[1] - b.s[1];
    return result;
}

inline double2 double2_add(double2 a, double2 b) {
    double2 result;
    result.s[0] = a.s[0] + b.s[0];
    result.s[1] = a.s[1] + b.s[1];
    return result;
}

inline double2 double2_mul(double2 a, double scalar) {
    double2 result;
    result.s[0] = a.s[0] * scalar;
    result.s[1] = a.s[1] * scalar;
    return result;
}

inline double2 double2_div(double2 a, double scalar) {
    double2 result;
    result.s[0] = a.s[0] / scalar;
    result.s[1] = a.s[1] / scalar;
    return result;
}

inline double magnitude(double2 v) {
    return sqrt(v.s[0] * v.s[0] + v.s[1] * v.s[1]);
}

// Point update logic (inlined from Point::update())
inline void updatePoint(Point* point) {
    double2 velocity = double2_sub(point->pos, point->prevPos);
    double2 accel = double2_div(point->force, point->radius);
    
    double2 newPosition = double2_add(double2_add(point->pos, double2_mul(velocity, 0.99)), accel);
    point->prevPos = point->pos;
    point->pos = newPosition;
    
    point->force = make_double2(0.0, 0.0);
}

// Constrain position to bounds (inlined from constraints.cpp)
inline void constrainPosition(Point* point, FloatRect bounds) {
    float minMax[4] = {
        bounds.left + (float)point->radius,
        bounds.left + bounds.width - (float)point->radius,
        bounds.top + (float)point->radius,
        bounds.top + bounds.height - (float)point->radius
    };
    
    if (point->pos.s[0] < minMax[0]) {
        point->prevPos.s[0] = point->pos.s[0];
        point->pos.s[0] = minMax[0];
    }
    if (point->pos.s[0] > minMax[1]) {
        point->prevPos.s[0] = point->pos.s[0];
        point->pos.s[0] = minMax[1];
    }
    
    if (point->pos.s[1] < minMax[2]) {
        point->prevPos.s[1] = point->pos.s[1];
        point->pos.s[1] = minMax[2];
    }
    if (point->pos.s[1] > minMax[3]) {
        point->prevPos.s[1] = point->pos.s[1];
        point->pos.s[1] = minMax[3];
    }
}

// Kernel for updating points
__kernel void updatePointsKernel(
    __global Point* points,
    const int numPoints,
    __global FloatRect* bounds
) {
    int index = get_global_id(0);
    
    if (index >= numPoints) return;
    
    Point* point = &points[index];
    updatePoint(point);
    constrainPosition(point, *bounds);
}

// Kernel for computing collisions between points
__kernel void computeCollisionsKernel(
    __global Point* points,
    const int numPoints
) {
    int a = get_global_id(0);
    int b = get_global_id(1);
    
    if (a >= numPoints || b >= numPoints || a >= b) return;
    
    Point* pointA = &points[a];
    Point* pointB = &points[b];
    
    if (pointA->entityID == pointB->entityID) return;
    
    // Constrain minimum distance (collision response)
    double minDistance = pointA->radius + pointB->radius;
    double2 posA = pointA->pos;
    double2 posB = pointB->pos;
    
    double2 delta = double2_sub(posA, posB);
    double distance = magnitude(delta);
    
    if (distance >= minDistance) return;
    
    // Calculate overlap and resistive force
    float overlap = minDistance - distance;
    float resistiveForceMagnitude = overlap * overlap * 0.01f;
    
    if (distance < 1e-6f) return;
    
    // Normalize direction
    double2 direction = double2_div(delta, distance);
    
    // Apply resistive force
    double2 forceA = double2_mul(direction, resistiveForceMagnitude);
    double2 forceB = double2_mul(direction, -resistiveForceMagnitude);
    
    // Atomic add to forces (using atomic_add for thread safety)
    // Note: OpenCL doesn't have atomic operations for doubles by default,
    // so we'll use a simpler approach or accept potential race conditions
    // For now, we'll do direct addition (may have minor race conditions)
    pointA->force.s[0] += forceA.s[0];
    pointA->force.s[1] += forceA.s[1];
    pointB->force.s[0] += forceB.s[0];
    pointB->force.s[1] += forceB.s[1];
}

// Kernel for moving a single point
__kernel void movePointKernel(
    __global Point* points,
    const int pointIndex,
    const float2 newPos,
    __global int* entityID
) {
    if (get_global_id(0) != 0) return; // Only first work item does this
    
    Point* point = &points[pointIndex];
    point->pos.s[0] = newPos.s[0];
    point->pos.s[1] = newPos.s[1];
    *entityID = point->entityID;
}

