// OpenCL kernel for updating points with physics simulation
// Using float precision for Apple Silicon Metal compatibility

struct Point {
  unsigned long entityID;
  float2 pos;
  float2 prevPos;
  float2 force;
  float angle;
  float radius;
};

struct FloatRect {
  float left;
  float top;
  float width;
  float height;
};

// Kernel for updating points
kernel void updatePointsKernel(global struct Point *points, int numPoints,
                               global struct FloatRect *bounds) {
  int index = get_global_id(0);

  if (index >= numPoints)
    return;

  global struct Point *point = &points[index];

  // Update physics (verlet integration)
  float2 velocity = point->pos - point->prevPos;
  float2 accel = point->force / point->radius;
  float2 newPosition = point->pos + velocity * 0.99f + accel;

  point->prevPos = point->pos;
  point->pos = newPosition;
  point->force = (float2)(0.0f, 0.0f);

  // Constrain to bounds
  float minX = bounds->left + point->radius;
  float maxX = bounds->left + bounds->width - point->radius;
  float minY = bounds->top + point->radius;
  float maxY = bounds->top + bounds->height - point->radius;

  if (point->pos.x < minX) {
    point->prevPos.x = point->pos.x;
    point->pos.x = minX;
  }
  if (point->pos.x > maxX) {
    point->prevPos.x = point->pos.x;
    point->pos.x = maxX;
  }
  if (point->pos.y < minY) {
    point->prevPos.y = point->pos.y;
    point->pos.y = minY;
  }
  if (point->pos.y > maxY) {
    point->prevPos.y = point->pos.y;
    point->pos.y = maxY;
  }
}

// Kernel for computing collisions between points
kernel void computeCollisionsKernel(global struct Point *points,
                                    int numPoints) {
  int a = get_global_id(0);
  int b = get_global_id(1);

  if (a >= numPoints || b >= numPoints || a >= b)
    return;

  global struct Point *pointA = &points[a];
  global struct Point *pointB = &points[b];

  if (pointA->entityID == pointB->entityID)
    return;

  // Constrain minimum distance (collision response)
  float minDistance = pointA->radius + pointB->radius;
  float2 delta = pointA->pos - pointB->pos;
  float distance = length(delta);

  if (distance >= minDistance)
    return;
  if (distance < 1e-6f)
    return;

  // Calculate overlap and resistive force
  float overlap = minDistance - distance;
  float resistiveForceMagnitude = overlap * overlap * 0.01f;

  // Normalize direction and apply force
  float2 direction = delta / distance;
  float2 forceA = direction * resistiveForceMagnitude;

  pointA->force += forceA;
  pointB->force -= forceA;
}

// Kernel for moving a single point
kernel void movePointKernel(global struct Point *points, int pointIndex,
                            float2 newPos, global int *entityID) {
  if (get_global_id(0) != 0)
    return;

  global struct Point *point = &points[pointIndex];
  point->pos.x = newPos.x;
  point->pos.y = newPos.y;
  *entityID = (int)point->entityID;
}
