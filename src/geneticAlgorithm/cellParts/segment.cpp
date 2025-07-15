#include "geneticAlgorithm/lifeform.hpp"
#include <simulator/simulator.hpp>
#include <geneticAlgorithm/cellParts/cell.hpp>

Segment::Segment(size_t startPointIdx, size_t endPointIdx, size_t startPointAttachedIdx, size_t endPointAttachedIdx,
  float startLength, float targetLength) : startPointIdx(startPointIdx), endPointIdx(endPointIdx),
  startPointAttachedIdx(startPointAttachedIdx), endPointAttachedIdx(endPointAttachedIdx),
  startLength(startLength), targetLength(targetLength) {
  products = 0;
  startPointSize = 1.0f;
  endPointSize = 1.0f;
  length = startLength;
  stiffness = 1.0f;
  startPointTargetRadius = 0.0f;
  endPointTargetRadius = 0.0f;
  startDivisionRotation = 0.0f;
  endDivisionRotation = 0.0f;
  hue = 200.0f;
  saturation = 0.0f;
  luminosity = 0.0f;
  membraneThickness = 1.0f;
  generation = 0;
  lastDivideTime = 0;
  energy = 0.0f;
  dividing = false;
  dead = false;
  frozen = false;
  numDivisions = 0;
}

void Segment::renderBody(VertexManager& vertexManager, vector<Point>& points) const {
  const Point point1 = points[startPointIdx];
  const Point point2 = points[endPointIdx];
  const sf::Color startPointColor = getColor();
  const sf::Color endPointColor = getColor();

  // Find the angle between the points and draw a polygon connecting them:
  // From the center of the first, add a vertex on the circumference of the point tangential to the angle between p1 and p2
  // Then connect it to the vertex on the circumference of the second tangential to the angle between p2 and p1
  // Repeat for the other side
  const float angle = atan2(point2.pos.y - point1.pos.y, point2.pos.x - point1.pos.x);
  const float angle1 = angle + M_PI_HALF;
  const float angle2 = angle - M_PI_HALF;
  float2 v1 = point1.getPos() + make_float2(cos(angle1), sin(angle1)) * point1.radius;
  float2 v2 = point2.getPos() + make_float2(cos(angle1), sin(angle1)) * point2.radius;
  float2 v3 = point2.getPos() + make_float2(cos(angle2), sin(angle2)) * point2.radius;
  float2 v4 = point1.getPos() + make_float2(cos(angle2), sin(angle2)) * point1.radius;
  vertexManager.addPolygon(std::vector<VertexManager::Vertex>({
      {v1, startPointColor},
      {v2, endPointColor},
      {v3, endPointColor},

      {v3, endPointColor},
      {v4, startPointColor},
      {v1, startPointColor}
    }));
}

void Segment::renderDetails(VertexManager& vertexManager, vector<Point>& points) const {
}

void Segment::renderCellWalls(VertexManager& vertexManager, vector<Point>& points) const {
  const Point point1 = points[startPointIdx];
  const Point point2 = points[endPointIdx];
  const sf::Color startPointColor = brightness(getColor(), 0.6);
  const sf::Color endPointColor = brightness(getColor(), 0.6);

  // Find the angle between the points and draw a polygon connecting them:
  // From the center of the first, add a vertex on the circumference of the point tangential to the angle between p1 and p2
  // Then connect it to the vertex on the circumference of the second tangential to the angle between p2 and p1
  // Repeat for the other side
  const float angle = atan2(point2.pos.y - point1.pos.y, point2.pos.x - point1.pos.x);
  const float angle1 = angle + M_PI_HALF;
  const float angle2 = angle - M_PI_HALF;

  if (vertexManager.getSizeInView(point1.radius) < 5 && vertexManager.getSizeInView(point2.radius) < 5) return;
  float2 v1 = point1.getPos() + make_float2(cos(angle1), sin(angle1)) * (point1.radius + membraneThickness);
  float2 v2 = point2.getPos() + make_float2(cos(angle1), sin(angle1)) * (point2.radius + membraneThickness);
  float2 v3 = point2.getPos() + make_float2(cos(angle2), sin(angle2)) * (point2.radius + membraneThickness);
  float2 v4 = point1.getPos() + make_float2(cos(angle2), sin(angle2)) * (point1.radius + membraneThickness);
  vertexManager.addPolygon(std::vector<VertexManager::Vertex>({
      {v1, startPointColor},
      {v2, endPointColor},
      {v3, endPointColor},

      {v3, endPointColor},
      {v4, startPointColor},
      {v1, startPointColor}
    }));
}

