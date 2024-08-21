#ifndef QUAD_TREE
#define QUAD_TREE

#include <vector>
#include <vector_types.h>
#include <modules/physics/point.hpp>
#include "quadtreeNode.hpp"

class Quadtree {
private:
    QuadtreeNode root;

public:
    Quadtree(float2 center, float2 dimensions);

    void insert(Point* point);
    void update();
    void render(VertexManager& vertexManager);
    void reset();

    std::vector<Point*> queryRange(float2 rangeMin, float2 rangeMax);
    std::vector<Point*> queryCircle(float2 center, float radius);
    Point* findNearestPoint(float2 position, float maxDistance);
};

#endif