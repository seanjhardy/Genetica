#ifndef QUAD_TREE_NODE
#define QUAD_TREE_NODE

#include <vector>
#include <vector_types.h>
#include <modules/verlet/point.hpp>

class Quadtree;

class QuadtreeNode {
private:
    static const int MAX_POINTS = 4;
    static const int MAX_DEPTH = 12;

    float2 center;
    float2 halfDimension;
    float looseBoundary = 0.2f;
    int depth;

    std::vector<Point*> points;
    std::unique_ptr<QuadtreeNode> children[4];

public:
    QuadtreeNode(float2 center, float2 halfDimension, int depth = 0);

    void update(Quadtree* quadtree);
    bool insert(Point* point);
    void render(VertexManager& vertexManager);
    void reset();

    std::vector<Point*> queryRange(float2 rangeMin, float2 rangeMax);
    std::vector<Point*> queryCircle(float2 center, float radius);


private:
    [[nodiscard]] bool inBoundary(float2 position) const;

    bool isSubdivided();

    void subdivide() ;

    [[nodiscard]] bool intersects(float2 rangeMin, float2 rangeMax) const;

    static bool inRange(float2 position, float2 rangeMin, float2 rangeMax);
};

#endif