#ifndef QUAD_TREE_NODE
#define QUAD_TREE_NODE

#include <vector>
#include <modules/utils/vector_types.hpp>
#include <modules/physics/point.hpp>

class Quadtree;

class QuadtreeNode {
private:
    static const int MAX_POINTS = 4;
    static const int MAX_DEPTH = 12;

    float looseBoundary = 0.2f;
    int depth;

    std::unique_ptr<QuadtreeNode> children[4];
    float2 center;
    float2 halfDimension;
    std::vector<Point*> points;

public:
    QuadtreeNode(float2 center, float2 halfDimension, int depth = 0);

    void update(Quadtree* quadtree);
    bool insert(Point* point);
    void render(VertexManager& vertexManager);
    void reset();

    std::vector<Point*> queryRange(float2 rangeMin, float2 rangeMax);
    std::vector<Point*> queryCircle(float2 center, float radius);
    void findNearestPoint(float2 position, Point*& nearestPoint, float& nearestDistanceSquared);
    bool overlapsSearch(float2 position, float distanceSquared);
private:
    [[nodiscard]] bool inBoundary(float2 position) const;
    void subdivide() ;
    [[nodiscard]] bool intersects(float2 rangeMin, float2 rangeMax) const;
    static bool inRange(float2 position, float2 rangeMin, float2 rangeMax);
    bool isSubdivided();
};

#endif