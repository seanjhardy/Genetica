#include <modules/quadtree/quadtree.hpp>
#include <modules/utils/print.hpp>

Quadtree::Quadtree(float2 center, float2 dimensions)
: root(center, make_float2(dimensions.x / 2, dimensions.y / 2)) {}

void Quadtree::insert(Point* point) {
    root.insert(point);
}

std::vector<Point*> Quadtree::queryRange(float2 rangeMin, float2 rangeMax) {
    return root.queryRange(rangeMin, rangeMax);
}

std::vector<Point*> Quadtree::queryCircle(float2 center, float radius) {
    return root.queryCircle(center, radius);
}

void Quadtree::update() {
    root.update(this);
}

void Quadtree::render(VertexManager& vertexManager) {
    root.render(vertexManager);
}

void Quadtree::reset() {
    root.reset();
}