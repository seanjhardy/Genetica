#include <modules/quadtree/quadtreeNode.hpp>
#include "modules/utils/floatOps.hpp"
#include "modules/quadtree/quadtree.hpp"
#include <modules/utils/print.hpp>
#include <modules/graphics/vertexManager.hpp>

QuadtreeNode::QuadtreeNode(float2 center, float2 halfDimension, int depth)
    : center(center), halfDimension(halfDimension), depth(depth) {
    for (auto & i : children) {
        i = nullptr;
    }
}

bool QuadtreeNode::insert(Point* point) {
    if (!inBoundary(point->getPos())) {
        return false;
    }

    if (!isSubdivided()) {
        if (points.size() < MAX_POINTS || depth >= MAX_DEPTH) {
            points.push_back(point);
            return true;
        }
        subdivide();
    }

    for (int i = 0; i < 4; ++i) {
        if (children[i]->insert(point)) {
            return true;
        }
    }

    // This should never happen if the quadtree is implemented correctly
    return false;
}

std::vector<Point*> QuadtreeNode::queryRange(float2 rangeMin, float2 rangeMax) {
    std::vector<Point*> result;

    if (!intersects(rangeMin, rangeMax)) {
        return result;
    }

    for (const auto& point : points) {
        if (inRange(point->getPos(), rangeMin, rangeMax)) {
            result.push_back(point);
        }
    }

    if (isSubdivided()) {
        for (int i = 0; i < 4; ++i) {
            auto childResult = children[i]->queryRange(rangeMin, rangeMax);
            result.insert(result.end(), childResult.begin(), childResult.end());
        }
    }

    return result;
}

std::vector<Point*> QuadtreeNode::queryCircle(float2 queryCenter, float radius) {
    std::vector<Point*> result;

    if (!intersects(queryCenter - make_float2(radius, radius), queryCenter + make_float2(radius, radius))) {
        return result;
    }

    for (const auto& point : points) {
        if (distanceBetween(point->getPos(), queryCenter) <= radius) {
            result.push_back(point);
        }
    }

    if (isSubdivided()) {
        for (int i = 0; i < 4; ++i) {
            auto childResult = children[i]->queryCircle(queryCenter, radius);
            result.insert(result.end(), childResult.begin(), childResult.end());
        }
    }

    return result;
}

bool QuadtreeNode::inBoundary(float2 position) const {
    return position.x >= center.x - (halfDimension.x + looseBoundary) &&
           position.x <= center.x + (halfDimension.x + looseBoundary) &&
           position.y >= center.y - (halfDimension.y + looseBoundary) &&
           position.y <= center.y + (halfDimension.y + looseBoundary);
}

bool QuadtreeNode::isSubdivided() {
    return children[0] != nullptr;
}

void QuadtreeNode::subdivide() {
    float2 quarterDimension = make_float2(halfDimension.x / 2, halfDimension.y / 2);

    children[0] = std::make_unique<QuadtreeNode>(make_float2(center.x - quarterDimension.x, center.y - quarterDimension.y), quarterDimension, depth + 1);
    children[1] = std::make_unique<QuadtreeNode>(make_float2(center.x + quarterDimension.x, center.y - quarterDimension.y), quarterDimension, depth + 1);
    children[2] = std::make_unique<QuadtreeNode>(make_float2(center.x - quarterDimension.x, center.y + quarterDimension.y), quarterDimension, depth + 1);
    children[3] = std::make_unique<QuadtreeNode>(make_float2(center.x + quarterDimension.x, center.y + quarterDimension.y), quarterDimension, depth + 1);

    // Store existing points temporarily
    std::vector<Point*> tempPoints = std::move(points);

    // Clear the points vector of this node
    points.clear();

    // Reinsert the points into the subdivided tree
    for (const auto point : tempPoints) {
        insert(point);
    }
}

bool QuadtreeNode::intersects(float2 rangeMin, float2 rangeMax) const {
    return !(rangeMax.x < center.x - halfDimension.x ||
             rangeMin.x > center.x + halfDimension.x ||
             rangeMax.y < center.y - halfDimension.y ||
             rangeMin.y > center.y + halfDimension.y);
}

bool QuadtreeNode::inRange(float2 position, float2 rangeMin, float2 rangeMax) {
    return position.x >= rangeMin.x && position.x <= rangeMax.x &&
           position.y >= rangeMin.y && position.y <= rangeMax.y;
}

void QuadtreeNode::update(Quadtree* quadtree) {
    if (isSubdivided()) {
        for (auto &child: children) {
            child->update(quadtree);
        }
        return;
    }

    for (auto it = points.begin(); it != points.end();) {
        Point* point = *it;
        if (point->pos != point->prevPos) {
            if (!inBoundary(point->getPos())) {
                it = points.erase(it);
                quadtree->insert(point);
            } else {
                ++it;
            }
        } else {
            ++it;
        }
    }
}

bool QuadtreeNode::overlapsSearch(float2 position, float distanceSquared) {
    float dx = std::abs(center.x - position.x);
    float dy = std::abs(center.y - position.y);
    float distance = std::sqrt(distanceSquared);
    if (dx > halfDimension.x + distance) return false;
    if (dy > halfDimension.y + distance) return false;

    if (dx <= halfDimension.x) return true;
    if (dy <= halfDimension.y) return true;

    float cornerDistanceSquared = (dx - halfDimension.x) * (dx - halfDimension.x) +
                                  (dy - halfDimension.y) * (dy - halfDimension.y);

    return cornerDistanceSquared <= distanceSquared;
}

void QuadtreeNode::findNearestPoint(float2 position, Point*& nearestPoint, float& nearestDistanceSquared) {
    if (!overlapsSearch(position, nearestDistanceSquared)) return;

    // Check if this node is too far away
    float dx = std::abs(center.x - position.x) - halfDimension.x;
    float dy = std::abs(center.y - position.y) - halfDimension.y;
    dx = std::max(dx, 0.0f);
    dy = std::max(dy, 0.0f);
    if (dx * dx + dy * dy > nearestDistanceSquared) return;

    // Check points in this node
    for (const auto& point : points) {
        float2 d = position - point->getPos();
        float distSquared = sum(d*d);
        if (distSquared < nearestDistanceSquared) {
            nearestDistanceSquared = distSquared;
            nearestPoint = point;
        }
    }

    // Recursively search children
    if (isSubdivided()) {
        for (const auto& child : children) {
            child->findNearestPoint(position,nearestPoint, nearestDistanceSquared);
        }
    }
}

void QuadtreeNode::render(VertexManager& vertexManager) {
    vertexManager.addFloatRectOutline({center.x - halfDimension.x, center.y - halfDimension.y,
                                 halfDimension.x * 2, halfDimension.y * 2},
                               sf::Color(0, 255, 0, 100), 1);

    if (isSubdivided()) {
        for (auto &child: children) {
            child->render(vertexManager);
        }
    } else {
        vertexManager.addText(std::to_string(points.size()), {center.x, center.y}, 16, sf::Color(0, 255, 0));
    }
}

void QuadtreeNode::reset() {
    points.clear();
    // Delete the pointers to all child nodes
    for (int i = 0; i < 4; ++i) {
        if (children[i]) {
            children[i]->reset();
            children[i].reset();
        }
    }
}
