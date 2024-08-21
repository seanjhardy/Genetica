#include <SFML/Graphics.hpp>
#include "vector"
#include "cmath"
#include "vector_types.h"
#include <modules/graphics/vertexManager.hpp>
#include <modules/utils/floatOps.hpp>
#include <modules/utils/fastMath.hpp>
#include <modules/utils/print.hpp>
#include <modules/graphics/fontManager.hpp>

#ifndef M_PI
    #define M_PI 3.14159
#endif


void VertexManager::addCircle(const float2& center, float radius, const sf::Color& color, int maxPoints) {
    float angle = 0;
    int LOD = getCircleLOD(radius);
    if (LOD > maxPoints) LOD = maxPoints;
    for (int i = 0; i < LOD; ++i) {
        float angle2 = (i + 1.0f) * 2 * M_PI / LOD;
        vertices.append(sf::Vertex({center.x, center.y}, color));
        vertices.append(sf::Vertex({center.x + cosf(angle) * radius,
                                                center.y + sinf(angle) * radius}, color));
        vertices.append(sf::Vertex({center.x + cosf(angle2) * radius,
                                                center.y + sinf(angle2) * radius}, color));
        angle = angle2;
    }
}

void VertexManager::addRectangle(const float2& p1, const float2& p2, const float2& p3, const float2& p4, const sf::Color& color) {
    addTriangle(p1, p2, p3, color);
    addTriangle(p3, p4, p1, color);
}

void VertexManager::addFloatRect(const sf::FloatRect& rect, const sf::Color& color) {
    addTriangle({rect.left, rect.top},
                {rect.left + rect.width, rect.top},
                {rect.left, rect.top + rect.height}, color);

    addTriangle({rect.left + rect.width, rect.top},
                {rect.left, rect.top + rect.height},
                {rect.left + rect.width, rect.top + rect.height}, color);
}

void VertexManager::addFloatRectOutline(const sf::FloatRect& rect, const sf::Color& color, float thickness) {
    addLine({rect.left, rect.top}, {rect.left + rect.width, rect.top}, color, thickness);
    addLine({rect.left + rect.width, rect.top}, {rect.left + rect.width, rect.top + rect.height}, color, thickness);
    addLine({rect.left + rect.width, rect.top + rect.height}, {rect.left, rect.top + rect.height}, color, thickness);
    addLine({rect.left, rect.top + rect.height}, {rect.left, rect.top}, color, thickness);
}

void VertexManager::addTriangle(const float2& p1, const float2& p2, const float2& p3, const sf::Color& color) {
    vertices.append(sf::Vertex({p1.x, p1.y}, color));
    vertices.append(sf::Vertex({p2.x, p2.y}, color));
    vertices.append(sf::Vertex({p3.x, p3.y}, color));
}

void VertexManager::addPolygon(const std::vector<float2>& points, const sf::Color& color) {
    if (points.size() < 3) return;
    for (int i = 1; i < points.size() - 1; ++i) {
        vertices.append(sf::Vertex({points[0].x, points[0].y}, color));
        vertices.append(sf::Vertex({points[i].x, points[i].y}, color));
        vertices.append(sf::Vertex({points[i + 1].x, points[i + 1].y}, color));
    }
}

void VertexManager::addSegment(float2 p1, float2 p2, float r1, float r2, float angle, const sf::Color& color) {
    // Keep track of body polygon points
    float2 polygon[4] = {{},{},{},{}};
    int LOD1 = getCircleLOD(r1) / 2;
    int LOD2 = getCircleLOD(r2) / 2;

    // Create body
    float2 prevVertex = p1 + vec(angle + M_PI/2) * r1;
    polygon[0] = prevVertex;
    for (int i = 0; i < LOD1; ++i) {
        float currentAngle = (i + 1) * M_PI / LOD1 + angle + M_PI/2;
        float2 nextVertex = p1 + vec(currentAngle) * r1;
        vertices.append(sf::Vertex({p1.x, p1.y}, color));
        vertices.append(sf::Vertex({prevVertex.x, prevVertex.y}, color));
        vertices.append(sf::Vertex({nextVertex.x, nextVertex.y}, color));
        prevVertex = nextVertex;
    }
    polygon[1] = prevVertex;
    // Second semicircle
    prevVertex = p2 + vec(angle + 3 *M_PI/2) * r2;
    polygon[2] = prevVertex;
    for (int i = 0; i < LOD2; i++) {
        float currentAngle = (i + 1) * M_PI / LOD2 + angle + 3*M_PI/2;
        float2 nextVertex = p2 + vec(currentAngle) * r1;
        vertices.append(sf::Vertex({p2.x, p2.y}, color));
        vertices.append(sf::Vertex({prevVertex.x, prevVertex.y}, color));
        vertices.append(sf::Vertex({nextVertex.x, nextVertex.y}, color));
        prevVertex = nextVertex;
    }
    polygon[3] = prevVertex;
    //Add polygon
    vertices.append(sf::Vertex({polygon[0].x, polygon[0].y}, color));
    vertices.append(sf::Vertex({polygon[1].x, polygon[1].y}, color));
    vertices.append(sf::Vertex({polygon[2].x, polygon[2].y}, color));
    // Second half of polygon
    vertices.append(sf::Vertex({polygon[2].x, polygon[2].y}, color));
    vertices.append(sf::Vertex({polygon[3].x, polygon[3].y}, color));
    vertices.append(sf::Vertex({polygon[0].x, polygon[0].y}, color));
}

void VertexManager::addLine(const float2 start, const float2 end, const sf::Color& color, const float thickness) {
    float angle = std::atan2(end.y - start.y, end.x - start.x);
    float2 d = vec(angle + M_PI/2) * thickness * 0.5f;

    vertices.append(sf::Vertex(sf::Vector2f(start.x + d.x, start.y + d.y), color));
    vertices.append(sf::Vertex(sf::Vector2f(end.x + d.x, end.y + d.y), color));
    vertices.append(sf::Vertex(sf::Vector2f(end.x - d.x, end.y - d.y), color));

    vertices.append(sf::Vertex(sf::Vector2f(end.x - d.x, end.y - d.y), color));
    vertices.append(sf::Vertex(sf::Vector2f(start.x - d.x, start.y - d.y), color));
    vertices.append(sf::Vertex(sf::Vector2f(start.x + d.x, start.y + d.y), color));
}

void VertexManager::addText(const std::string text, const float2& pos, float size, const sf::Color& color) {
    sf::Text label;
    label.setFont(*FontManager::get("russo"));
    label.setString(text);
    label.setCharacterSize(size);
    label.setFillColor(color);
    label.setPosition(pos.x, pos.y);
    labels.push_back(label);
}

int VertexManager::getCircleLOD(float radius) {
    // Linearly interpolate between 3 points and 30 points based on apparent size from 10 pixels to over 100 pixels wide
    int value = 4 + 30*std::clamp(getSizeInView(radius) / 100.0f, 0.0f, 1.0f);
    return value;
}

float VertexManager::getSizeInView(float size) {
    return camera->getZoom() * size;
}

void VertexManager::clear() {
    vertices.clear();
    labels.clear();
}

void VertexManager::draw(sf::RenderTarget& target) {
    target.draw(vertices, states);
    for (auto& label : labels) {
        target.draw(label);
    }
    // Automatically clear the vertexArray after drawing
    clear();
}

void VertexManager::setCamera(CameraController* cam) {
    this->camera = cam;
}
