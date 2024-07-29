#ifndef VERTEX_MANAGER
#define VERTEX_MANAGER
#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <vector_types.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class VertexManager {
private:
    sf::VertexArray vertices;
    sf::RenderStates states;

public:
    VertexManager() : vertices(sf::PrimitiveType::Triangles) {}

    void addCircle(const float2& center, float radius, const sf::Color& color, int points=10) {
        float angle = 0;
        for (int i = 0; i < points; ++i) {
            float angle2 = (i + 1) * 2 * M_PI / points;
            vertices.append(sf::Vertex(sf::Vector2f(center.x, center.y), color));
            vertices.append(sf::Vertex(sf::Vector2f(center.x + std::cos(angle) * radius,
                                                    center.y + std::sin(angle) * radius), color));
            vertices.append(sf::Vertex(sf::Vector2f(center.x + std::cos(angle2) * radius,
                                                    center.y + std::sin(angle2) * radius), color));
            angle = angle2;
        }
    }

    void addRectangle(const float2& p1, const float2& p2, const float2& p3, const float2& p4, const sf::Color& color) {
        addTriangle(p1, p2, p3, color);
        addTriangle(p2, p3, p4, color);
    }

    void addTriangle(const float2& p1, const float2& p2, const float2& p3, const sf::Color& color) {
        vertices.append(sf::Vertex(sf::Vector2f(p1.x, p1.y), color));
        vertices.append(sf::Vertex(sf::Vector2f(p2.x, p2.y), color));
        vertices.append(sf::Vertex(sf::Vector2f(p3.x, p3.y), color));
    }

    void addPolygon(const std::vector<float2>& points, const sf::Color& color) {
        if (points.size() < 3) return;
        for (size_t i = 1; i < points.size() - 1; ++i) {
            vertices.append(sf::Vertex(sf::Vector2f(points[0].x, points[0].y), color));
            vertices.append(sf::Vertex(sf::Vector2f(points[i].x, points[i].y), color));
            vertices.append(sf::Vertex(sf::Vector2f(points[i + 1].x, points[i + 1].y), color));
        }
    }

    void clear() {
        vertices.clear();
    }

    void draw(sf::RenderTarget& target) const {
        target.draw(vertices, states);
    }
};
#endif