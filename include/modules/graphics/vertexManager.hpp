#ifndef VERTEX_MANAGER
#define VERTEX_MANAGER
#include "SFML/Graphics.hpp"
#include "vector"
#include "cmath"
#include "vector_types.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class VertexManager {
private:
    sf::VertexArray vertices;
    sf::RenderStates states;

public:
    VertexManager() : vertices(sf::PrimitiveType::Triangles) {}

    void addCircle(const float2& center, float radius, const sf::Color& color, int points=10);
    void addRectangle(const float2& p1, const float2& p2, const float2& p3, const float2& p4, const sf::Color& color);
    void addFloatRect(const sf::FloatRect& rect, const sf::Color& color);
    void addTriangle(const float2& p1, const float2& p2, const float2& p3, const sf::Color& color);
    void addPolygon(const std::vector<float2>& points, const sf::Color& color);
    void addLine(const float2 start, const float2 end, const sf::Color& color, const float thickness = 1.0f);

    void clear();
    void draw(sf::RenderTarget& target);
};
#endif