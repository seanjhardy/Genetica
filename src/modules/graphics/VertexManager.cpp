#include <SFML/Graphics.hpp>
#include <vector>
#include <array>
#include <modules/utils/vector_types.hpp>
#include <modules/graphics/vertexManager.hpp>
#include <modules/utils/print.hpp>
#include <modules/graphics/fontManager.hpp>
#include <modules/graphics/spriteManager.hpp>
#include <modules/graphics/shaderManager.hpp>
#include <modules/utils/operations.hpp>


VertexManager::VertexManager()
    : vertices(sf::PrimitiveType::Triangles), texturedVertices(sf::PrimitiveType::Triangles) {
    ShaderManager::get("texture")->setParameter("texture", sf::Shader::CurrentTexture);
    states.shader = ShaderManager::get("texture");
    states.texture = SpriteManager::getTexture("cellTexture");
    states.blendMode = sf::BlendAlpha;
}

void VertexManager::addTriangle(const float2& p1, const float2& p2, const float2& p3,
    const sf::Color& color) {
    vertices.append(sf::Vertex({ p1.x, p1.y }, color));
    vertices.append(sf::Vertex({ p2.x, p2.y }, color));
    vertices.append(sf::Vertex({ p3.x, p3.y }, color));
}

void VertexManager::addTexturedTriangle(const float2& p1, const float2& p2, const float2& p3,
    const sf::Color& color, const sf::FloatRect& bbox, float angle) {
    float textureWidth = 50;
    float textureHeight = 50;
    float textureAspect = textureWidth / textureHeight;

    // Adjust the scale factors to maintain aspect ratio
    float scaleX = textureAspect / textureWidth;
    float scaleY = (1.0f / textureAspect) / textureWidth;
    float2 scale = { scaleX, scaleY };

    float centerX = (bbox.left + bbox.width / 2);
    float centerY = (bbox.top + bbox.height / 2);
    float2 center = { centerX, centerY };

    float2 texCoord1 = rotate(scale * (p1 - center), -angle);
    float2 texCoord2 = rotate(scale * (p2 - center), -angle);
    float2 texCoord3 = rotate(scale * (p3 - center), -angle);

    texturedVertices.append(sf::Vertex({ p1.x, p1.y }, color, { texCoord1.x, texCoord1.y }));
    texturedVertices.append(sf::Vertex({ p2.x, p2.y }, color, { texCoord2.x, texCoord2.y }));
    texturedVertices.append(sf::Vertex({ p3.x, p3.y }, color, { texCoord3.x, texCoord3.y }));
}

void VertexManager::addCircle(const float2& center, float radius, const sf::Color& color, int maxPoints) {
    float angle = 0;
    int LOD = getCircleLOD(radius);
    if (LOD > maxPoints) LOD = maxPoints;
    for (int i = 0; i < LOD; ++i) {
        float angle2 = (i + 1.0f) * 2 * M_PI / LOD;
        addTriangle(center, center + vec(angle) * radius, center + vec(angle2) * radius, color);
        angle = angle2;
    }
}

void VertexManager::addRectangle(const float2& p1, const float2& p2, const float2& p3, const float2& p4,
    const sf::Color& color) {
    addTriangle(p1, p2, p3, color);
    addTriangle(p3, p4, p1, color);
}

void VertexManager::addFloatRect(const sf::FloatRect& rect, const sf::Color& color) {
    addTriangle({ rect.left, rect.top },
        { rect.left + rect.width, rect.top },
        { rect.left, rect.top + rect.height }, color);

    addTriangle({ rect.left + rect.width, rect.top },
        { rect.left, rect.top + rect.height },
        { rect.left + rect.width, rect.top + rect.height }, color);
}

void VertexManager::addFloatRectOutline(const sf::FloatRect& rect, const sf::Color& color, float thickness) {
    addLine({ rect.left, rect.top }, { rect.left + rect.width, rect.top }, color, thickness);
    addLine({ rect.left + rect.width, rect.top }, { rect.left + rect.width, rect.top + rect.height }, color, thickness);
    addLine({ rect.left + rect.width, rect.top + rect.height }, { rect.left, rect.top + rect.height }, color, thickness);
    addLine({ rect.left, rect.top + rect.height }, { rect.left, rect.top }, color, thickness);
}

void VertexManager::addPolygon(const std::vector<float2>& points, const sf::Color& color) {
    if (points.size() < 3) return;
    for (int i = 1; i < points.size() - 1; ++i) {
        addTriangle(points[0], points[i], points[i + 1], color);
    }
}

void VertexManager::addPolygon(const std::vector<Vertex>& points) {
    if (points.size() % 3 != 0) return;
    for (int i = 0; i < points.size() - 2; i += 3) {
        vertices.append(sf::Vertex({ points[i].pos.x, points[i].pos.y }, points[i].color));
        vertices.append(sf::Vertex({ points[i + 1].pos.x, points[i + 1].pos.y }, points[i + 1].color));
        vertices.append(sf::Vertex({ points[i + 2].pos.x, points[i + 2].pos.y }, points[i + 2].color));
    }
}

void VertexManager::addSegment(float2 p1, float2 p2, float r1, float r2, float angle, const sf::Color& color) {
    // Keep track of body polygon points
    std::array<float2, 4> polygon = { 0, 0, 0, 0 };
    int LOD1 = getCircleLOD(r1) / 2;
    int LOD2 = getCircleLOD(r2) / 2;

    sf::FloatRect bbox = {
        min(p1.x - r1, p2.x - r2),
        min(p1.y - r1, p2.y - r2),
        max(p1.x + r1, p2.x + r2) - min(p1.x - r1, p2.x - r2),
        max(p1.y + r1, p2.y + r2) - min(p1.y - r1, p2.y - r2)
    };

    // Create body
    float2 prevVertex = p1 + vec(angle + M_PI / 2) * r1;
    polygon[0] = prevVertex;
    for (int i = 0; i < LOD1; ++i) {
        float currentAngle = (i + 1) * M_PI / LOD1 + angle + M_PI / 2;
        float2 nextVertex = p1 + vec(currentAngle) * r1;
        addTexturedTriangle(p1, prevVertex, nextVertex, color, bbox, angle);
        prevVertex = nextVertex;
    }
    polygon[1] = prevVertex;

    // Second semicircle
    prevVertex = p2 + vec(angle - M_PI / 2) * r2;
    polygon[2] = prevVertex;
    for (int i = 0; i < LOD2; i++) {
        float currentAngle = (i + 1) * M_PI / LOD2 + angle + 3 * M_PI / 2;
        float2 nextVertex = p2 + vec(currentAngle) * r2;
        addTexturedTriangle(p2, prevVertex, nextVertex, color, bbox, angle);
        prevVertex = nextVertex;
    }
    polygon[3] = prevVertex;

    //Add polygon
    addTexturedTriangle(polygon[0], polygon[1], polygon[2], color, bbox, angle);
    // Second half of polygon
    addTexturedTriangle(polygon[2], polygon[3], polygon[0], color, bbox, angle);
}

void VertexManager::addLine(const float2 start, const float2 end, const sf::Color& color, const float thickness) {
    float angle = atan2f(end.y - start.y, end.x - start.x);
    float2 d = vec(angle + M_PI / 2) * thickness * 0.5f;

    addTriangle(start + d, end + d, end - d, color);
    addTriangle(end - d, start - d, start + d, color);
}

void VertexManager::addText(const std::string& text, const float2& pos,
    float size, const sf::Color& color, const TextAlignment alignment, const float outline) {
    sf::Text label;
    label.setFont(*FontManager::get("russo"));
    label.setString(text);
    label.setFillColor(color);
    label.setOutlineThickness(outline);
    label.setScale(size, size);

    sf::FloatRect bounds = label.getGlobalBounds();
    switch (alignment) {
    case TextAlignment::Left:
        label.setPosition(pos.x, pos.y);
        break;

    case TextAlignment::Center:
        label.setPosition(pos.x - bounds.width / 2, pos.y - bounds.height / 2);
        break;

    case TextAlignment::Right:
        label.setPosition(pos.x - bounds.width, pos.y);
        break;
    }
    labels.push_back(label);
}

void VertexManager::addSprite(const sf::Sprite& sprite) {
    sprites.push_back(sprite);
}

int VertexManager::getCircleLOD(float radius) {
    // Linearly interpolate between 3 points and 30 points based on apparent size from 10 pixels to over 100 pixels wide
    int value = 4 + 30 * std::clamp(getSizeInView(radius) / 100.0f, 0.0f, 1.0f);
    return value;
}

float VertexManager::getSizeInView(float size) {
    return camera->getZoom() * size;
}

void VertexManager::clear() {
    vertices.clear();
    texturedVertices.clear();
    labels.clear();
    sprites.clear();
}

void VertexManager::draw(sf::RenderTarget& target) {
    for (auto& sprite : sprites) {
        target.draw(sprite);
    }

    target.draw(vertices);
    target.draw(texturedVertices, states);

    for (auto& label : labels) {
        target.draw(label);
    }
    // Automatically clear the vertexArray after drawing
    clear();
}

void VertexManager::setCamera(Camera* cam) {
    this->camera = cam;
}
