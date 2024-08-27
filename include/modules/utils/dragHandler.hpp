#ifndef DRAG_HANDLER
#define DRAG_HANDLER

#include <SFML/Graphics.hpp>
#include "modules/graphics/vertexManager.hpp"

class DragHandler {
private:
    enum class DragHandle {
        None,
        Left,
        Right,
        Top,
        Bottom,
        TopLeft,
        TopRight,
        BottomLeft,
        BottomRight
    };

    DragHandle dragHandle = DragHandle::None;
    sf::Vector2f lastMousePos;
    bool isDragging = false;

public:
    void handleEvent(const sf::Vector2f mousePos, const sf::Event& event);
    sf::FloatRect update(const sf::Vector2f& mousePos, const sf::FloatRect& bounds);
    void render(VertexManager &vertexManager, const sf::FloatRect& bounds);

    int horizontalDirection() const;
    int verticalDirection() const;
};
#endif