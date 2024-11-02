#ifndef DRAG_HANDLER
#define DRAG_HANDLER

#include "SFML/Graphics.hpp"
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
    bool dragging = false;

public:
    void handleEvent(const sf::Event &event);
    sf::FloatRect update(const sf::Vector2f& mousePos, const sf::FloatRect& bounds, float sensitivity = 15.0f);
    void render(VertexManager &vertexManager, const sf::FloatRect& bounds);
    void reset();

    [[nodiscard]] int horizontalDirection() const;
    [[nodiscard]] int verticalDirection() const;

    [[nodiscard]] bool isDragging() const;
};
#endif