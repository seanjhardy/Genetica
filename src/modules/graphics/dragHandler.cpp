#include "modules/graphics/dragHandler.hpp"
#include "SFML/Graphics.hpp"
#include "simulator/simulator.hpp"
#include "modules/graphics/cursorManager.hpp"

void DragHandler::handleEvent(const sf::Event &event) {
    if (event.type == sf::Event::MouseButtonPressed) {
        if (event.mouseButton.button == sf::Mouse::Left) {
            if (dragHandle != DragHandle::None) {
                dragging = true;
            }
        }
    } else if (event.type == sf::Event::MouseButtonReleased) {
        if (event.mouseButton.button == sf::Mouse::Left) {
            dragging = false;
        }
    }
}

sf::FloatRect DragHandler::update(const sf::Vector2f& mousePos, const sf::FloatRect& bounds, float sensitivity) {
    sf::Vector2f delta = mousePos - lastMousePos;
    lastMousePos = mousePos;

    float left = abs(mousePos.x - bounds.left);
    float right = abs(mousePos.x - bounds.left - bounds.width);
    float top = abs(mousePos.y - bounds.top);
    float bottom = abs(mousePos.y - bounds.top - bounds.height);

    DragHandle newDragHandlePos;
    if (left < sensitivity * 2 && top < sensitivity * 2) {
        newDragHandlePos = DragHandle::TopLeft;
    } else if (right < sensitivity * 2 && top < sensitivity * 2) {
        newDragHandlePos = DragHandle::TopRight;
    } else if (left < sensitivity * 2 && bottom < sensitivity * 2) {
        newDragHandlePos = DragHandle::BottomLeft;
    } else if (right < sensitivity * 2 && bottom < sensitivity * 2) {
        newDragHandlePos = DragHandle::BottomRight;
    } else if (left < sensitivity && mousePos.y < bounds.top + bounds.height && mousePos.y > bounds.top) {
        newDragHandlePos = DragHandle::Left;
    } else if (right < sensitivity && mousePos.y < bounds.top + bounds.height && mousePos.y > bounds.top) {
        newDragHandlePos = DragHandle::Right;
    } else if (top < sensitivity && mousePos.x < bounds.left + bounds.width && mousePos.x > bounds.left) {
        newDragHandlePos = DragHandle::Top;
    } else if (bottom < sensitivity && mousePos.x < bounds.left + bounds.width && mousePos.x > bounds.left) {
        newDragHandlePos = DragHandle::Bottom;
    } else {
        newDragHandlePos = DragHandle::None;
    }
    if (newDragHandlePos != dragHandle && !dragging) {
        dragHandle = newDragHandlePos;
        if (dragHandle == DragHandle::None) {
            Simulator::get().getWindow().setMouseCursor(CursorManager::getDefault());
        } else if (dragHandle == DragHandle::Left || dragHandle == DragHandle::Right) {
            Simulator::get().getWindow().setMouseCursor(CursorManager::get("dragHorizontal"));
        } else if (dragHandle == DragHandle::Top || dragHandle == DragHandle::Bottom) {
            Simulator::get().getWindow().setMouseCursor(CursorManager::get("dragVertical"));
        } else if (dragHandle == DragHandle::BottomRight) {
            Simulator::get().getWindow().setMouseCursor(CursorManager::get("dragBottomRight"));
        } else if (dragHandle == DragHandle::TopRight) {
            Simulator::get().getWindow().setMouseCursor(CursorManager::get("dragTopRight"));
        } else if (dragHandle == DragHandle::BottomLeft) {
            Simulator::get().getWindow().setMouseCursor(CursorManager::get("dragBottomLeft"));
        } else {
            Simulator::get().getWindow().setMouseCursor(CursorManager::get("dragTopLeft"));
        }
    }

    if (dragging) {

        int horizontal = horizontalDirection();
        int vertical = verticalDirection();
        return {horizontal == -1 ? delta.x : 0,
                vertical == -1 ? delta.y : 0,
                horizontal * delta.x,
             vertical * delta.y};
    } else {
        return {};
    }
}

void DragHandler::render(VertexManager &vertexManager, const sf::FloatRect& bounds) {
    if (dragHandle == DragHandle::None) return;

    sf::Color highlight = sf::Color(0,255,0);

    if (horizontalDirection() == -1) {
        vertexManager.addLine({bounds.left, bounds.top}, {bounds.left, bounds.top + bounds.height}, highlight, 2);
    } if (horizontalDirection() == 1) {
        vertexManager.addLine({bounds.left + bounds.width, bounds.top}, {bounds.left + bounds.width, bounds.top + bounds.height}, highlight, 2);
    } if (verticalDirection() == -1) {
        vertexManager.addLine({bounds.left, bounds.top}, {bounds.left + bounds.width, bounds.top}, highlight, 2);
    } if (verticalDirection() == 1) {
        vertexManager.addLine({bounds.left, bounds.top + bounds.height}, {bounds.left + bounds.width, bounds.top + bounds.height}, highlight, 2);
    }
}

int DragHandler::horizontalDirection() const {
    if (dragHandle == DragHandle::Left || dragHandle == DragHandle::TopLeft || dragHandle == DragHandle::BottomLeft) {
        return -1;
    } else if (dragHandle == DragHandle::Right || dragHandle == DragHandle::TopRight || dragHandle == DragHandle::BottomRight) {
        return 1;
    }
    return 0;
}

int DragHandler::verticalDirection() const {
    if (dragHandle == DragHandle::Top || dragHandle == DragHandle::TopLeft || dragHandle == DragHandle::TopRight) {
        return -1;
    } else if (dragHandle == DragHandle::Bottom || dragHandle == DragHandle::BottomLeft || dragHandle == DragHandle::BottomRight) {
        return 1;
    }
    return 0;
}

bool DragHandler::isDragging() const {
    return dragging;
}