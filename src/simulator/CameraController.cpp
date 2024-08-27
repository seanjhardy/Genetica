#include "simulator/Camera.hpp"
#include "cmath"
#include "vector_types.h"

Camera::Camera(sf::FloatRect bounds,
               sf::RenderWindow* window)
        : window(window),
          position(bounds.left + bounds.width/2,
                   bounds.top + bounds.height/2),
          sceneBounds(bounds){
    view = window->getDefaultView();
    windowView = window->getDefaultView();
}

void Camera::update(float deltaTime) {
    sf::Vector2f movement(0.0f, 0.0f);
    bool didUpdate = false;
    if (keyStates[sf::Keyboard::W]) {
        movement.y -= moveSpeed;
        didUpdate = true;
    }
    if (keyStates[sf::Keyboard::S]) {
        movement.y += moveSpeed;
        didUpdate = true;
    }
    if (keyStates[sf::Keyboard::A]) {
        movement.x -= moveSpeed;
        didUpdate = true;
    }
    if (keyStates[sf::Keyboard::D]) {
        movement.x += moveSpeed;
        didUpdate = true;
    }

    position += (movement * deltaTime / zoomLevel);

    if (didUpdate) {
        constrainToBounds();
        updateView();
    }
}

void Camera::updateEvent(const sf::Event& event) {
    if (event.type == sf::Event::KeyPressed) {
        keyStates[event.key.code] = true;
    } else if (event.type == sf::Event::KeyReleased) {
        keyStates[event.key.code] = false;
    } else if (event.type == sf::Event::MouseWheelScrolled) {
        sf::Vector2i mousePosition = sf::Mouse::getPosition(*window);
        zoom(event.mouseWheelScroll.delta, mousePosition);
        updateView();
    } else if (event.type == sf::Event::Resized) {
        updateView();
    }
}

void Camera::constrainToBounds() {
    // Calculate the visible area size
    sf::Vector2f visibleSize = window->getDefaultView().getSize() / zoomLevel;

    // Calculate the maximum allowed distance from scene edges
    sf::Vector2f maxDistance = visibleSize / 8.0f;

    // Constrain the camera position
    position.x = std::clamp(position.x,
                            sceneBounds.left - maxDistance.x,
                            sceneBounds.left + sceneBounds.width + maxDistance.x);
    position.y = std::clamp(position.y,
                            sceneBounds.top - maxDistance.y,
                            sceneBounds.top + sceneBounds.height + maxDistance.y);
}

void Camera::zoom(float delta, const sf::Vector2i& mousePos) {
    const float maxZoomLevel = 10.0f;

    float boxMaxDimension = std::max(sceneBounds.width, sceneBounds.height);
    float screenMaxDimension = (sceneBounds.width > sceneBounds.height) ? window->getSize().x : window->getSize().y;

    // The minimum zoom level is the ratio of the screen dimension to twice the box dimension
    float minZoomLevel = screenMaxDimension / (2.0f * boxMaxDimension);

    // Convert mouse position from screen to world coordinates
    sf::Vector2f mouseWorldBeforeZoom = window->mapPixelToCoords(mousePos, view);

    // Calculate new zoom level
    float newZoomLevel = zoomLevel * std::pow(1.2f, delta);
    zoomLevel = std::clamp(newZoomLevel, minZoomLevel, maxZoomLevel);

    // Update the view to apply the new zoom level
    updateView();

    // Convert mouse position from screen to world coordinates after zooming
    sf::Vector2f mouseWorldAfterZoom = window->mapPixelToCoords(mousePos, view);

    // Calculate the movement needed to keep the mouse position stable
    position += mouseWorldBeforeZoom - mouseWorldAfterZoom;

    // Constrain the camera position
    constrainToBounds();

    // Update the view with the new camera position
    updateView();
}

void Camera::updateView() {
    sf::Vector2f windowSize(static_cast<float>(window->getSize().x), static_cast<float>(window->getSize().y));

    // Calculate the visible size considering the zoom level
    sf::Vector2f viewSize = windowSize / zoomLevel;
    // Set the view size and center
    view.setSize(viewSize);
    view.setCenter(position);
    windowView.setSize(windowSize);
    windowView.setCenter(windowSize / 2.0f);
}

// Check if a point is within "r" distance of the bounds
bool Camera::isCircleVisible(const float2& point, float r) {
    // Get the current view bounds
    sf::Vector2f viewCenter = view.getCenter();
    sf::Vector2f viewSize = view.getSize();

    // Calculate the bounds of the view, shrunk by r
    float left = viewCenter.x - viewSize.x / 2.0f - r;
    float right = viewCenter.x + viewSize.x / 2.0f + r;
    float top = viewCenter.y - viewSize.y / 2.0f - r;
    float bottom = viewCenter.y + viewSize.y / 2.0f + r;

    // Check if the point is within these bounds
    return (point.x >= left && point.x <= right &&
            point.y >= top && point.y <= bottom);
}

sf::Vector2f Camera::getCoords(const sf::Vector2f& screenSpacePos) {
    return window->mapPixelToCoords(sf::Vector2i(screenSpacePos), view);
}

float Camera::getZoom() const {
    return zoomLevel;
}

sf::View Camera::getView() {
    return view;
}

sf::View Camera::getWindowView() {
    return windowView;
}

void Camera::setBounds(const sf::FloatRect& bounds) {
    sceneBounds = bounds;
}