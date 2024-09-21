#include <modules/graphics/utils/camera.hpp>
#include <cmath>
#include <vector_types.h>
#include <simulator/simulator.hpp>

Camera::Camera(sf::RenderTarget* target,
               sf::FloatRect* targetLayout,
               sf::FloatRect* bounds)
        : target(target),
          targetLayout(targetLayout){
    view = target->getDefaultView();

    if (bounds != nullptr) {
        position = {bounds->left + bounds->width/2,
                 bounds->top + bounds->height/2};
        zoomLevel = std::min(targetLayout->width / bounds->width,
                             targetLayout->height / bounds->height);
        sceneBounds = bounds;
    }
    updateView();
}

void Camera::update(float deltaTime) {
    if (locked) return;

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

void Camera::handleEvent(const sf::Event& event) {
    if (locked) return;

    if (event.type == sf::Event::KeyPressed) {
        keyStates[event.key.code] = true;
    } else if (event.type == sf::Event::KeyReleased) {
        keyStates[event.key.code] = false;
    } else if (event.type == sf::Event::MouseWheelScrolled) {
        sf::Vector2i mousePosition = sf::Mouse::getPosition(Simulator::get().getWindow());
        mousePosition -= sf::Vector2i(targetLayout->left, targetLayout->top);
        zoom(event.mouseWheelScroll.delta, mousePosition);
        updateView();
    } else if (event.type == sf::Event::Resized) {
        updateView();
    }
}

void Camera::constrainToBounds() {
    // Calculate the visible area size
    sf::Vector2f visibleSize = target->getDefaultView().getSize() / zoomLevel;

    // Calculate the maximum allowed distance from scene edges
    sf::Vector2f maxDistance = visibleSize / 8.0f;

    // Constrain the camera position
    position.x = std::clamp(position.x,
                            sceneBounds->left - maxDistance.x,
                            sceneBounds->left + sceneBounds->width + maxDistance.x);
    position.y = std::clamp(position.y,
                            sceneBounds->top - maxDistance.y,
                            sceneBounds->top + sceneBounds->height + maxDistance.y);
}

void Camera::zoom(float delta, const sf::Vector2i& mousePos) {
    // The minimum zoom level is the ratio of the screen dimension to twice the box dimension
    float minZoomLevel = std::min(target->getSize().x / (2.0f * sceneBounds->width),
                                  target->getSize().y / (2.0f * sceneBounds->height));

    // Convert mouse position from screen to world coordinates
    sf::Vector2f mouseWorldBeforeZoom = target->mapPixelToCoords(mousePos, view);

    // Calculate new zoom level
    float newZoomLevel = zoomLevel * std::pow(1.2f, delta);
    zoomLevel = std::max(newZoomLevel, minZoomLevel);

    // Update the view to apply the new zoom level
    updateView();

    // Convert mouse position from screen to world coordinates after zooming
    sf::Vector2f mouseWorldAfterZoom = target->mapPixelToCoords(mousePos, view);

    // Calculate the movement needed to keep the mouse position stable
    position += mouseWorldBeforeZoom - mouseWorldAfterZoom;

    // Constrain the camera position
    constrainToBounds();

    // Update the view with the new camera position
    updateView();
}

void Camera::updateView() {
    sf::Vector2f targetSize(target->getSize().x, target->getSize().y);

    // Calculate the visible size considering the zoom level
    sf::Vector2f viewSize = targetSize / zoomLevel;

    // Set the view size and center
    view.setSize(viewSize);
    view.setCenter(position);

    target->setView(view);
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

sf::Vector2f Camera::mapPixelToCoords(const sf::Vector2f& screenPos) {
    sf::Vector2f mouseRelativeToTarget = {screenPos.x - targetLayout->left, screenPos.y - targetLayout->top};
    return target->mapPixelToCoords(sf::Vector2i(mouseRelativeToTarget), view);
}

float Camera::getZoom() const {
    return zoomLevel;
}

void Camera::setZoom(float zoom) {
    zoomLevel = zoom;
    updateView();
}

sf::View Camera::getView() {
    return view;
}

void Camera::setBounds(sf::FloatRect* bounds) {
    if (sceneBounds == nullptr) {
        position = {bounds->left + bounds->width/2,
                    bounds->top + bounds->height/2};
    }
    sceneBounds = bounds;
}

void Camera::setView(const sf::View& view) {
    this->view = view;
}

void Camera::setLocked(bool locked) {
    this->locked = locked;
}

void Camera::setTargetLayout(sf::FloatRect* targetLayout) {
    this->targetLayout = targetLayout;
}

void Camera::setPosition(const sf::Vector2f& position) {
    this->position = position;
    updateView();
}