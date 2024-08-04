#include "simulator/CameraController.hpp"
#include "iostream"
#include "cmath"

CameraController::CameraController(sf::FloatRect bounds,
                                   sf::RenderWindow& window)
        : window(window),
          zoomLevel(1.0f),
          position(bounds.left + bounds.width/2,
                   bounds.top + bounds.height/2),
          moveSpeed(1000.0f),
          sceneBounds(bounds),
          zoomSpeed(1.0f) {
    view = window.getDefaultView();
}

void CameraController::update(float deltaTime) {
    sf::Vector2f movement(0.0f, 0.0f);

    if (keyStates[sf::Keyboard::W]) {
        movement.y -= moveSpeed;
    }
    if (keyStates[sf::Keyboard::S]) {
        movement.y += moveSpeed;
    }
    if (keyStates[sf::Keyboard::A]) {
        movement.x -= moveSpeed;
    }
    if (keyStates[sf::Keyboard::D]) {
        movement.x += moveSpeed;
    }

    position += (movement * deltaTime / zoomLevel);

    constrainToBounds();
    updateView();
}

void CameraController::updateEvent(const sf::Event& event) {
    if (event.type == sf::Event::KeyPressed) {
        keyStates[event.key.code] = true;
    } else if (event.type == sf::Event::KeyReleased) {
        keyStates[event.key.code] = false;
    } else if (event.type == sf::Event::MouseWheelScrolled) {
        sf::Vector2i mousePosition = sf::Mouse::getPosition(window);
        zoom(event.mouseWheelScroll.delta, mousePosition);
        updateView();
    } else if (event.type == sf::Event::Resized) {
        updateView();
    }
}

void CameraController::constrainToBounds() {
    // Calculate the visible area considering the zoom level
    sf::Vector2f visibleSize = window.getDefaultView().getSize() * (1.0f / zoomLevel);
    sf::Vector2f halfVisibleSize = visibleSize * 0.5f;

    // Calculate the bounds of the camera view
    cameraBounds = sf::FloatRect(position.x - halfVisibleSize.x, position.y - halfVisibleSize.y,
                               visibleSize.x, visibleSize.y);

    // Constrain horizontally
    if (cameraBounds.left < sceneBounds.left) {
        position.x = sceneBounds.left + halfVisibleSize.x;
    } else if (cameraBounds.left + cameraBounds.width > sceneBounds.left + sceneBounds.width) {
        position.x = sceneBounds.left + sceneBounds.width - halfVisibleSize.x;
    }

    // Constrain vertically
    if (cameraBounds.top < sceneBounds.top) {
        position.y = sceneBounds.top + halfVisibleSize.y;
    } else if (cameraBounds.top + cameraBounds.height > sceneBounds.top + sceneBounds.height) {
        position.y = sceneBounds.top + sceneBounds.height - halfVisibleSize.y;
    }
}

void CameraController::zoom(float delta, const sf::Vector2i& mousePos) {
    const float maxZoomLevel = 10.0f;
    float minZoomX = sceneBounds.width / window.getDefaultView().getSize().x;
    float minZoomY = sceneBounds.height / window.getDefaultView().getSize().y;
    float minZoomLevel = std::max(minZoomX, minZoomY);

    // Convert mouse position from screen to world coordinates
    sf::Vector2f mouseWorldBeforeZoom = window.mapPixelToCoords(mousePos, view);

    // Calculate new zoom level
    float newZoomLevel = zoomLevel * std::pow(1.2f, delta);
    zoomLevel = std::clamp(newZoomLevel, minZoomLevel, maxZoomLevel);

    // Update the view to apply the new zoom level
    updateView();

    // Convert mouse position from screen to world coordinates after zooming
    sf::Vector2f mouseWorldAfterZoom = window.mapPixelToCoords(mousePos, view);

    // Calculate the movement needed to keep the mouse position stable
    // Update the camera's position to center on the mouse position after zoom
    position += mouseWorldBeforeZoom - mouseWorldAfterZoom;

    // Update the view with the new camera position
    updateView();
}

void CameraController::updateView() {
    sf::Vector2f windowSize(static_cast<float>(window.getSize().x), static_cast<float>(window.getSize().y));

    // Calculate the visible size considering the zoom level
    sf::Vector2f viewSize = windowSize / zoomLevel;

    // Calculate the actual view size (clamp to scene bounds if necessary)
    /*if (viewSize.y > sceneBounds.height) {
        viewSize.y = sceneBounds.height;
        viewSize.x = sceneBounds.height * windowSize.x / windowSize.y;
    }
    if (viewSize.x > sceneBounds.width) {
        viewSize.x = sceneBounds.width;
        viewSize.y = sceneBounds.width * windowSize.y / windowSize.x;
    }*/

    // Calculate the position of the view center (clamped to stay within scene bounds)
    sf::Vector2f viewCenter = position;
    viewCenter.x = std::max(viewCenter.x, sceneBounds.left + viewSize.x / 2.0f);
    viewCenter.x = std::min(viewCenter.x, sceneBounds.left + sceneBounds.width - viewSize.x / 2.0f);
    viewCenter.y = std::max(viewCenter.y, sceneBounds.top + viewSize.y / 2.0f);
    viewCenter.y = std::min(viewCenter.y, sceneBounds.top + sceneBounds.height - viewSize.y / 2.0f);

    // Set the view size and center
    view.setSize(viewSize);
    view.setCenter(viewCenter);

    // Apply the view
    window.setView(view);
}

sf::View CameraController::getView() {
    return view;
}