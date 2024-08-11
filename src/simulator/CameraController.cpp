#include "simulator/CameraController.hpp"
#include "cmath"

CameraController::CameraController(sf::FloatRect bounds,
                                   sf::RenderWindow* window)
        : window(window),
          zoomLevel(1.0f),
          position(bounds.left + bounds.width/2,
                   bounds.top + bounds.height/2),
          moveSpeed(1000.0f),
          sceneBounds(bounds),
          zoomSpeed(1.0f) {
    view = window->getDefaultView();
    windowView = window->getDefaultView();
}

void CameraController::update(float deltaTime) {
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

void CameraController::updateEvent(const sf::Event& event) {
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

void CameraController::constrainToBounds() {
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

void CameraController::zoom(float delta, const sf::Vector2i& mousePos) {
    const float maxZoomLevel = 10.0f;

    // Calculate minimum zoom level to ensure the entire scene fits on screen with some extra space
    float sceneAspectRatio = sceneBounds.width / sceneBounds.height;
    float windowAspectRatio = window->getSize().x / static_cast<float>(window->getSize().y);

    float minZoomLevel;
    if (sceneAspectRatio > windowAspectRatio) {
        // Scene is wider relative to the window
        minZoomLevel = (sceneBounds.width * 1.1f) / window->getSize().x;
    } else {
        // Scene is taller relative to the window
        minZoomLevel = (sceneBounds.height * 1.1f) / window->getSize().y;
    }

    // Allow zooming out 5 times further than the minimum required to fit the scene
    minZoomLevel = std::max(minZoomLevel / 5.0f, 0.01f);  // Ensure minimum zoom is always positive

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

void CameraController::updateView() {
    sf::Vector2f windowSize(static_cast<float>(window->getSize().x), static_cast<float>(window->getSize().y));

    // Calculate the visible size considering the zoom level
    sf::Vector2f viewSize = windowSize / zoomLevel;
    // Set the view size and center
    view.setSize(viewSize);
    view.setCenter(position);
    windowView.setSize(windowSize);
    windowView.setCenter(windowSize / 2.0f);
}

sf::View CameraController::getView() {
    return view;
}

sf::View CameraController::getWindowView() {
    return windowView;
}