#pragma once

#include "SFML/Graphics.hpp"
#include "iostream"
#include "unordered_map"

class CameraController {
public:
    CameraController(sf::FloatRect bounds, sf::RenderWindow* window);

    void update(float deltaTime);
    void updateEvent(const sf::Event& event);

    sf::View getView();
    sf::View getWindowView();

private:
    std::unordered_map<sf::Keyboard::Key, bool> keyStates;
    sf::RenderWindow* window;
    sf::View view;
    sf::View windowView;
    float zoomLevel;
    sf::Vector2f position;
    float moveSpeed;
    float zoomSpeed;
    sf::FloatRect sceneBounds;
    sf::FloatRect cameraBounds;

    void zoom(float delta, const sf::Vector2i& mousePos);
    void updateView();
    void constrainToBounds();
};