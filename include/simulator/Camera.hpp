#pragma once

#include <vector_types.h>
#include "SFML/Graphics.hpp"
#include "iostream"
#include "unordered_map"

class Camera {
public:
    Camera() {};
    Camera(sf::FloatRect startPos, sf::RenderWindow* window);

    void update(float deltaTime);
    void updateEvent(const sf::Event& event);
    float getZoom() const;
    bool isCircleVisible(const float2 &point, float r);
    sf::Vector2f getCoords(const sf::Vector2f& screenPos);
    void setBounds(const sf::FloatRect& bounds);

    sf::View getView();
    sf::View getWindowView();

private:
    std::unordered_map<sf::Keyboard::Key, bool> keyStates;
    sf::RenderWindow* window;
    sf::View view;
    sf::View windowView;
    float zoomLevel = 1.0f;
    float moveSpeed = 1000.0f;
    sf::Vector2f position;
    sf::FloatRect sceneBounds;

    void zoom(float delta, const sf::Vector2i& mousePos);
    void updateView();
    void constrainToBounds();
};