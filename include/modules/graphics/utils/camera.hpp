#ifndef CAMERA
#define CAMERA

#include <vector_types.h>
#include "SFML/Graphics.hpp"
#include "unordered_map"

class Camera {
public:
    Camera() {};
    Camera(sf::RenderTarget* target, sf::FloatRect* targetLayout, sf::FloatRect* bounds = nullptr);

    void update(float deltaTime);
    void handleEvent(const sf::Event& event);
    float getZoom() const;
    bool isCircleVisible(const float2 &point, float r);
    sf::Vector2f mapPixelToCoords(const sf::Vector2f& screenPos);
    void setBounds(sf::FloatRect* bounds);
    void setView(const sf::View& view);
    void setZoom(float zoom);
    void setLocked(bool locked);
    void setTargetLayout(sf::FloatRect* targetLayout);
    void setPosition(const sf::Vector2f& position);

    sf::View getView();

    void updateView();

private:
    std::unordered_map<sf::Keyboard::Key, bool> keyStates;
    sf::RenderTarget* target{};
    sf::RenderWindow* window{};
    sf::View view;
    float zoomLevel = 1.0f;
    float moveSpeed = 1000.0f;
    bool locked = false;
    sf::Vector2f position;
    sf::FloatRect* sceneBounds{};
    sf::FloatRect* targetLayout{};

    void zoom(float delta, const sf::Vector2i& mousePos);

    void constrainToBounds();
};

#endif