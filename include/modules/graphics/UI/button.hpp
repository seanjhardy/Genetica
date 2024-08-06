#ifndef BUTTON
#define BUTTON

#include "SFML/Graphics.hpp"
#include <modules/graphics/UI/UIElement.hpp>
#include <functional>

class Button : public UIElement {
public:
    Button(const sf::FloatRect& bounds,
           const std::string& text,
           std::function<void()> onClick);
    void draw(sf::RenderTarget& target) const override;
    void handleEvent(const sf::Event& event) override;
    bool contains(const sf::Vector2f& point) const override;

private:
    sf::RectangleShape buttonShape;
    sf::Text buttonText;
    sf::Font font;
    sf::FloatRect bounds;
    std::function<void()> onClick;
};

#endif