#include "SFML/Graphics.hpp"
#include "modules/graphics/UI/UIElement.hpp"
#include "modules/graphics/UI/button.hpp"
#include <functional>
#include <utility>

Button::Button(const sf::FloatRect& bounds, const std::string& text, auto onClick)
        : bounds(bounds), onClick(std::move(onClick)) {
    buttonShape.setSize(sf::Vector2f(bounds.width, bounds.height));
    buttonShape.setFillColor(sf::Color::Blue);

    if (!font.loadFromFile("arial.ttf")) {
        // Handle error
    }
    buttonText.setFont(font);
    buttonText.setString(text);
    buttonText.setCharacterSize(24);
    buttonText.setFillColor(sf::Color::White);
    buttonText.setPosition(bounds.left + 10, bounds.top + 10);
}

void Button::draw(sf::RenderTarget& target) const {
    target.draw(buttonShape);
    target.draw(buttonText);
}

void Button::handleEvent(const sf::Event& event) {
    if (event.type == sf::Event::MouseButtonPressed) {
        if (event.mouseButton.button == sf::Mouse::Left && contains({(float)event.mouseButton.x, (float)event.mouseButton.y})) {
            if (onClick) onClick();
        }
    }
}

bool Button::contains(const sf::Vector2f& point) const {
    return bounds.contains(point);
}