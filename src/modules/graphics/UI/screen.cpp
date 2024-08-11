#include <modules/graphics/UI/screen.hpp>
#include <SFML/Graphics.hpp>

void Screen::addElement(UIElement* element) {
    element->onLayout();
    elements.push_back(element);
}

void Screen::draw(sf::RenderTarget& target) const {
    for (const auto& element : elements) {
        element->draw(target);
    }
}

void Screen::update(const sf::Vector2u& size) {
    for (auto& element : elements) {
        element->layout.width = size.x;
        element->layout.height = size.y;
        element->onLayout();
    }
}

void Screen::handleEvent(const sf::Event& event) {
    for (auto& element : elements) {
        element->handleEvent(event);
    }
}

void Screen::handleHover(const sf::Vector2f& position) {
    for (auto& element : elements) {
        if (element->contains(position)) {
            element->handleHover(position);
            return;
        }
    }
}
