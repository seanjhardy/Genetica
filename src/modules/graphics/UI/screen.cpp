#include <modules/graphics/UI/screen.hpp>
#include <SFML/Graphics.hpp>

void Screen::addElement(UIElement* element) {
    elements.push_back(element);
}

void Screen::draw(sf::RenderTarget& target) const {
    for (const auto& element : elements) {
        element->draw(target);
    }
}

void Screen::handleEvent(const sf::Event& event) {
    for (auto& element : elements) {
        element->handleEvent(event);
    }
}
