#include "modules/graphics/UI/screen.hpp"
#include "SFML/Graphics.hpp"
#include "vector"

void Screen::addElement(std::unique_ptr<UIElement> element) {
    elements.push_back(std::move(element));
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
