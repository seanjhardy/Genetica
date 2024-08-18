#include <modules/graphics/UI/screen.hpp>
#include <SFML/Graphics.hpp>

void Screen::addElement(UIElement* element) {
    element->onLayout();
    elements.push_back(element);
}

void Screen::draw(sf::RenderTarget& target) const {
    for (const auto& function : functions) {
        function();
    }
    for (const auto& element : elements) {
        element->draw(target);
    }
}

void Screen::resize(const sf::Vector2u& size) {
    for (auto& element : elements) {
        element->base_layout.width = size.x;
        element->base_layout.height = size.y;
        element->onLayout();
    }
}

void Screen::handleEvent(const sf::Event& event) {
    for (auto& element : elements) {
        element->handleEvent(event);
    }
}

void Screen::update(const float dt, const sf::Vector2f& position) {
    for (auto& element : elements) {
        if (element->contains(position)) {
            element->update(dt, position);
            return;
        }
    }
}

std::vector<UIElement*> Screen::getElements() {
    return elements;
}

void Screen::reset() {
    elements.clear();
}

void Screen::addFunction(const function<void()>& function) {
    functions.push_back(function);
}
