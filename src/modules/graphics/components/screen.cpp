#include <modules/graphics/components/screen.hpp>
#include <SFML/Graphics.hpp>

void Screen::addElement(UIElement* element) {
    element->onLayout();
    elements.push_back(element);
    addElementKey(element);
}

void Screen::addElementKey(UIElement* element) {
    if (!element->key.empty()) {
        keys[element->key] = element;
    }
    for (const auto& child : element->children) {
        addElementKey(child);
    }
}

UIElement* Screen::getElement(const std::string& key) {
    if (keys.contains(key)) {
        return keys[key];
    }
    return nullptr;
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

bool Screen::handleEvent(const sf::Event& event) {
    for (auto& element : elements) {
        bool consumed = element->handleEvent(event);
        if (consumed) return true;
    }
    return false;
}

void Screen::update(const float dt, const sf::Vector2f& position) {
    for (auto& element : elements) {
        if (element->contains(position)) {
            element->update(dt, position);
        }
    }
}

void Screen::reset() {
    elements.clear();
}

void Screen::addFunction(const function<void()>& function) {
    functions.push_back(function);
}
