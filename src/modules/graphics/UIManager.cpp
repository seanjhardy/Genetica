#include <SFML/Graphics.hpp>
#include <modules/graphics/UIManager.hpp>

UIManager::UIManager(sf::RenderWindow* w) : window(w) {
}

void UIManager::addScreen(const std::string& name, Screen* screen) {
    screens.insert({name, screen});
    screens.at(name)->resize(window->getSize());
}

void UIManager::draw(sf::RenderTarget& target) const {
    screens.at(currentScreen)->draw(target);
}

bool UIManager::handleEvent(const sf::Event& event) {
    if (event.type == sf::Event::Resized) {
        screens.at(currentScreen)->resize(window->getSize());
    }

    return screens.at(currentScreen)->handleEvent(event);
}

bool UIManager::update(float dt, const sf::Vector2f& position) {
    return screens.at(currentScreen)->update(dt, position);
}

void UIManager::setCurrentScreen(const std::string& screen) {
    if (screens.contains(screen)) {
        currentScreen = screen;
        screens.at(currentScreen)->resize(window->getSize());
    }
}

Screen* UIManager::getScreen(std::string name) {
    return screens.at(name);
}