#include <SFML/Graphics.hpp>
#include <modules/graphics/UIManager.hpp>

UIManager::UIManager(sf::RenderWindow* w) : window(w) {
}

void UIManager::addScreen(const std::string& name, Screen* screen) {
    screens.insert({name, screen});
    screens.at(name)->update(window->getSize());
}

void UIManager::draw(sf::RenderTarget& target) const {
    screens.at(currentScreen)->draw(target);
}

void UIManager::handleEvent(const sf::Event& event) {
    if (event.type == sf::Event::Resized) {
        screens.at(currentScreen)->update(window->getSize());
    }

    screens.at(currentScreen)->handleEvent(event);
}

void UIManager::handleHover(const sf::Vector2f& position) {
    screens.at(currentScreen)->handleHover(position);
}

void UIManager::setCurrentScreen(const std::string& screen) {
    if (screens.contains(screen)) {
        currentScreen = screen;
        screens.at(currentScreen)->update(window->getSize());
    }
}