#include "SFML/Graphics.hpp"
#include "cmath"
#include "modules/graphics/UIManager.hpp"

void UIManager::addScreen(const std::string& name, std::unique_ptr<Screen> screen) {
    screens.insert({name, std::move(screen)});
}

void UIManager::draw(sf::RenderTarget& target) const {
    screens.at(currentScreen)->draw(target);
}

void UIManager::handleEvent(const sf::Event& event) {
    screens.at(currentScreen)->handleEvent(event);
}

void UIManager::setCurrentScreen(const std::string& screen) {
    if (screens.contains(screen)) {
        currentScreen = screen;
    }
}