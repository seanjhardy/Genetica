#include <SFML/Graphics.hpp>
#include "cmath"
#include <modules/graphics/UIManager.hpp>
#include <modules/utils/print.hpp>

void UIManager::addScreen(const std::string& name, Screen* screen) {
    screens.insert({name, screen});
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