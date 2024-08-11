#ifndef UI_MANAGER
#define UI_MANAGER

#include "SFML/Graphics.hpp"
#include "vector"
#include "cmath"
#include "vector_types.h"
#include <modules/graphics/UI/screen.hpp>
#include "unordered_map"

class UIManager {
public:
    UIManager(sf::RenderWindow* window);
    void addScreen(const std::string& name, Screen* screen);
    void draw(sf::RenderTarget& target) const;
    void handleEvent(const sf::Event& event);
    void handleHover(const sf::Vector2f& position);
    void setCurrentScreen(const std::string& screen);
private:
    sf::RenderWindow* window;
    std::unordered_map<std::string, Screen*> screens;
    std::string currentScreen;

};

#endif