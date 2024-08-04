#ifndef UI_MANAGER
#define UI_MANAGER

#include "SFML/Graphics.hpp"
#include "vector"
#include "cmath"
#include "vector_types.h"
#include "modules/graphics/UI/screen.hpp"
#include "unordered_map"

class UIManager {
public:
    void addScreen(const std::string& name, std::unique_ptr<Screen> screen);
    void draw(sf::RenderTarget& target) const;
    void handleEvent(const sf::Event& event);
    void setCurrentScreen(const std::string& screen);

private:
    std::unordered_map<std::string, std::unique_ptr<Screen>> screens;
    std::string currentScreen;

};

#endif