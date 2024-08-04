#ifndef SCREEN
#define SCREEN

#include "SFML/Graphics.hpp"
#include "modules/graphics/UI/UIElement.hpp"
#include "vector"

class Screen {
public:
    void addElement(std::unique_ptr<UIElement> element);
    void draw(sf::RenderTarget& target) const;
    void handleEvent(const sf::Event& event);

private:
    std::vector<std::unique_ptr<UIElement>> elements;
};

#endif