#ifndef SCREEN
#define SCREEN

#include "SFML/Graphics.hpp"
#include <modules/graphics/UI/UIElement.hpp>
#include "vector"

class Screen {
public:
    void addElement(UIElement* element);
    void draw(sf::RenderTarget& target) const;
    void handleEvent(const sf::Event& event);

private:
    std::vector<UIElement*> elements;
};

#endif