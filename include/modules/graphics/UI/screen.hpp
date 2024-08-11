#ifndef SCREEN
#define SCREEN

#include "SFML/Graphics.hpp"
#include "modules/graphics/UI/utils/UIElement.hpp"
#include "vector"
#include <modules/graphics/UI/container.hpp>

class Screen {
public:
    void addElement(UIElement* element);
    void draw(sf::RenderTarget& target) const;
    void handleEvent(const sf::Event& event);
    void update(const sf::Vector2u& size);
    void handleHover(const sf::Vector2f& position);
private:
    std::vector<UIElement*> elements;
};

#endif