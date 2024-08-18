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
    void resize(const sf::Vector2u& size);
    void update(float dt, const sf::Vector2f& position);
    void addFunction(const function<void()>& function);
    std::vector<UIElement*> getElements();
    void reset();

private:
    std::vector<UIElement*> elements;
    std::vector<function<void()>> functions;
};

#endif