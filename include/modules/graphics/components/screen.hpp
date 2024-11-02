#ifndef SCREEN
#define SCREEN

#include "SFML/Graphics.hpp"
#include "modules/graphics/utils/UIElement.hpp"
#include "vector"
#include <modules/graphics/components/view.hpp>

class Screen {
public:
    void addElement(UIElement* element);
    void addElementKey(UIElement* element);
    void draw(sf::RenderTarget& target);
    bool handleEvent(const sf::Event& event);
    void resize(const sf::Vector2u& size);
    bool update(float dt, const sf::Vector2f& position);
    void addFunction(const function<void()>& function);
    UIElement* getElement(const string& key);
    void reset();

private:
    std::vector<UIElement*> elements;
    std::unordered_map<string, UIElement*> keys;
    std::vector<function<void()>> functions;

};

#endif