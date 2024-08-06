#ifndef TEXT_ELEMENT
#define TEXT_ELEMENT

#include "SFML/Graphics.hpp"
#include <modules/graphics/UI/UIElement.hpp>

class TextElement : public UIElement {
public:
    TextElement(const sf::Vector2f& position, const std::string& text);
    void draw(sf::RenderTarget& target) const override;
    void handleEvent(const sf::Event&) override;
    bool contains(const sf::Vector2f&) const override;
    void setPosition(const sf::Vector2f& position);

private:
    sf::Text textElement;
    sf::Font font;
};

#endif