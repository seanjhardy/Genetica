#ifndef LABEL
#define LABEL

#include "SFML/Graphics.hpp"
#include "modules/graphics/UI/utils/UIElement.hpp"

class Text : public UIElement {
public:
    Text(const std::string& text, const std::string& styleString);
    void draw(sf::RenderTarget& target) const override;
    void onLayout() override;
    void setText(const std::string& text);

private:
    sf::Text labelElement;
    sf::Font* font;
    float fontSize = 20;
};

#endif