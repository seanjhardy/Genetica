#ifndef TEXT_ELEMENT
#define TEXT_ELEMENT

#include "SFML/Graphics.hpp"
#include "modules/graphics/UI/utils/UIElement.hpp"

class Label : public UIElement {
public:
    Label(const std::string& text, const std::string& styleString);
    void draw(sf::RenderTarget& target) const override;
    void onLayout() override;
    void updateText(const std::string& text);

private:
    sf::Text textElement;
    sf::Font* font;
    float fontSize = 20;
    function<void(string)> update;
};

#endif