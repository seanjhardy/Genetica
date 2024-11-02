#ifndef IMAGE_ELEMENT
#define IMAGE_ELEMENT

#include "SFML/Graphics.hpp"
#include "modules/graphics/utils/UIElement.hpp"

class ImageElement : public UIElement {
public:
    explicit ImageElement(const unordered_map<string, string>& properties);
    void draw(sf::RenderTarget& target) override;
    void onLayout() override;
    void getSprite();

    Size calculateWidth() override;
    Size calculateHeight() override;

private:
    sf::Sprite sprite;
    sf::Color tintColor = sf::Color::Transparent;
    string resizeMode = "contain";
};

#endif