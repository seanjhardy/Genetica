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

private:
    sf::Sprite sprite;
    string resizeMode = "contain";
};

#endif