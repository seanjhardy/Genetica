#ifndef IMAGE_ELEMENT
#define IMAGE_ELEMENT

#include "SFML/Graphics.hpp"
#include "modules/graphics/UI/UIElement.hpp"

class ImageElement : public UIElement {
public:
    ImageElement(const sf::FloatRect& bounds, const std::string& imagePath);
    void draw(sf::RenderTarget& target) const override;
    void handleEvent(const sf::Event&) override;
    bool contains(const sf::Vector2f& point) const override;

private:
    sf::Texture texture;
    sf::Sprite sprite;
    sf::FloatRect bounds;
};

#endif