#ifndef IMAGE_ELEMENT
#define IMAGE_ELEMENT

#include "SFML/Graphics.hpp"
#include "modules/graphics/UI/utils/UIElement.hpp"

class ImageElement : public UIElement {
public:
    ImageElement(const std::string &imagePath,
                 const std::string& styleString,
                 const std::string& styleOnHoverString);
    void draw(sf::RenderTarget& target) const override;
    void onLayout() override;

private:
    sf::Texture texture;
    sf::Sprite sprite;
};

#endif