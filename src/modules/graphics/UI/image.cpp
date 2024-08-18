#include <SFML/Graphics.hpp>
#include <modules/graphics/UI/utils/UIElement.hpp>
#include <modules/graphics/SpriteManager.hpp>
#include <modules/graphics/UI/image.hpp>

ImageElement::ImageElement(const std::string &imagePath,
                           const std::string& styleString,
                           const std::string& styleOnHoverString) : UIElement(styleString, styleOnHoverString){
    sprite = *SpriteManager::get(imagePath);
}

void ImageElement::onLayout() {
    UIElement::onLayout();
    sprite.setPosition(layout.left, layout.top);
    sprite.setScale(layout.width / texture.getSize().x, layout.height / texture.getSize().y);
}

void ImageElement::draw(sf::RenderTarget& target) const {
    target.draw(sprite);
}
