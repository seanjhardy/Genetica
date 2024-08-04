#include "SFML/Graphics.hpp"
#include "modules/graphics/UI/UIElement.hpp"
#include "modules/graphics/UI/image.hpp"

ImageElement::ImageElement(const sf::FloatRect& bounds, const std::string& imagePath) : bounds(bounds) {
    if (!texture.loadFromFile(imagePath)) {
        // Handle error
    }
    sprite.setTexture(texture);
    sprite.setPosition(bounds.left, bounds.top);
    sprite.setScale(bounds.width / texture.getSize().x, bounds.height / texture.getSize().y);
}

void ImageElement::draw(sf::RenderTarget& target) const {
    target.draw(sprite);
}

void ImageElement::handleEvent(const sf::Event&) {
    // No event handling needed for images
}

bool ImageElement::contains(const sf::Vector2f& point) const  {
    return bounds.contains(point);
}