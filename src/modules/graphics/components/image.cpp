#include <SFML/Graphics.hpp>
#include <modules/graphics/utils/UIElement.hpp>
#include <modules/graphics/SpriteManager.hpp>
#include <modules/graphics/components/image.hpp>

ImageElement::ImageElement(const unordered_map<string, string>& properties)
                        : UIElement(properties, {}){
    styleSetters["resizeMode"] = [this](const string& value) {
        resizeMode = value;
    };
    styleSetters["image"] = [this](const string& value) {
        sprite = *SpriteManager::get(value);
    };

    setProperties(properties);
    restyle();
}

void ImageElement::onLayout() {
    UIElement::onLayout();
    sprite.setPosition(layout.left, layout.top);

    if (resizeMode == "contain") {
        float scale = min(layout.width / sprite.getTextureRect().getSize().x,
                          layout.height / sprite.getTextureRect().getSize().y);
        sprite.setScale(scale, scale);
        sprite.setPosition(layout.left + (layout.width - sprite.getTextureRect().getSize().x * scale) / 2,
                           layout.top + (layout.height - sprite.getTextureRect().getSize().y * scale) / 2);
    } else if (resizeMode == "cover") {
        float scale = max(layout.width / sprite.getTextureRect().getSize().x,
                          layout.height / sprite.getTextureRect().getSize().y);
        sprite.setScale(scale, scale);
        sprite.setPosition(layout.left + (layout.width - sprite.getTextureRect().getSize().x * scale) / 2,
                           layout.top + (layout.height - sprite.getTextureRect().getSize().y * scale) / 2);
    } else {
        sprite.setScale(layout.width / sprite.getTextureRect().getSize().x,
                        layout.height / sprite.getTextureRect().getSize().y);
    }
}

void ImageElement::draw(sf::RenderTarget& target) {
    if (!visible) return;
    target.draw(sprite);
}
