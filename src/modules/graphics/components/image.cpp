#include <SFML/Graphics.hpp>
#include <modules/graphics/utils/UIElement.hpp>
#include <modules/graphics/SpriteManager.hpp>
#include <modules/graphics/components/image.hpp>

ImageElement::ImageElement(const unordered_map<string, string>& properties)
    : UIElement(properties, {}) {
    styleSetters["resizeMode"] = [this](const string& value) {
        resizeMode = value;
    };
    styleSetters["tint"] = [this](const string& value) {
        tintColor = parseColor(value);
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
    }
    else if (resizeMode == "cover") {
        float scale = max(layout.width / sprite.getTextureRect().getSize().x,
                          layout.height / sprite.getTextureRect().getSize().y);
        sprite.setScale(scale, scale);
        sprite.setPosition(layout.left + (layout.width - sprite.getTextureRect().getSize().x * scale) / 2,
                           layout.top + (layout.height - sprite.getTextureRect().getSize().y * scale) / 2);
    }
    else {
        sprite.setScale(layout.width / sprite.getTextureRect().getSize().x,
                        layout.height / sprite.getTextureRect().getSize().y);
    }

    if (tintColor != sf::Color::Transparent) {
        sprite.setColor(tintColor);
    }
}

void ImageElement::draw(sf::RenderTarget& target) {
    if (!visible) return;
    target.draw(sprite);
}

Size ImageElement::calculateWidth() {
    return Size::Pixel(width.getValue() + padding[0].getValue() + padding[2].getValue());
}

Size ImageElement::calculateHeight() {
    return Size::Pixel(height.getValue() + padding[1].getValue() + padding[3].getValue());
}
