#include <SFML/Graphics.hpp>
#include "modules/graphics/UI/utils/UIElement.hpp"
#include <modules/graphics/spriteManager.hpp>
#include <modules/graphics/fontManager.hpp>
#include <modules/graphics/shaderManager.hpp>
#include <modules/graphics/UI/button.hpp>
#include <functional>
#include <utility>

Button::Button(const std::string& text, std::function<void()> onClick,
               const std::string& styleString, const std::string& styleOnHoverString)
        : UIElement(styleString, styleOnHoverString), onClick(std::move(onClick)) {
    propertySetters["background"] = [this](const string& v) {
        backgroundColor = parseColor(v);
    };
    propertySetters["font-size"] = [this](const string& v) {
        fontSize = parseValue(v);
    };
    propertySetters["icon"] = [this](const string& v) {
        icon = *SpriteManager::getSprite(v);
    };
    propertySetters["shadow"] = [this](const string& v) {
        shadow = parseShadow(v);
    };

    font = FontManager::getFont("russo");
    buttonText = sf::Text(text, *font, fontSize);
    setStyle(style);
}

void Button::onLayout() {
    if (shadow.getColor() != sf::Color::Transparent) {
        buttonShadow = sf::RoundedRectangleShape(layout.getSize() +
          sf::Vector2f(shadow.getSize(), shadow.getSize()));
        buttonShadow.setFillColor(shadow.getColor());
        buttonShadow.setRadius(border.getRadius()[0]);
        buttonShadow.setPosition(layout.getPosition() +
                    sf::Vector2f(shadow.getOffset()[0] - shadow.getSize()/2,
                                 shadow.getOffset()[1] - shadow.getSize()/2));

        shader = ShaderManager::getShader("blur");
    }

    shape.setFillColor(backgroundColor);
    shape.setOutlineColor(border.getColor());
    shape.setOutlineThickness(border.getStroke());
    shape.setRadius(border.getRadius()[0]);
    shape.setPosition(layout.getPosition());
    shape.setSize(layout.getSize());

    buttonText.setFillColor(sf::Color::White);
    buttonText.setCharacterSize(fontSize);
    buttonText.setOrigin(buttonText.getGlobalBounds().getSize() / 2.f + buttonText.getLocalBounds().getPosition());
    buttonText.setPosition(layout.getPosition() + (layout.getSize() / 2.f));

    // Scale the icon to fit within the layout size (preserving aspect ratio), and place in the center of the layout rect
    if (icon.getTexture() != nullptr) {
        float scale = std::min((layout.getSize().x - padding[0].getValue() - padding[2].getValue()) / icon.getTexture()->getSize().x,
                               (layout.getSize().y - padding[1].getValue() - padding[3].getValue()) / icon.getTexture()->getSize().y);
        icon.setScale(scale, scale);
        sf::Vector2f iconSize = {icon.getTexture()->getSize().x * scale,
                                 icon.getTexture()->getSize().y * scale};
        icon.setPosition(layout.getPosition() + (layout.getSize() / 2.f) - iconSize / 2.f);
    }
}

void Button::draw(sf::RenderTarget& target) const {
    if (shadow.getColor() != sf::Color::Transparent) {
        //shader->setUniform("radius", shadow.getSize());
        //shader->setUniform("resolution", layout.getSize());
        target.draw(buttonShadow);
    }
    target.draw(shape);
    if (buttonText.getString() != "") {
        target.draw(buttonText);
    }
    if (icon.getTexture() != nullptr) {
        target.draw(icon);
    }
}

void Button::setOnClick(std::function<void()> onClick) {
    this->onClick = std::move(onClick);
}

void Button::handleEvent(const sf::Event& event) {
    if (event.type == sf::Event::MouseButtonPressed) {
        if (event.mouseButton.button == sf::Mouse::Left
            && contains({(float)event.mouseButton.x, (float)event.mouseButton.y})) {
            if (onClick) {
                onClick();
            }
        }
    }
}