#include <modules/graphics/UI/text.hpp>
#include <SFML/Graphics.hpp>
#include <modules/graphics/fontManager.hpp>
#include "modules/graphics/UI/utils/UIElement.hpp"

Label::Label(const std::string& text,
             const std::string& styleString)
             : UIElement(styleString, ""){
    propertySetters["font-size"] = [this](const string& v) {
        fontSize = parseValue(v);
    };

    font = FontManager::getFont("russo");
    textElement.setFont(*font);
    textElement.setCharacterSize(24);
    textElement.setString(text);
    textElement.setFillColor(sf::Color::White);
    setStyle(style);
}

void Label::draw(sf::RenderTarget& target) const {
    target.draw(textElement);
}

void Label::onLayout() {
    textElement.setCharacterSize((int)fontSize);
    textElement.setFillColor(sf::Color::White);
    textElement.setOrigin(textElement.getGlobalBounds().getSize() / 2.f + textElement.getLocalBounds().getPosition());
    textElement.setPosition(layout.getPosition() + (layout.getSize() / 2.f));
}

void Label::updateText(const std::string& text) {
    textElement.setString(text);
}