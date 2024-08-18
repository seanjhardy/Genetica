#include <SFML/Graphics.hpp>
#include <modules/graphics/fontManager.hpp>
#include <modules/graphics/UI/text.hpp>

Text::Text(const std::string& text,
           const std::string& styleString)
             : UIElement(styleString, ""){
    propertySetters["font-size"] = [this](const string& v) {
        fontSize = parseValue(v);
    };

    font = FontManager::get("russo");
    labelElement.setFont(*font);
    labelElement.setCharacterSize(24);
    labelElement.setString(text);
    labelElement.setFillColor(sf::Color::White);
    setStyle(style);
}

void Text::draw(sf::RenderTarget& target) const {
    target.draw(labelElement);
}

void Text::onLayout() {
    UIElement::onLayout();
    labelElement.setCharacterSize((int)fontSize);
    labelElement.setFillColor(sf::Color::White);
    labelElement.setOrigin(labelElement.getGlobalBounds().getSize() / 2.f + labelElement.getLocalBounds().getPosition());
    labelElement.setPosition(layout.getPosition() + (layout.getSize() / 2.f));
}

void Text::setText(const std::string& text) {
    labelElement.setString(text);
    labelElement.setOrigin(labelElement.getGlobalBounds().getSize() / 2.f + labelElement.getLocalBounds().getPosition());
    labelElement.setPosition(layout.getPosition() + (layout.getSize() / 2.f));
}