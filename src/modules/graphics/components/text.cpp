#include <SFML/Graphics.hpp>
#include <modules/graphics/fontManager.hpp>
#include <modules/graphics/components/text.hpp>

using namespace std;


Text::Text(const unordered_map<string, string>& properties, const string& value)
             : UIElement(properties, {}){
    text = value;

    styleSetters["font-size"] = [this](const string& v) {
        fontSize = parseValue(v);
    };
    styleSetters["font"] = [this](const string& v) {
        font = FontManager::get(v);
    };
    styleSetters["outline"] = [this](const string& v) {
        outlineThickness = parseValue(v);
    };

    setProperties(properties);
    restyle();
}

void Text::draw(sf::RenderTarget& target) {
    if (!visible) return;
    target.draw(labelElement);
}

void Text::onLayout() {
    UIElement::onLayout();
    labelElement.setFont(*font);
    labelElement.setCharacterSize(24);
    labelElement.setString(text);
    labelElement.setCharacterSize((int)fontSize);
    labelElement.setFillColor(sf::Color::White);
    labelElement.setOutlineThickness(outlineThickness);
    labelElement.setOutlineColor(sf::Color::Black);
    labelElement.setOrigin(labelElement.getGlobalBounds().getSize() / 2.f + labelElement.getLocalBounds().getPosition());
    labelElement.setPosition(layout.getPosition() + (layout.getSize() / 2.f));
}

void Text::setText(const string& value) {
    text = value;
    labelElement.setString(text);
    labelElement.setOrigin(labelElement.getGlobalBounds().getSize() / 2.f + labelElement.getLocalBounds().getPosition());
    labelElement.setPosition(layout.getPosition() + (layout.getSize() / 2.f));
}