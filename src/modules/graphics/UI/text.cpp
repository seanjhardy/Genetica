#include <modules/graphics/UI/text.hpp>
#include <SFML/Graphics.hpp>
#include <modules/graphics/UI/UIElement.hpp>

TextElement::TextElement(const sf::Vector2f& position, const std::string& text) {
    if (!font.loadFromFile("arial.ttf")) {
        // Handle error
    }
    textElement.setFont(font);
    textElement.setString(text);
    textElement.setCharacterSize(24);
    textElement.setFillColor(sf::Color::White);
    textElement.setPosition(position);
}

void TextElement::draw(sf::RenderTarget& target) const {
    target.draw(textElement);
}

void TextElement::handleEvent(const sf::Event&) {
    // No event handling needed for static text
}

bool TextElement::contains(const sf::Vector2f&) const {
    return false;
}

void TextElement::setPosition(const sf::Vector2f& position) {
    UIElement::setPosition(position);
    textElement.setPosition(position);
}