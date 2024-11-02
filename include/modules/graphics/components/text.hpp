#ifndef LABEL
#define LABEL

#include "SFML/Graphics.hpp"
#include "modules/graphics/utils/UIElement.hpp"
#include "modules/graphics/fontManager.hpp"

class Text : public UIElement {
public:
    Text(const unordered_map<string, string>& properties, const std::string& text);
    void draw(sf::RenderTarget& target) override;
    void onLayout() override;
    void setText(const std::string& text);

    Size calculateWidth() override;
    Size calculateHeight() override;

private:
    sf::Text labelElement;
    std::string text;
    sf::Font* font = FontManager::get("russo");
    TextAlignment textAlignment = TextAlignment::Left;
    float fontSize = 20;
    float outlineThickness = 0.0f;
};

#endif