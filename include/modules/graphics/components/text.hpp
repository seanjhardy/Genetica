#ifndef LABEL
#define LABEL

#include "SFML/Graphics.hpp"
#include "modules/graphics/utils/UIElement.hpp"
#include "modules/graphics/fontManager.hpp"

class Text : public UIElement {
public:
    Text(const unordered_map<string, string>& properties, const std::string& text);
    void draw(sf::RenderTarget& target) const override;
    void onLayout() override;
    void setText(const std::string& text);

private:
    sf::Text labelElement;
    std::string text;
    sf::Font* font = FontManager::get("russo");
    float fontSize = 20;
};

#endif