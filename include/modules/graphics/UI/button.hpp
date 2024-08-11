#ifndef BUTTON
#define BUTTON

#include "modules/graphics/UI/utils/UIElement.hpp"
#include <modules/graphics/UI/utils/roundedRectangleShape.hpp>
#include <functional>
#include <unordered_map>
#include <string>

using namespace std;

class Button : public UIElement {
public:
    Button(const string& text,
           function<void()> onClick, const std::string& styleString = "", const std::string& styleOnHoverString = "");
    void draw(sf::RenderTarget& target) const override;
    void handleEvent(const sf::Event& event) override;
    void onLayout() override;
    void setOnClick(function<void()> onClick);

    sf::RoundedRectangleShape shape;
    sf::RoundedRectangleShape buttonShadow;
    sf::Shader* shader;
    Shadow shadow = Shadow(5, sf::Color(0, 0, 0, 50), 0, 3);

    sf::Color backgroundColor;
    sf::Color backgroundHoverColor;
    sf::Text buttonText;
    sf::Font* font;
    sf::Sprite icon;
    int fontSize = 24;
    function<void()> onClick;
};

#endif