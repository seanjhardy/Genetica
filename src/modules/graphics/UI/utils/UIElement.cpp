#include <unordered_map>
#include <string>
#include "modules/graphics/UI/utils/styleParser.hpp"
#include "modules/graphics/UI/utils/UIElement.hpp"

using namespace std;

UIElement::UIElement(const std::string &styleString, const std::string &styleOnHoverString) {
    style = parseStyleString(styleString);
    styleOnHover = parseStyleString(styleOnHoverString);
}

void UIElement::setStyle(std::unordered_map<std::string, std::string> styleProps) {
    for (const auto& [property, setter] : propertySetters) {
        if (styleProps.count(property)) {
            setter(styleProps[property]);
        }
    }
}

void UIElement::update(const float dt, const sf::Vector2f &position) {
    if (!hovered && contains(position)) {
        hovered = true;
        restyle();
    }
    if (hovered && !contains(position)) {
        hovered = false;
        restyle();
    }
    if (animation != nullptr && !animation->completed) {
        animation->update(dt);
    }
}

void UIElement::restyle() {
    if (hovered) {
        setStyle(style);
        setStyle(styleOnHover);
        if (animation) {
            animation->fromValue = 0;
            animation->toValue = 1;
            animation->reset();
        }
    } else {
        setStyle(style);
        if (animation) {
            animation->fromValue = 1;
            animation->toValue = 0;
            animation->reset();
        }
    }
    onLayout();
}

void UIElement::overrideStyle(const std::string& s) {
    // Style is a big style class like "background-color: red; border: 1px solid black; ..."
    // s is a subset of style overrides, e.g. "icon: 'icon.png'; background-color: blue;"
    // We want to check for matching keys and override the existing style using the parseStyleString
    auto styleProps = parseStyleString(s);
    for (const auto& [property, value] : styleProps) {
        style[property] = value;
    }
    restyle();
}

void UIElement::overrideStyleOnHover(const std::string& s) {
    auto styleProps = parseStyleString(s);
    for (const auto& [property, value] : styleProps) {
        styleOnHover[property] = value;
    }
    restyle();
}

void UIElement::onLayout() {
    layout = base_layout;
    layout.left += margin[0].getValue();
    layout.top += margin[1].getValue();
    layout.width -= margin[0].getValue() + margin[2].getValue();
    layout.height -= margin[1].getValue() + margin[3].getValue();

    if (transform.getValue() != 1) {
        float newWidth = base_layout.width * transform.getValue();
        float newHeight = base_layout.height * transform.getValue();
        layout.left += (layout.width - newWidth) / 2;
        layout.top += (layout.height - newHeight) / 2;
        layout.width = newWidth;
        layout.height = newHeight;
    }
};
