#include <unordered_map>
#include <string>
#include "modules/graphics/utils/styleParser.hpp"
#include "modules/graphics/utils/UIElement.hpp"
#include <simulator/simulator.hpp>

using namespace std;

UIElement::UIElement(const unordered_map<string, string>& properties, const vector<UIElement*>& children)
    : children(children) {
    CursorManager::set(cursor, "default");

    for (const auto& child : children) {
        if (child->key.empty()) {
            keys[child->key] = child;
        }
    }
}

void UIElement::setProperties(unordered_map<string, string> properties) {
    for (const auto& [property, setter] : propertySetters) {
        if (properties.count(property)) {
            setter(properties[property]);
        }
    }
}

void UIElement::setStyle(unordered_map<string, string> styleProps) {
    for (const auto& [property, setter] : styleSetters) {
        if (styleProps.count(property)) {
            setter(styleProps[property]);
        }
    }
}

void UIElement::update(const float dt, const sf::Vector2f &position) {
    if (!hovered && contains(position)) {
        hovered = true;
        //Simulator::get().setMouseCursor(depth, CursorManager::getDefault());
        restyle();
    }
    if (hovered && !contains(position)) {
        hovered = false;
        //Simulator::get().getWindow().setMouseCursor(cursor);
        restyle();
    }
    if (animation != nullptr && !animation->completed) {
        animation->update(dt);
    }
}

void UIElement::restyle() {
    setStyle(classStyle);
    setStyle(style);

    if (hovered) {
        setStyle(classStyleOnHover);
        setStyle(styleOnHover);
        if (animation) {
            animation->fromValue = 0;
            animation->toValue = 1;
            animation->reset();
        }
    } else {
        if (animation) {
            animation->fromValue = 1;
            animation->toValue = 0;
            animation->reset();
        }
    }
    onLayout();
}

void UIElement::overrideProperty(const string& property, const string& s) {
    propertySetters[property](s);
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
