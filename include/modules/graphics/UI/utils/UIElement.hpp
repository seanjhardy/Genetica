#ifndef UI_ELEMENT
#define UI_ELEMENT

#include <vector_types.h>
#include "SFML/Graphics.hpp"
#include "size.hpp"
#include "modules/graphics/UI/utils/border.hpp"
#include <unordered_map>
#include <functional>
#include "styleParser.hpp"
#include "modules/utils/print.hpp"

using namespace std;

class UIElement {
public:
    UIElement(const std::string& style, const std::string& styleOnHover);
    virtual void draw(sf::RenderTarget &target) const = 0;

    virtual void handleEvent(const sf::Event &event) {};
    virtual void handleHover(const sf::Vector2f &position);

    virtual void onLayout() = 0;
    virtual void overrideStyle(const std::string& s);
    virtual void overrideStyleOnHover(const std::string& s);
    virtual void restyle();

    [[nodiscard]] virtual bool contains(const sf::Vector2f &point) const {
        return layout.contains(point);
    };

    void setStyle(std::unordered_map<std::string, std::string> style);

    std::unordered_map<std::string, std::string> style;
    std::unordered_map<std::string, std::string> styleOnHover;
    bool hovered = false;

    sf::FloatRect layout;
    std::string key;
    Size width = Size::Flex(1);
    Size height = Size::Flex(1);
    vector<Size> margin = {Size::Pixel(0), Size::Pixel(0),
                           Size::Pixel(0), Size::Pixel(0)};
    vector<Size> padding = {Size::Pixel(0), Size::Pixel(0),
                            Size::Pixel(0), Size::Pixel(0)};
    Border border = Border(0.0f, sf::Color::Black);

    std::unordered_map<string, function<void(const std::string &)>> propertySetters = {
            {"key",   [this](const std::string &v) { key = v; }},
            {"width",   [this](const std::string &v) { width = parseSize(v); }},
            {"height",  [this](const std::string &v) { height = parseSize(v); }},
            {"border",  [this](const std::string &v) { border = parseBorder(v); }},
            {"margin",  [this](const std::string &v) {
                margin = parseMultiValue(v);
            }},
            {"padding", [this](const std::string &v) {
                padding = parseMultiValue(v);
            }}
    };
};

#endif