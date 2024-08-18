#ifndef UI_ELEMENT
#define UI_ELEMENT

#include <vector_types.h>
#include "SFML/Graphics.hpp"
#include "size.hpp"
#include "modules/graphics/UI/utils/border.hpp"
#include "modules/graphics/UI/utils/transform.hpp"
#include <unordered_map>
#include <functional>
#include "styleParser.hpp"
#include "modules/utils/print.hpp"

using namespace std;

class UIElement {
public:
    UIElement(const std::string &style, const std::string &styleOnHover);

    virtual void draw(sf::RenderTarget &target) const = 0;

    virtual void handleEvent(const sf::Event &event) {};

    virtual void update(float dt, const sf::Vector2f &position);

    virtual void onLayout();

    virtual void overrideStyle(const std::string &s);

    virtual void overrideStyleOnHover(const std::string &s);

    virtual void restyle();

    [[nodiscard]] virtual bool contains(const sf::Vector2f &point) const {
        return layout.contains(point);
    };

    void setStyle(std::unordered_map<std::string, std::string> style);

    std::unordered_map<std::string, std::string> style;
    std::unordered_map<std::string, std::string> styleOnHover;
    bool hovered = false;

    sf::FloatRect base_layout;
    sf::FloatRect layout;
    std::string key;
    Size width = Size::Flex(1);
    Size height = Size::Flex(1);
    Size margin[4] = {Size::Pixel(0), Size::Pixel(0),
                      Size::Pixel(0), Size::Pixel(0)};
    Size padding[4] = {Size::Pixel(0), Size::Pixel(0),
                       Size::Pixel(0), Size::Pixel(0)};
    Border border = Border(0.0f, sf::Color::Black);
    Transform transform = Transform::Scale(1);
    std::unique_ptr<Animation> animation;

    std::unordered_map<string, function<void(const std::string &)>> propertySetters = {
      {"key",       [this](const std::string &v) { key = v; }},
      {"width",     [this](const std::string &v) { width = parseSize(v); }},
      {"height",    [this](const std::string &v) { height = parseSize(v); }},
      {"border",    [this](const std::string &v) { border = parseBorder(v); }},
      {"margin",    [this](const std::string &v) {
          parseMultiValue(v, margin);
      }},
      {"padding",   [this](const std::string &v) {
          parseMultiValue(v, padding);
      }},
      {"transform", [this](const std::string &v) {
          transform = parseTransform(v);
          auto update = [this](float progress) {
              transform.setCurrentValue(progress);
              onLayout();
          };
          animation = std::make_unique<Animation>(update);
          animation->completed = true;
      }}
    };
};

#endif