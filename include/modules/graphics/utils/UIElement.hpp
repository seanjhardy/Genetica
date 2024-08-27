#ifndef UI_ELEMENT
#define UI_ELEMENT

#include <vector_types.h>
#include "SFML/Graphics.hpp"
#include "size.hpp"
#include "modules/graphics/styleManager.hpp"
#include "border.hpp"
#include "transform.hpp"
#include "modules/utils/stringUtils.hpp"
#include <modules/graphics/cursorManager.hpp>
#include <unordered_map>
#include <functional>
#include "styleParser.hpp"
#include "modules/utils/print.hpp"
#include "modules/utils/mapUtils.hpp"

using namespace std;

class UIElement {
public:
    explicit UIElement(const unordered_map<string, string>& properties, const vector<UIElement*>& children = {});

    virtual void draw(sf::RenderTarget &target) const = 0;

    virtual bool handleEvent(const sf::Event &event) { return false; };

    virtual void update(float dt, const sf::Vector2f &position);

    virtual void onLayout();

    virtual void overrideProperty(const string& property, const string &s);

    virtual void restyle();

    [[nodiscard]] virtual bool contains(const sf::Vector2f &point) const {
        return layout.contains(point);
    };

    void setStyle(unordered_map<string, string> style);
    void setProperties(unordered_map<string, string> properties);

    unordered_map<string, string> classStyle;
    unordered_map<string, string> classStyleOnHover;
    unordered_map<string, string> style;
    unordered_map<string, string> styleOnHover;
    bool hovered = false;
    int depth;

    sf::FloatRect base_layout;
    sf::FloatRect layout;
    string key;
    Size width = Size::Flex(1);
    Size height = Size::Flex(1);
    Size margin[4] = {Size::Pixel(0), Size::Pixel(0),
                      Size::Pixel(0), Size::Pixel(0)};
    Size padding[4] = {Size::Pixel(0), Size::Pixel(0),
                       Size::Pixel(0), Size::Pixel(0)};
    Border border = Border(0.0f, sf::Color::Black);
    Transform transform = Transform::Scale(1);
    unique_ptr<Animation> animation;
    vector<UIElement*> children;
    unordered_map<string, UIElement*> keys;
    bool visible = true;
    bool allowClick = true;
    sf::Cursor cursor;

    unordered_map<string, function<void(const string &)>> styleSetters = {
      {"key",       [this](const string &v) { key = v; }},
      {"width",     [this](const string &v) { width = parseSize(v); }},
      {"height",    [this](const string &v) { height = parseSize(v); }},
      {"border",    [this](const string &v) { border = parseBorder(v); }},
      {"margin",    [this](const string &v) { parseMultiValue(v, margin); }},
      {"padding",   [this](const string &v) { parseMultiValue(v, padding); }},
      {"cursor",   [this](const string &v) { CursorManager::set(cursor, v); }},
      {"visible",   [this](const string &v) { visible = (v == "true"); }},
      {"allowClick",   [this](const string &v) { allowClick = (v == "true"); }},
      {"transform", [this](const string &v) {
          transform = parseTransform(v);
          auto update = [this](float progress) {
              transform.setCurrentValue(progress);
              onLayout();
          };
          animation = make_unique<Animation>(update);
          animation->completed = true;
      }}
    };

    unordered_map<string, function<void(const string &)>> propertySetters = {
      {"key", [this](const string& string) { key = string; }},
      {"class", [this](const string& string) {
        vector<std::string> classes = split(string);
        for (const auto& c : classes) {
            std::string classValue = Styles::get(c);
            overrideValues(classStyle, parseStyleString(classValue));
            std::string classOnHoverValue = Styles::get(c + ":hover");
            overrideValues(classStyleOnHover, parseStyleString(classOnHoverValue));
        }
      }},
      {"style", [this](const string& string) {
          overrideValues(style, parseStyleString(string));
          restyle();
      }},
      {"styleOnHover", [this](const string& string) {
          overrideValues(styleOnHover, parseStyleString(string));
          restyle();
      }},
    };
};

#endif