#ifndef CONTAINER
#define CONTAINER

#include <SFML/Graphics.hpp>
#include <modules/graphics/UI/utils/roundedRectangleShape.hpp>
#include "modules/graphics/UI/utils/UIElement.hpp"
#include "utils/size.hpp"
#include "vector"
#include "memory"

class Container : public UIElement {
public:
    explicit Container(const std::string& style = "", std::vector<UIElement*> children = {});
    void addChild(UIElement* child);
    void removeChild(UIElement* child);

    void draw(sf::RenderTarget& target) const override;
    void handleEvent(const sf::Event& event) override;
    void update(float dt, const sf::Vector2f& position) override;
    void onLayout() override;

    sf::RoundedRectangleShape shape;
    std::vector<UIElement*> children;
    Direction flexDirection = Direction::Row;
    Alignment rowAlignment = Alignment::Start;
    Alignment columnAlignment = Alignment::Start;
    sf::Color backgroundColor = sf::Color::Transparent;
    float gap = 0;

private:
    void updateLayout();
    void applyOffset(float offset);
    void distributeSpace(float space, bool includeEnds);
};

#endif