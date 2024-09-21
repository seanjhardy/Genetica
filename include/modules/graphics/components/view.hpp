#ifndef VIEW
#define VIEW

#include <SFML/Graphics.hpp>
#include <modules/graphics/utils/roundedRectangleShape.hpp>
#include "modules/graphics/utils/UIElement.hpp"
#include "../utils/size.hpp"
#include "vector"
#include "memory"

class View : public UIElement {
public:
    explicit View(const unordered_map<string, string>& properties, std::vector<UIElement*> children = {});

    void draw(sf::RenderTarget& target) override;
    bool handleEvent(const sf::Event& event) override;
    void update(float dt, const sf::Vector2f& position) override;
    void onLayout() override;

    sf::RoundedRectangleShape shape;
    Direction flexDirection = Direction::Row;
    Alignment rowAlignment = Alignment::Start;
    Alignment columnAlignment = Alignment::Start;
    sf::Color backgroundColor = sf::Color::Transparent;
    sf::RoundedRectangleShape shadowShape;
    Shadow shadow = Shadow(0, sf::Color(0, 0, 0, 0), 0, 0);
    float gap = 0;
    function<void()> onClick;

private:
    void updateLayout();
    void applyOffset(float offset);
    void distributeSpace(float space, bool includeEnds, int numVisibleChildren);
};

#endif