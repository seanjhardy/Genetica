#ifndef CONTAINER
#define CONTAINER

#include "SFML/Graphics.hpp"
#include "UIElement.hpp"
#include "utils/flexItem.hpp"
#include "utils/size.hpp"
#include "vector"
#include "memory"

class Container : public UIElement, public FlexItem {
public:
    enum class Direction { Row, Column };
    enum class Alignment { Start, Center, End, SpaceBetween, SpaceAround };

    explicit Container(Direction direction = Direction::Row,
                       Alignment mainAlignment = Alignment::Start,
                       Alignment crossAlignment = Alignment::Start);
    void addChild(UIElement* child, Size width, Size height);
    void removeChild(UIElement* child);

    void setDirection(Direction direction);
    void setMainAlignment(Alignment alignment);
    void setCrossAlignment(Alignment alignment);
    void setPadding(float padding);
    void setGap(float gap);
    void draw(sf::RenderTarget& target) const override;
    void handleEvent(const sf::Event& event) override;
    bool contains(const sf::Vector2f& point) const override;

private:
    void updateLayout();
    void applyOffset(float offset);
    void distributeSpace(float space, bool includeEnds);

    std::vector<std::unique_ptr<FlexItem>> m_children;
    Direction m_direction;
    Alignment m_mainAlignment;
    Alignment m_crossAlignment;
    float m_padding = 0;
    float m_gap = 0;
};

#endif