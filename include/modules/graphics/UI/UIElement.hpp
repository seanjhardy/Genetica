#ifndef UI_ELEMENT
#define UI_ELEMENT

#include <vector_types.h>
#include "SFML/Graphics.hpp"

class UIElement {
public:
    virtual ~UIElement() = default;
    virtual void draw(sf::RenderTarget& target) const = 0;
    virtual void handleEvent(const sf::Event& event) = 0;
    [[nodiscard]] virtual bool contains(const sf::Vector2f& point) const = 0;

    [[nodiscard]] const sf::Vector2f& getPosition() const { return shape.getPosition(); }
    [[nodiscard]] const sf::Vector2f& getSize() const { return shape.getSize(); }

    virtual void setPosition(const sf::Vector2f& position) { shape.setPosition(position); }
    void setSize(const sf::Vector2f& size) { shape.setSize(size); }

protected:
    sf::RectangleShape shape;
};

#endif