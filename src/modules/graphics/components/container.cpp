#include <modules/graphics/components/container.hpp>
#include <modules/graphics/functionManager.hpp>
#include <modules/graphics/cursorManager.hpp>
#include "vector"
#include <utility>

using namespace std;

Container::Container(const unordered_map<string, string>& properties, vector<UIElement*> children) :
    UIElement(properties, children) {
    styleSetters["background"] = [this](const string& v) {
        backgroundColor = parseColor(v);
    };
    styleSetters["flex-direction"] = [this](const string& v) {
        flexDirection = parseDirection(v);
    };
    styleSetters["align-row"] = [this](const string& v) {
        rowAlignment = parseAlignment(v);
    };
    styleSetters["align-col"] = [this](const string& v) {
        columnAlignment = parseAlignment(v);
    };
    styleSetters["gap"] = [this](const string& v) {
        gap = parseValue(v);
    };
    styleSetters["shadow"] = [this](const string& v) {
        shadow = parseShadow(v);
    };

    CursorManager::set(cursor, "pointer");

    propertySetters["onClick"] = [this](const string& v) {
        onClick = *FunctionManager::get(v);
    };

    children = std::move(children);
    setProperties(properties);
    restyle();
}

void Container::onLayout() {
    UIElement::onLayout();
    if (shadow.getColor() != sf::Color::Transparent) {
        shadowShape = sf::RoundedRectangleShape(layout.getSize() +
                                                 sf::Vector2f(shadow.getSize(), shadow.getSize()));
        shadowShape.setFillColor(shadow.getColor());
        shadowShape.setRadius(border.getRadius()[0]);
        shadowShape.setPosition(layout.getPosition() +
                                 sf::Vector2f(shadow.getOffset()[0] - shadow.getSize()/2,
                                              shadow.getOffset()[1] - shadow.getSize()/2));
    }
    shape.setFillColor(backgroundColor);
    shape.setOutlineColor(border.getColor());
    shape.setOutlineThickness(border.getStroke());
    shape.setRadius(border.getRadius()[0]);
    shape.setPosition(layout.getPosition());
    shape.setSize(layout.getSize());
    updateLayout();
    for (auto& child : children) {
        child->onLayout();
    }
}

void Container::updateLayout() {
    // Get all visible children
    int numChildren = std::count_if(children.begin(), children.end(), [](UIElement* x) {
        return x->visible;
    });
    if (numChildren == 0) return;

    sf::Vector2f containerSize = layout.getSize();

    float horizontalPadding = padding[0].getValue() + padding[2].getValue();
    float verticalPadding = padding[1].getValue() + padding[3].getValue();
    containerSize -= sf::Vector2f(horizontalPadding, verticalPadding);

    float availableMainSize = (flexDirection == Direction::Row) ? containerSize.x : containerSize.y;
    float availableCrossSize = (flexDirection == Direction::Row) ? containerSize.y : containerSize.x;

    availableMainSize -= gap * (numChildren - 1);

    // First pass: calculate sizes for pixel and percent items, and count flex items
    float totalFixedMainSize = 0;
    int flexItemCount = 0;
    float totalFlexGrow = 0;

    for (const auto& child : children) {
        if (!child->visible) continue;
        const Size& mainSize = (flexDirection == Direction::Row) ? child->width : child->height;
        switch (mainSize.getMode()) {
            case Size::Mode::Pixel:
                totalFixedMainSize += mainSize.getValue();
                break;
            case Size::Mode::Percent:
                totalFixedMainSize += availableMainSize * mainSize.getValue() / 100.0f;
                break;
            case Size::Mode::Flex:
                flexItemCount++;
                totalFlexGrow += max(0.0f, mainSize.getValue());
                break;
        }
    }

    float remainingMainSize = max(0.0f, availableMainSize - totalFixedMainSize);
    float flexUnit = (flexItemCount > 0) ? remainingMainSize / totalFlexGrow : 0;

    // Second pass: set sizes and positions
    float currentMainPos = (flexDirection == Direction::Row) ?
            layout.left + padding[0].getValue() :
            layout.top + padding[1].getValue();

    for (const auto& child : children) {
        UIElement* element = child;
        if (!child->visible) continue;
        const Size& mainSize = (flexDirection == Direction::Row) ? child->width : child->height;
        const Size& crossSize = (flexDirection == Direction::Row) ? child->height : child->width;

        float itemMainSize = 0;
        float itemCrossSize = 0;

        // Calculate main axis size
        switch (mainSize.getMode()) {
            case Size::Mode::Pixel:
                itemMainSize = mainSize.getValue();
                break;
            case Size::Mode::Percent:
                itemMainSize = availableMainSize * mainSize.getValue() / 100.0f;
                break;
            case Size::Mode::Flex:
                itemMainSize = flexUnit * max(0.0f, mainSize.getValue());
                break;
        }

        // Calculate cross axis size
        switch (crossSize.getMode()) {
            case Size::Mode::Pixel:
                itemCrossSize = crossSize.getValue();
                break;
            case Size::Mode::Percent:
                itemCrossSize = availableCrossSize * crossSize.getValue() / 100.0f;
                break;
            case Size::Mode::Flex:
                itemCrossSize = availableCrossSize;
                break;
        }

        // Set element size
        sf::Vector2f elementSize = (flexDirection == Direction::Row)
                                   ? sf::Vector2f(itemMainSize, itemCrossSize)
                                   : sf::Vector2f(itemCrossSize, itemMainSize);
        element->base_layout.width = elementSize.x;
        element->base_layout.height = elementSize.y;

        // Set element position
        Alignment& alignment = (flexDirection == Direction::Row) ? columnAlignment : rowAlignment;
        float crossPos = (flexDirection == Direction::Row) ?
                layout.top + padding[1].getValue() :
                layout.left + padding[0].getValue();
        switch (alignment) {
            case Alignment::Center:
                crossPos += (availableCrossSize - itemCrossSize) / 2;
                break;
            case Alignment::End:
                crossPos += availableCrossSize - itemCrossSize -
                        ((flexDirection == Direction::Row) ? padding[2].getValue() : padding[3].getValue());
                break;
            default:
                break;
        }

        sf::Vector2f elementPos = (flexDirection == Direction::Row)
                                  ? sf::Vector2f(currentMainPos, crossPos)
                                  : sf::Vector2f(crossPos, currentMainPos);
        element->base_layout.left = elementPos.x;
        element->base_layout.top = elementPos.y;
        currentMainPos += itemMainSize + gap;
    }

    // Apply main alignment
    float mainPadding = (flexDirection == Direction::Row)
            ? padding[0].getValue() + padding[2].getValue()
            : padding[1].getValue() + padding[3].getValue();
    float totalChildrenSize = currentMainPos - gap - mainPadding;
    float extraSpace = availableMainSize - totalChildrenSize;

    if (extraSpace > 0) {
        switch (columnAlignment) {
            case Alignment::Center:
                applyOffset(extraSpace / 2);
                break;
            case Alignment::End:
                applyOffset(extraSpace);
                break;
            case Alignment::SpaceBetween:
                distributeSpace(extraSpace, false, numChildren);
                break;
            case Alignment::SpaceAround:
                distributeSpace(extraSpace, true, numChildren);
                break;
            default:
                break;
        }
    }
}

void Container::applyOffset(float offset) {
    for (auto& child : children) {
        if (flexDirection == Direction::Row) {
            child->base_layout.left += offset;
        } else {
            child->base_layout.top += offset;
        }
    }
}

void Container::distributeSpace(float space, bool includeEnds, int numVisibleChildren) {
    int divisions = includeEnds ? numVisibleChildren + 1 : numVisibleChildren - 1;
    if (divisions <= 0) return;

    float gap = space / divisions;
    float currentOffset = includeEnds ? gap : 0;

    for (auto& child : children) {
        if (flexDirection == Direction::Row) {
            child->base_layout.left += currentOffset;
        } else {
            child->base_layout.top += currentOffset;
        }
        currentOffset += gap;
    }
}

void Container::draw(sf::RenderTarget& target) const {
    if (!visible) return;
    if (shadow.getColor() != sf::Color::Transparent) {
        target.draw(shadowShape);
    }
    target.draw(shape);
    for (const auto& child : children) {
        child->draw(target);
    }
}

bool Container::handleEvent(const sf::Event& event) {
    if (!visible) return false;
    for (auto& child : children) {
        bool consumed = child->handleEvent(event);
        if (consumed) return true;
    }
    if (event.type == sf::Event::MouseButtonPressed &&
        event.mouseButton.button == sf::Mouse::Left) {
        if (contains({(float)event.mouseButton.x, (float)event.mouseButton.y})
            && visible && allowClick) {
            if (onClick) {
                onClick();
            }
            // Consume the mouse click as long as it's within the bounds of the container
            return true;
        }
    }
    return false;
}

void Container::update(float dt, const sf::Vector2f& position) {
    UIElement::update(dt, position);
    for (auto& child : children) {
        child->update(dt, position);
    }
}
