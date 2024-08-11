#include <modules/graphics/UI/container.hpp>
#include "vector"
#include <utility>

Container::Container(const std::string& styleString, std::vector<UIElement*> childrenVector) :
    UIElement(styleString, "") {
    propertySetters["background"] = [this](const std::string& v) {
        backgroundColor = parseColor(v);
    };
    propertySetters["flex-direction"] = [this](const std::string& v) {
        flexDirection = parseDirection(v);
    };
    propertySetters["align-row"] = [this](const std::string& v) {
        rowAlignment = parseAlignment(v);
    };
    propertySetters["align-col"] = [this](const std::string& v) {
        columnAlignment = parseAlignment(v);
    };
    propertySetters["gap"] = [this](const std::string& v) {
        gap = parseValue(v);
    };
    
    setStyle(style);
    shape.setFillColor(backgroundColor);
    shape.setOutlineColor(border.getColor());
    shape.setOutlineThickness(border.getStroke());
    shape.setRadius(border.getRadius()[0]);
    children = std::move(childrenVector);
}

void Container::addChild(UIElement* child) {
    children.push_back(child);
}

void Container::onLayout() {
    shape.setPosition(layout.getPosition());
    shape.setSize(layout.getSize());
    updateLayout();
    for (auto& child : children) {
        child->onLayout();
    }
}

void Container::removeChild(UIElement* child) {
    auto it = std::find_if(children.begin(), children.end(),
                           [child](const auto& item) {
        return item == child;
    });
    if (it != children.end()) {
        children.erase(it);
        updateLayout();
    }
}

void Container::updateLayout() {
    if (children.empty()) return;

    sf::Vector2f containerSize = layout.getSize();

    float horizontalPadding = padding[0].getValue() + padding[2].getValue();
    float verticalPadding = padding[1].getValue() + padding[3].getValue();
    containerSize -= sf::Vector2f(horizontalPadding, verticalPadding);

    float availableMainSize = (flexDirection == Direction::Row) ? containerSize.x : containerSize.y;
    float availableCrossSize = (flexDirection == Direction::Row) ? containerSize.y : containerSize.x;

    availableMainSize -= gap * (children.size() - 1);

    // First pass: calculate sizes for pixel and percent items, and count flex items
    float totalFixedMainSize = 0;
    int flexItemCount = 0;
    float totalFlexGrow = 0;

    for (const auto& child : children) {
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
                totalFlexGrow += std::max(0.0f, mainSize.getValue());
                break;
        }
    }

    float remainingMainSize = std::max(0.0f, availableMainSize - totalFixedMainSize);
    float flexUnit = (flexItemCount > 0) ? remainingMainSize / totalFlexGrow : 0;

    // Second pass: set sizes and positions
    float currentMainPos = (flexDirection == Direction::Row) ?
            layout.left + padding[0].getValue() :
            layout.top + padding[1].getValue();

    for (const auto& child : children) {
        UIElement* element = child;
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
                itemMainSize = flexUnit * std::max(0.0f, mainSize.getValue());
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
        element->layout.width = elementSize.x;
        element->layout.height = elementSize.y;

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
        element->layout.left = elementPos.x;
        element->layout.top = elementPos.y;
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
                distributeSpace(extraSpace, false);
                break;
            case Alignment::SpaceAround:
                distributeSpace(extraSpace, true);
                break;
            default:
                break;
        }
    }
}

void Container::applyOffset(float offset) {
    for (auto& child : children) {
        if (flexDirection == Direction::Row) {
            child->layout.left += offset;
        } else {
            child->layout.top += offset;
        }
    }
}

void Container::distributeSpace(float space, bool includeEnds) {
    int divisions = includeEnds ? children.size() + 1 : children.size() - 1;
    if (divisions <= 0) return;

    float gap = space / divisions;
    float currentOffset = includeEnds ? gap : 0;

    for (auto& child : children) {
        if (flexDirection == Direction::Row) {
            child->layout.left += currentOffset;
        } else {
            child->layout.top += currentOffset;
        }
        currentOffset += gap;
    }
}

void Container::draw(sf::RenderTarget& target) const {
    target.draw(shape);
    for (const auto& child : children) {
        child->draw(target);
    }
}

void Container::handleEvent(const sf::Event& event) {
    for (auto& child : children) {
        child->handleEvent(event);
    }
}

void Container::handleHover(const sf::Vector2f& position) {
    UIElement::handleHover(position);
    for (auto& child : children) {
        child->handleHover(position);
    }
}
