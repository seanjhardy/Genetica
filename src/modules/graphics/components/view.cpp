#include <modules/graphics/components/view.hpp>
#include <modules/graphics/functionManager.hpp>
#include "vector"
#include <utility>

using namespace std;

View::View(const unordered_map<string, string>& properties, vector<UIElement*> children) :
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

    std::vector<UIElement*> baseLayer;
    for (auto child : children) {
        if (child->absolute) {
            layers.push_back({child});
        }
        else {
            baseLayer.push_back(child);
        }
    }
    layers.push_back(baseLayer);

    setProperties(properties);
    restyle();
}

void View::onLayout() {
    UIElement::onLayout();
    sf::FloatRect clampedBorder = border.getRadius();
    float maxBorderRadius = std::min(layout.width / 2, layout.height / 2);
    clampedBorder.left = clamp(0, clampedBorder.left, maxBorderRadius);
    clampedBorder.top = clamp(0, clampedBorder.top, maxBorderRadius);
    clampedBorder.width = clamp(0, clampedBorder.width, maxBorderRadius);
    clampedBorder.height = clamp(0, clampedBorder.height, maxBorderRadius);

    if (shadow.getColor() != sf::Color::Transparent) {
        shadowShape = sf::RoundedRectangleShape(layout.getSize() +
            sf::Vector2f(shadow.getSize(), shadow.getSize()));
        shadowShape.setFillColor(shadow.getColor());
        shadowShape.setRadius(clampedBorder);
        shadowShape.setPosition(layout.getPosition() +
            sf::Vector2f(shadow.getOffset()[0] - shadow.getSize() / 2,
                         shadow.getOffset()[1] - shadow.getSize() / 2));
    }
    shape.setFillColor(backgroundColor);
    shape.setOutlineColor(border.getColor());
    shape.setOutlineThickness(border.getStroke());
    shape.setRadius(clampedBorder);
    shape.setPosition(layout.getPosition());
    shape.setSize(layout.getSize());
    updateLayout();
    for (auto& child : children) {
        child->onLayout();
    }
}


void View::draw(sf::RenderTarget& target) {
    if (!visible) return;
    if (shadow.getColor() != sf::Color::Transparent) {
        target.draw(shadowShape);
    }
    target.draw(shape);
    for (const auto& child : children) {
        child->draw(target);
    }
}

bool View::handleEvent(const sf::Event& event) {
    if (!visible) return false;
    for (auto& child : children) {
        bool consumed = child->handleEvent(event);
        if (consumed) return true;
    }
    if (event.type == sf::Event::MouseButtonPressed &&
        event.mouseButton.button == sf::Mouse::Left) {
        if (contains({(float)event.mouseButton.x, (float)event.mouseButton.y})
            && visible && allowClick) {
            if (animation) {
                animation->reset();
            }
            if (onClick) {
                onClick();
            }
            // Consume the mouse click as long as it's within the bounds of the container
            return true;
        }
    }
    return false;
}

bool View::update(float dt, const sf::Vector2f& position) {
    bool hovered = false;
    for (auto& child : children) {
        if (child->update(dt, position)) {
            hovered = true;
        };
    }
    if (UIElement::update(dt, position) && allowClick && visible) {
        hovered = true;
    }
    return hovered;
}

void View::updateLayout() {
    for (auto& layer : layers) {
        // Get all visible children
        int numChildren = std::count_if(layer.begin(), layer.end(), [](UIElement* child) {
            return child->visible;
        });
        if (numChildren == 0) continue;

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

        for (const auto& child : layer) {
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
        float totalSpaceTaken = flexItemCount > 0 ? availableMainSize : totalFixedMainSize;

        // Second pass: set sizes and positions
        float currentMainPos = (flexDirection == Direction::Row)
                                   ? layout.left + padding[0].getValue()
                                   : layout.top + padding[1].getValue();

        for (const auto& child : layer) {
            UIElement* element = child;
            if (!child->visible) continue;
            Size childWidth = child->width.getValue() == 0 ? child->calculateWidth() : child->width;
            Size childHeight = child->height.getValue() == 0 ? child->calculateHeight() : child->height;
            Size mainSize = (flexDirection == Direction::Row) ? childWidth : childHeight;
            Size crossSize = (flexDirection == Direction::Row) ? childHeight : childWidth;

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
            float crossPos = (flexDirection == Direction::Row)
                                 ? layout.top + padding[1].getValue()
                                 : layout.left + padding[0].getValue();

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
        float extraSpace = max(availableMainSize - totalSpaceTaken, 0.0f);

        if (extraSpace > 0) {
            Alignment mainAlignment = (flexDirection == Direction::Row) ? rowAlignment : columnAlignment;
            switch (mainAlignment) {
            case Alignment::Center:
                applyOffset(layer, extraSpace / 2);
                break;
            case Alignment::End:
                applyOffset(layer, extraSpace);
                break;
            case Alignment::SpaceBetween:
                distributeSpace(layer, extraSpace, false, numChildren);
                break;
            case Alignment::SpaceAround:
                distributeSpace(layer, extraSpace, true, numChildren);
                break;
            default:
                break;
            }
        }
    }
}

void View::applyOffset(const std::vector<UIElement*>& elements, float offset) {
    for (auto& child : elements) {
        if (!child->visible) continue;
        if (flexDirection == Direction::Row) {
            child->base_layout.left += offset;
        }
        else {
            child->base_layout.top += offset;
        }
    }
}

void View::distributeSpace(const std::vector<UIElement*>& elements, float space, bool includeEnds,
                           int numVisibleChildren) {
    int divisions = includeEnds ? numVisibleChildren + 1 : numVisibleChildren - 1;
    if (divisions <= 0) return;

    float gap = space / divisions;
    float currentOffset = includeEnds ? gap : 0;

    for (auto& child : elements) {
        if (!child->visible) continue;
        if (flexDirection == Direction::Row) {
            child->base_layout.left += currentOffset;
        }
        else {
            child->base_layout.top += currentOffset;
        }
        currentOffset += gap;
    }
}

Size View::calculateWidth() {
    float currentWidth = padding[0].getValue() + padding[2].getValue();
    for (auto child : children) {
        if (!child->visible || child->absolute) continue;
        currentWidth += child->calculateWidth().getValue() + gap;
    }
    return Size::Pixel(currentWidth - gap);
}

Size View::calculateHeight() {
    float currentHeight = padding[1].getValue() + padding[3].getValue();
    for (auto child : children) {
        if (!child->visible || child->absolute) continue;
        currentHeight += child->calculateHeight().getValue() + gap;
    }
    return Size::Pixel(currentHeight - gap);
}
