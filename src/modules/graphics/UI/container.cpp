#include <SFML/Graphics.hpp>
#include <modules/graphics/UI/container.hpp>
#include "vector"
#include "memory"
#include <modules/utils/print.hpp>

Container::Container(Size width, Size height, Direction direction, Alignment mainAlignment, Alignment crossAlignment)
        : FlexItem(this, width, height),
        m_direction(direction), m_mainAlignment(mainAlignment), m_crossAlignment(crossAlignment) {}

void Container::addChild(UIElement* child, Size width, Size height) {
    m_children.emplace_back(std::make_unique<FlexItem>(child, width, height));
    updateLayout();
}

void Container::setParentSize(const sf::Vector2f& parentSize) {
    m_parentSize = parentSize;
    updateLayout();
}


void Container::removeChild(UIElement* child) {
    auto it = std::find_if(m_children.begin(), m_children.end(),
                           [child](const auto& item) { return item->getElement() == child; });
    if (it != m_children.end()) {
        m_children.erase(it);
        updateLayout();
    }
}

void Container::updateLayout() {
    if (m_children.empty()) return;

    sf::Vector2f containerSize = shape.getSize();
    float availableMainSize = (m_direction == Direction::Row ? containerSize.x : containerSize.y) - 2 * m_padding;
    float availableCrossSize = (m_direction == Direction::Row ? containerSize.y : containerSize.x) - 2 * m_padding;

    // First pass: calculate sizes for pixel and percent items, and count flex items
    float totalFixedMainSize = 0;
    int flexItemCount = 0;
    float totalFlexGrow = 0;

    for (const auto& child : m_children) {
        const Size& mainSize = (m_direction == Direction::Row) ? child->getWidth() : child->getHeight();
        switch (mainSize.getMode()) {
            case Size::Mode::Pixel:
                totalFixedMainSize += mainSize.getValue();
                break;
            case Size::Mode::Percent:
                totalFixedMainSize += availableMainSize * mainSize.getValue() / 100.0f;
                break;
            case Size::Mode::Flex:
                flexItemCount++;
                totalFlexGrow += mainSize.getValue();
                break;
        }
    }

    totalFixedMainSize += m_gap * (m_children.size() - 1);
    float remainingMainSize = std::max(0.0f, availableMainSize - totalFixedMainSize);
    float flexUnit = (flexItemCount > 0) ? remainingMainSize / totalFlexGrow : 0;

    // Second pass: set sizes and positions
    float currentMainPos = m_padding;
    for (const auto& child : m_children) {
        UIElement* element = child->getElement();
        const Size& mainSize = (m_direction == Direction::Row) ? child->getWidth() : child->getHeight();
        const Size& crossSize = (m_direction == Direction::Row) ? child->getHeight() : child->getWidth();

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
                itemMainSize = flexUnit * mainSize.getValue();
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
        sf::Vector2f elementSize = (m_direction == Direction::Row)
                                   ? sf::Vector2f(itemMainSize, itemCrossSize)
                                   : sf::Vector2f(itemCrossSize, itemMainSize);
        element->setSize(elementSize);

        // Set element position
        float crossPos = m_padding;
        switch (m_crossAlignment) {
            case Alignment::Center:
                crossPos += (availableCrossSize - itemCrossSize) / 2;
                break;
            case Alignment::End:
                crossPos += availableCrossSize - itemCrossSize;
                break;
            default:
                break;
        }

        sf::Vector2f elementPos = (m_direction == Direction::Row)
                                  ? sf::Vector2f(currentMainPos, crossPos)
                                  : sf::Vector2f(crossPos, currentMainPos);
        element->setPosition(elementPos);

        currentMainPos += itemMainSize + m_gap;
    }

    // Apply main alignment
    float totalChildrenSize = currentMainPos - m_gap - m_padding;
    float extraSpace = availableMainSize - totalChildrenSize;

    if (extraSpace > 0) {
        switch (m_mainAlignment) {
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
    for (auto& child : m_children) {
        sf::Vector2f pos = child->getElement()->getPosition();
        if (m_direction == Direction::Row) {
            pos.x += offset;
        } else {
            pos.y += offset;
        }
        child->getElement()->setPosition(pos);
    }
}

void Container::distributeSpace(float space, bool includeEnds) {
    int divisions = includeEnds ? m_children.size() + 1 : m_children.size() - 1;
    if (divisions <= 0) return;

    float gap = space / divisions;
    float currentOffset = includeEnds ? gap : 0;

    for (auto& child : m_children) {
        sf::Vector2f pos = child->getElement()->getPosition();
        if (m_direction == Direction::Row) {
            pos.x += currentOffset;
        } else {
            pos.y += currentOffset;
        }
        child->getElement()->setPosition(pos);
        currentOffset += gap;
    }
}

void Container::setDirection(Direction direction) {
    m_direction = direction;
    updateLayout();
}

void Container::setMainAlignment(Alignment alignment) {
    m_mainAlignment = alignment;
    updateLayout();
}

void Container::setCrossAlignment(Alignment alignment) {
    m_crossAlignment = alignment;
    updateLayout();
}

void Container::setPadding(float padding) {
    m_padding = padding;
    updateLayout();
}

void Container::setGap(float gap) {
    m_gap = gap;
    updateLayout();
}

void Container::draw(sf::RenderTarget& target) const {
    target.draw(shape);
    for (const auto& child : m_children) {
        child->getElement()->draw(target);
    }
}

void Container::handleEvent(const sf::Event& event) {
    for (auto& child : m_children) {
        child->getElement()->handleEvent(event);
    }
}

bool Container::contains(const sf::Vector2f& point) const {
    return shape.getGlobalBounds().contains(point);
}