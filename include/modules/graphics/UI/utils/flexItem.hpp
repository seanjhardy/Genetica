#ifndef FLEX_ITEM
#define FLEX_ITEM

#include "../UIElement.hpp"
#include "size.hpp"

class FlexItem {
public:
    FlexItem(UIElement* element, Size width, Size height)
            : m_element(element), m_width(width), m_height(height) {}

    [[nodiscard]] UIElement* getElement() const { return m_element; }
    [[nodiscard]] const Size& getWidth() const { return m_width; }
    [[nodiscard]] const Size& getHeight() const { return m_height; }

private:
    UIElement* m_element;
    Size m_width;
    Size m_height;
};

#endif