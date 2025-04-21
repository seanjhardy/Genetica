////////////////////////////////////////////////////////////
//
// This software is provided 'as-is', without any express or implied warranty.
// In no event will the authors be held liable for any damages arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it freely,
// subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented;
// you must not claim that you wrote the original software.
// If you use this software in a product, an acknowledgment
// in the product documentation would be appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such,
// and must not be misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any source distribution.
//
////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
// Headers
////////////////////////////////////////////////////////////
#include "modules/graphics/utils/roundedRectangleShape.hpp"
#include <cmath>
#include <modules/utils/print.hpp>

namespace sf {
    RoundedRectangleShape::RoundedRectangleShape(const Vector2f& size, float radius, unsigned int cornerPointCount) {
        mySize = size;
        myRadius = {radius, radius, radius, radius};
        myCornerPointCount = cornerPointCount;
        update();
    }

    void RoundedRectangleShape::setSize(const Vector2f& size) {
        mySize = size;
        update();
    }

    const Vector2f& RoundedRectangleShape::getSize() const {
        return mySize;
    }

    void RoundedRectangleShape::setRadius(float radius) {
        myRadius = {radius, radius, radius, radius};
        update();
    }

    void RoundedRectangleShape::setRadius(sf::FloatRect radius) {
        myRadius = radius;
        update();
    }

    sf::FloatRect RoundedRectangleShape::getCornersRadius() const {
        return myRadius;
    }

    void RoundedRectangleShape::setCornerPointCount(unsigned int count) {
        myCornerPointCount = count;
        update();
    }

    std::size_t RoundedRectangleShape::getPointCount() const {
        return myCornerPointCount * 4;
    }

    sf::Vector2f RoundedRectangleShape::getPoint(std::size_t index) const {
        if (index >= myCornerPointCount * 4)
            return sf::Vector2f(0, 0);

        float deltaAngle = 90.0f / (myCornerPointCount - 1);
        sf::Vector2f center;
        unsigned int centerIndex = index / myCornerPointCount;
        static const float pi = 3.141592654f;

        float radius;
        if (index < myCornerPointCount) {
            radius = myRadius.left;
        }
        else if (index < myCornerPointCount * 2) {
            radius = myRadius.top;
        }
        else if (index < myCornerPointCount * 3) {
            radius = myRadius.width;
        }
        else {
            radius = myRadius.height;
        }

        switch (centerIndex) {
        case 0:
            center.x = mySize.x - radius;
            center.y = radius;
            break;
        case 1:
            center.x = radius;
            center.y = radius;
            break;
        case 2:
            center.x = radius;
            center.y = mySize.y - radius;
            break;
        case 3:
            center.x = mySize.x - radius;
            center.y = mySize.y - radius;
            break;
        }

        return {
            radius * cosf(deltaAngle * (index - centerIndex) * pi / 180) + center.x,
            -radius * sinf(deltaAngle * (index - centerIndex) * pi / 180) + center.y
        };
    }
} // namespace sf
