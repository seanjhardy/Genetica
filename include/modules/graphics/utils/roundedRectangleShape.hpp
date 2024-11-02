#ifndef ROUNDEDRECTANGLESHAPE_HPP
#define ROUNDEDRECTANGLESHAPE_HPP

#include "SFML/Graphics/Shape.hpp"

namespace sf {
    class RoundedRectangleShape : public sf::Shape {
    public:
        explicit RoundedRectangleShape(const Vector2f &size = Vector2f(0, 0), float radius = 0,
                                       unsigned int cornerPointCount = 20);

        void setSize(const Vector2f &size);
        const Vector2f &getSize() const;
        void setRadius(float radius);
        void setRadius(sf::FloatRect radius);
        sf::FloatRect getCornersRadius() const;
        void setCornerPointCount(unsigned int count);
        virtual std::size_t getPointCount() const;
        virtual sf::Vector2f getPoint(std::size_t index) const;

    private:
        Vector2f mySize;
        sf::FloatRect myRadius;
        unsigned int myCornerPointCount;
    };
}
#endif

////////////////////////////////////////////////////////////
/// \class sf::RoundedRectangleShape
/// \ingroup graphics
///
/// This class inherits all the functions of sf::Transformable
/// (position, rotation, scale, bounds, ...) as well as the
/// functions of sf::Shape (outline, color, texture, ...).
///
/// Usage example:
/// \code
/// sf::RoundedRectangleShape roundedRectangle;
/// rectangle.setSize(sf::Vector2f(100, 50));
/// rectangle.setCornersRadius(5);
/// rectangle.setOutlineThickness(5);
/// rectangle.setPosition(10, 20);
/// ...
/// window.draw(rectangle);
/// \endcode
///
/// \see sf::Shape, sf::CircleShape, sf::ConvexShape
///
////////////////////////////////////////////////////////////