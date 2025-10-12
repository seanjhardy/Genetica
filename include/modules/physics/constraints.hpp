#ifndef CONSTRAINTS_HPP
#define CONSTRAINTS_HPP

#include <modules/physics/point.hpp>
#include <SFML/Graphics.hpp>

// Constraint functions for physics simulation
void constrainDistance(Point& pointA, Point& pointB, double distance);
void constrainMinDistance(Point& pointA, Point& pointB, float minDistance);
void constrainAngle(Point& pointA, Point& pointB, float angleFromA, float angleFromB, float stiffness);
float constrainPosition(Point& point, sf::FloatRect bounds);
void checkCollisionCircleRec(Point& circle, Point& rect);

#endif // CONSTRAINTS_HPP

