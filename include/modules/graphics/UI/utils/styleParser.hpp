#ifndef STYLE_PARSER
#define STYLE_PARSER

#include <modules/graphics/UI/utils/size.hpp>
#include <modules/graphics/UI/utils/border.hpp>
#include <modules/graphics/UI/utils/shadow.hpp>
#include <modules/graphics/UI/utils/animation.hpp>
#include <SFML/Graphics.hpp>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <string>
#include <vector>
#include <regex>
#include <functional>
#include <memory>

class Transform;

using namespace std;

enum class Direction { Row, Column };
enum class Alignment { Start, Center, End, SpaceBetween, SpaceAround };

unordered_map<string, string> parseStyleString(const string& styleString);

Size parseSize(const string& value);
float parseValue(const string& value);
Border parseBorder(const string& value);
Shadow parseShadow(const string& value);
sf::Color parseColor(const string& value);
void parseMultiValue(const string& value, Size (&result)[4]);
Direction parseDirection(const string& value);
Alignment parseAlignment(const string& value);
Transform parseTransform(const string& value);

#endif