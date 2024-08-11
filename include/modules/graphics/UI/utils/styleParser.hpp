#ifndef STYLE_PARSER
#define STYLE_PARSER

#include <modules/graphics/UI/utils/size.hpp>
#include <modules/graphics/UI/utils/border.hpp>
#include <modules/graphics/UI/utils/shadow.hpp>
#include <SFML/Graphics.hpp>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <string>
#include <vector>
#include <regex>
#include <functional>
#include <memory>

using namespace std;

enum class Direction { Row, Column };
enum class Alignment { Start, Center, End, SpaceBetween, SpaceAround };

unordered_map<string, string> parseStyleString(const string& styleString);

Size parseSize(const string& value);
float parseValue(const string& value);
Border parseBorder(const string& value);
Shadow parseShadow(const string& value);
sf::Color parseColor(const string& value);
vector<Size> parseMultiValue(const string& value);
Direction parseDirection(const string& value);
Alignment parseAlignment(const string& value);

#endif