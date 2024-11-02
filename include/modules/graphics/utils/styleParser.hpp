#ifndef STYLE_PARSER
#define STYLE_PARSER

#include "size.hpp"
#include "border.hpp"
#include "shadow.hpp"
#include "animation.hpp"
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <string>
#include <vector>
#include <regex>
#include <functional>
#include <memory>
#include "transform.hpp"

using namespace std;

enum class Direction { Row, Column };
enum class Alignment { Start, Center, End, SpaceBetween, SpaceAround };
enum class TextAlignment { Left, Center, Right };

unordered_map<string, string> parseStyleString(const string& styleString);

Size parseSize(const string& value);
float parseValue(const string& value);
Border parseBorder(const string& value);
Shadow parseShadow(const string& value);
sf::Color parseColor(const string& value);
void parseMultiValue(const string& value, Size (&result)[4]);
Direction parseDirection(const string& value);
Alignment parseAlignment(const string& value);
TextAlignment parseTextAlignment(const string& value);
UITransform parseTransform(const string& value);

#endif