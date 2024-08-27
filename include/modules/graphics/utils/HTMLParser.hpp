#include "UIElement.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <memory>
#include <stdexcept>

vector<UIElement*> parseHTMLString(const string& html);
UIElement* createElement(string& tag, unordered_map<string, string>& properties, const string& value);
string getTagName(const string& tagContent);
size_t findMatchingClosingTag(const string& html, const string& tag, size_t pos);
bool isSelfClosingTag(const string& tagContent);
unordered_map<string, string> parseProperties(const string& tagContent);