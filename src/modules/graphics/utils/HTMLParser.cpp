#include <modules/graphics/utils/HTMLParser.hpp>
#include <modules/graphics/componentManager.hpp>
#include <modules/graphics/components/text.hpp>
#include <modules/graphics/components/view.hpp>
#include <modules/graphics/components/image.hpp>
#include <modules/graphics/components/viewport.hpp>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <stdexcept>

using namespace std;

vector<UIElement*> parseHTMLString(const string& html) {
    string content = stripHTML(html);
    vector<UIElement*> elements;
    size_t pos = 0;

    while (pos < content.length()) {
        size_t tagStart = content.find('<', pos);
        if (tagStart == string::npos) break;

        size_t tagEnd = content.find('>', tagStart);
        if (tagEnd == string::npos) break;

        string tagContent = content.substr(tagStart + 1, tagEnd - tagStart - 1);
        bool isSelfClosing = isSelfClosingTag(tagContent);
        if (isSelfClosing) {
            tagContent = tagContent.substr(0, tagContent.length() - 1);
        }
        string tag = getTagName(tagContent);
        unordered_map<string, string> properties = parseProperties(tagContent.substr(tag.length()));

        string innerContent;
        if (!isSelfClosing) {
            size_t closingTagStart = findMatchingClosingTag(content, tag, tagEnd + 1);
            size_t innerContentEnd = closingTagStart - (tag.length() + 3); // Offset for "</>"
            innerContent = trim(content.substr(tagEnd + 1, innerContentEnd - tagEnd));
            pos = closingTagStart + 1; // Skip past the closing tag
        } else {
            pos = tagEnd + 1;
        }

        UIElement* element = createElement(tag, properties, innerContent);
        elements.push_back(element);
    }

    return elements;
}

UIElement* createElement(string& tag, unordered_map<string, string>& properties, const string& innerContent) {
    if (tag == "Button" || tag == "View") {
        vector<UIElement*> children = parseHTMLString(innerContent);
        return new View(properties, children);
    } else if (tag == "Text") {
        return new Text(properties, innerContent);
    } else if (tag == "Image") {
        return new ImageElement(properties);
    } else if (tag == "Viewport") {
        return new Viewport(properties);
    } else {
        if (ComponentManager::contains(tag)) {
            return ComponentManager::get(tag)[0];
        } else {
            throw runtime_error("Invalid tag: " + tag);
        }
    }
}

string getTagName(const string& tagContent) {
    size_t spacePos = tagContent.find(' ');
    return spacePos != string::npos ? tagContent.substr(0, spacePos) : tagContent;
}

size_t findMatchingClosingTag(const string& html, const string& tag, size_t pos) {
    size_t depth = 1;
    size_t closingTagStart = pos;

    while (depth > 0) {
        size_t nextTagStart = html.find('<', closingTagStart);
        if (nextTagStart == string::npos) break;

        bool isClosingTag = (html[nextTagStart + 1] == '/');
        size_t nextTagEnd = html.find('>', nextTagStart);
        string nextTagContent = html.substr(nextTagStart + (isClosingTag ? 2 : 1), nextTagEnd - nextTagStart - (isClosingTag ? 2 : 1));
        string nextTag = getTagName(nextTagContent);
        bool isSelfClosing = isSelfClosingTag(nextTagContent);

        if (isClosingTag) {
            if (nextTag == tag) {
                depth--;
            }
        } else if (!isSelfClosing && nextTag == tag) {
            depth++;
        }
        closingTagStart = nextTagEnd + 1;
    }

    return closingTagStart - 1;
}

bool isSelfClosingTag(const string& tagContent) {
    return tagContent.back() == '/';
}

unordered_map<string, string> parseProperties(const string& tagContent) {
    unordered_map<string, string> properties;
    std::istringstream stream(tagContent);
    std::string line;

    while (std::getline(stream, line)) {
        std::istringstream lineStream(line);
        std::string key, value;

        while (lineStream) {
            // Read the key
            if (!std::getline(lineStream, key, '=')) {
                break;
            }

            // Read the value (enclosed in quotes)
            if (lineStream.get() == '"') {
                std::getline(lineStream, value, '"');
                properties[trim(key)] = value;
            }

            // Move past any whitespace
            lineStream >> std::ws;
        }
    }

    return properties;
}