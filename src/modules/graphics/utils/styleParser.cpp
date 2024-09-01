#include "modules/graphics/utils/size.hpp"
#include "modules/graphics/utils/styleParser.hpp"
#include "modules/utils/stringUtils.hpp"
#include <SFML/Graphics.hpp>
#include <sstream>
#include <unordered_map>
#include <string>
#include <vector>
#include <regex>

using namespace std;

unordered_map<string, string> parseStyleString(const string& styleString) {
    unordered_map<string, string> result;
    istringstream ss(styleString);
    string token;
    regex keyValuePattern("(.+):\\s*([^;]+)");

    while (getline(ss, token, ';')) {
        smatch match;
        if (regex_search(token, match, keyValuePattern)) {
            if (result.contains(match[1].str())) {
                result[match[1].str()] = trim(match[2].str());
            } else {
                result.insert({trim(match[1].str()),
                               trim(match[2].str())});
            }
        }
    }
    return result;
}

Size parseSize(const string& value) {
    if (value.find("px") != string::npos) {
        return Size::Pixel(stof(value.substr(0, value.find("px"))));
    } else if (value.find('%') != string::npos) {
        return Size::Percent(stof(value.substr(0, value.find('%'))));
    }
    return Size::Pixel(100); // Default
}

float parseValue(const string& value) {
    if (value.find("px") != string::npos) {
        return stof(value.substr(0, value.find("px")));
    }
    return 0; // Default
}

//Takes in a string of format <value>px or <value>px <value>px <value>px <value>px
void parseMultiValue(const string& value, Size (&result)[4]) {
    std::istringstream ss(value);
    std::string token;
    std::vector<Size> values;
    while (ss >> token) {
        values.push_back(parseSize(token));
    }

    switch (values.size()) {
        case 1:
            result[0] = result[1] = result[2] = result[3] = values[0];
            break;
        case 2:
            result[0] = result[2] = values[0];
            result[1] = result[3] = values[1];
            break;
        case 3:
            result[0] = values[0];
            result[1] = result[3] = values[1];
            result[2] = values[2];
            break;
        default:
            result[0] = result[1] = result[2] = result[3] = Size::Pixel(0);
            break;
    }
}

sf::Color parseColor(const string& value) {
    if (value == "transparent") {
        return sf::Color(0, 0, 0, 0);
    }
    if (value.find("rgba") != string::npos) {
        regex rgbaPattern(R"(rgba\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\))");
        smatch match;
        if (regex_search(value, match, rgbaPattern)) {
            return sf::Color(stoi(match[1].str()), stoi(match[2].str()),
                             stoi(match[3].str()), stoi(match[4].str()));
        }
    } else if (value.find("rgb") != string::npos) {
        regex rgbPattern(R"(rgb\((\d+),\s*(\d+),\s*(\d+)(?:,\s*(\d+))?\))");
        smatch match;
        if (regex_search(value, match, rgbPattern)) {
            return sf::Color(stoi(match[1].str()), stoi(match[2].str()), stoi(match[3].str()));
        }
    } else if (value.find("#") != string::npos) {
        string hex = value.substr(1);
        if (hex.size() == 6) {
            return sf::Color(stoi(hex.substr(0, 2), nullptr, 16),
                             stoi(hex.substr(2, 2), nullptr, 16),
                             stoi(hex.substr(4, 2), nullptr, 16));
        }
    }
    return sf::Color::White; // Default
}

Border parseBorder(const string& borderStr) {
    istringstream iss(borderStr);
    vector<string> tokens;
    string token;

    while (iss >> token) {
        tokens.push_back(token);
    }

    int stroke = 0;
    sf::Color color;
    vector<int> radii;

    for (const auto& t : tokens) {
        if (t.find("px") != string::npos) {
            int value = stoi(t.substr(0, t.find("px")));
            if (stroke == 0) {
                stroke = value;
            } else {
                radii.push_back(value);
            }
        } else {
            color = parseColor(t);
        }
    }

    // Fill missing radii with the first value or 0
    while (radii.size() < 4) {
        radii.push_back(radii.empty() ? 0 : radii[0]);
    }

    return Border(stroke, color,
                  radii[0], radii[1],
                  radii[2], radii[3]);
}

Shadow parseShadow(const string& shadowStr) {
    if (shadowStr == "none") {
        return Shadow(0, sf::Color::Transparent);
    }
    istringstream iss(shadowStr);
    vector<string> tokens;
    string token;

    while (iss >> token) {
        tokens.push_back(token);
    }

    int blur = -1;
    sf::Color color;
    vector<int> offset;

    for (const auto& t : tokens) {
        if (t.find("px") != string::npos) {
            int value = stoi(t.substr(0, t.find("px")));
            if (blur == -1) {
                blur = value;
            } else {
                offset.push_back(value);
            }
        } else {
            color = parseColor(t);
        }
    }

    // Fill missing radii with the first value or 0
    while (offset.size() < 2) {
        offset.push_back(offset.empty() ? 0 : offset[0]);
    }
    return Shadow(blur, color, offset[0], offset[1]);
}

Direction parseDirection(const string& value) {
    if (value == "row") {
        return Direction::Row;
    } else if (value == "column") {
        return Direction::Column;
    }
    return Direction::Row; // Default
}

Alignment parseAlignment(const string& value) {
    if (value == "start") {
        return Alignment::Start;
    } else if (value == "center") {
        return Alignment::Center;
    } else if (value == "end") {
        return Alignment::End;
    } else if (value == "space-between") {
        return Alignment::SpaceBetween;
    } else if (value == "space-around") {
        return Alignment::SpaceAround;
    }
    return Alignment::Start; // Default
}

UITransform parseTransform(const std::string& value) {
    std::istringstream iss(value);
    std::vector<std::string> tokens;
    std::string token;

    while (iss >> token) {
        tokens.push_back(token);
    }

    for (const auto& t : tokens) {
        if (t.find("scale") != string::npos) {
            float scale = stof(t.substr(t.find('(') + 1, t.find(')') - t.find('(') - 1));
            return UITransform::Scale(scale);
        }
    }
}