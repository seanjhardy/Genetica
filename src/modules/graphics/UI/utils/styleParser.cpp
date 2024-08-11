#include <modules/graphics/UI/utils/size.hpp>
#include <modules/graphics/UI/utils/styleParser.hpp>
#include <modules/graphics/UI/utils/shadow.hpp>
#include <sstream>
#include <unordered_map>
#include <string>
#include <vector>
#include <regex>

using namespace std;

// Function to trim leading and trailing spaces from a string
std::string trimSpaces(const std::string& str) {
    // Lambda function to check if a character is a whitespace
    auto isSpace = [](char ch) { return std::isspace(static_cast<unsigned char>(ch)); };

    // Find the first non-whitespace character
    auto start = std::find_if_not(str.begin(), str.end(), isSpace);

    // If the entire string is whitespace
    if (start == str.end()) {
        return "";
    }

    // Find the last non-whitespace character
    auto end = std::find_if_not(str.rbegin(), str.rend(), isSpace).base();

    // Create a new string with trimmed spaces
    return {start, end};
}

unordered_map<string, string> parseStyleString(const string& styleString) {
    unordered_map<string, string> result;
    istringstream ss(styleString);
    string token;
    regex keyValuePattern("(.+):\\s*([^;]+)");

    while (getline(ss, token, ';')) {
        smatch match;
        if (regex_search(token, match, keyValuePattern)) {
            result.insert({trimSpaces(match[1].str()),
                           trimSpaces(match[2].str())});
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
std::vector<Size> parseMultiValue(const string& value) {
    std::istringstream ss(value);
    std::string token;
    std::vector<Size> values;
    while (ss >> token) {
        values.push_back(parseSize(token));
    }

    switch (values.size()) {
        case 1:
            return std::vector<Size>({values[0], values[0], values[0], values[0]});
        case 2:
            return std::vector<Size>({values[0], values[1], values[0], values[1]});
        case 3:
            return std::vector<Size>({values[0], values[1], values[2], values[1]});
        default:
            return std::vector<Size>({Size::Pixel(0), Size::Pixel(0), Size::Pixel(0), Size::Pixel(0)});
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

Shadow parseShadow(const string& borderStr) {
    istringstream iss(borderStr);
    vector<string> tokens;
    string token;

    while (iss >> token) {
        tokens.push_back(token);
    }

    int blur = 0;
    sf::Color color;
    vector<int> offset;

    for (const auto& t : tokens) {
        if (t.find("px") != string::npos) {
            int value = stoi(t.substr(0, t.find("px")));
            if (blur == 0) {
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
