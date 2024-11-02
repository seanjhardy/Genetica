#include <modules/graphics/componentManager.hpp>
#include <modules/graphics/utils/HTMLParser.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>

using namespace std;

unordered_map<string, string> ComponentManager::components;

void ComponentManager::init() {
    std::unordered_map<std::string, std::string> componentMappings;

    // Recursively fetch all .xml components
    for (const auto& entry : std::filesystem::recursive_directory_iterator("./assets/components/")) {
        if (entry.is_regular_file() && entry.path().extension() == ".xml") {
            std::string filename = entry.path().stem().string(); // Get filename without extension
            std::string filepath = entry.path().string();
            componentMappings[filename] = filepath;
        }
    }

    for (const auto& pair : componentMappings) {
        const std::string& key = pair.first;
        const std::string& filePath = pair.second;

        components[key] = readFile(filePath);
    }
}

bool ComponentManager::contains(const string& key) {
    return components.contains(key);
}

vector<UIElement*> ComponentManager::get(const string& key) {
    if (!components.contains(key)) {
        throw runtime_error("Invalid element key:" + key);
    }
    return parseHTMLString(components[key]);
}