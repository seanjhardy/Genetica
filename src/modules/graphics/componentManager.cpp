#include <modules/graphics/componentManager.hpp>
#include <modules/graphics/utils/HTMLParser.hpp>
#include <functional>
#include <string>

using namespace std;

unordered_map<string, string> ComponentManager::components;

void ComponentManager::init() {
    std::unordered_map<std::string, std::string> componentMappings = {
      //Simulation Screen
      {"simulationScreen", "./assets/components/simulationScreen.xml"},
      {"ControlPanel", "./assets/components/controlPanel.xml"},
      {"SimulationTab", "./assets/components/simulationTab.xml"},
      {"LifeformTab", "./assets/components/lifeformTab.xml"},
      // Main menu
    };

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