#ifndef COMPONENT_MANAGER
#define COMPONENT_MANAGER

#include <unordered_map>
#include <string>
#include <SFML/Graphics.hpp>
#include <modules/graphics/utils/UIElement.hpp>
#include <functional>

using namespace std;

class ComponentManager {
public:
    static void init();
    static vector<UIElement*> get(const string& key);
    static bool contains(const string& key);
private:
    static unordered_map<string, string> components;
};

#endif