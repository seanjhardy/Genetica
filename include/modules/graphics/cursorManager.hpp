#ifndef CURSOR_MANAGER
#define CURSOR_MANAGER

#include <unordered_map>
#include <string>
#include <SFML/Graphics.hpp>
#include <unordered_map>

class CursorManager {
public:
    static void init();
    static sf::Cursor& getDefault();
    static sf::Cursor& get(const std::string& name);
    static void set(sf::Cursor& cursor, const std::string& name);
    static void loadFromFile(const std::string& name, const std::string& path, const sf::Vector2u& size, const sf::Vector2u& hotspot={0,0});
private:
    static std::unordered_map<std::string, sf::Cursor> cursors;
};

#endif