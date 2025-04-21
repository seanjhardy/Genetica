#include <modules/graphics/cursorManager.hpp>
#include <fstream>
#include <sstream>

std::unordered_map<std::string, sf::Cursor> CursorManager::cursors;

void CursorManager::init() {
    loadFromFile("default", "./assets/icons/cursor.png",
                 sf::Vector2u(19, 28), sf::Vector2u(0, 0));

    // Load cursor from image path
    /*loadFromFile("dragHorizontal", "./assets/icons/drag_horizontal.png",
                 sf::Vector2u(39, 33), sf::Vector2u(19, 16));
    loadFromFile("dragVertical", "./assets/icons/drag_vertical.png",
                 sf::Vector2u(33, 39), sf::Vector2u(16, 19));
    loadFromFile("dragBottomLeft", "./assets/icons/drag_bl.png",
                 sf::Vector2u(32, 32), sf::Vector2u(16, 16));
    loadFromFile("dragBottomRight", "./assets/icons/drag_br.png",
                 sf::Vector2u(32, 32), sf::Vector2u(16, 16));
    loadFromFile("dragTopLeft", "./assets/icons/drag_tl.png",
                 sf::Vector2u(32, 32), sf::Vector2u(16, 16));
    loadFromFile("dragTopRight", "./assets/icons/drag_tr.png",
                 sf::Vector2u(32, 32), sf::Vector2u(16, 16));*/
    //Cursors at 80% size
    loadFromFile("dragHorizontal", "./assets/icons/drag_horizontal.png",
                 sf::Vector2u(31, 26), sf::Vector2u(15, 13));
    loadFromFile("dragVertical", "./assets/icons/drag_vertical.png",
                 sf::Vector2u(26, 31), sf::Vector2u(13, 15));
    loadFromFile("dragBottomLeft", "./assets/icons/drag_bl.png",
                 sf::Vector2u(25, 25), sf::Vector2u(13, 13));
    loadFromFile("dragBottomRight", "./assets/icons/drag_br.png",
                 sf::Vector2u(25, 25), sf::Vector2u(13, 13));
    loadFromFile("dragTopLeft", "./assets/icons/drag_tl.png",
                 sf::Vector2u(25, 25), sf::Vector2u(13, 13));
    loadFromFile("dragTopRight", "./assets/icons/drag_tr.png",
                 sf::Vector2u(25, 25), sf::Vector2u(13, 13));


    // Load cursors from system
    cursors["pointer"].loadFromSystem(sf::Cursor::Hand);
}

void CursorManager::loadFromFile(const std::string& name, const std::string& path, const sf::Vector2u& size,
                                 const sf::Vector2u& hotspot) {
    sf::Image image;

    // Load the image from the file
    if (!image.loadFromFile(path)) {
        throw std::runtime_error("Failed to load image from file");
    }

    cursors[name].loadFromPixels(image.getPixelsPtr(), size, hotspot);
}

sf::Cursor& CursorManager::get(const std::string& value) {
    return cursors[value];
}

void CursorManager::set(sf::Cursor& cursor, const std::string& name) {
    //cursor.loadFromSystem(cursor[name]);
}

sf::Cursor& CursorManager::getDefault() {
    return cursors["default"];
}
