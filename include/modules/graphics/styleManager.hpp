#ifndef STYLE_MANAGER
#define STYLE_MANAGER

#include <unordered_map>
#include <string>

class Styles {
public:
    static void init();

    static std::string get(const std::string& key);

private:
    static std::unordered_map<std::string, std::string> styles;
};

#endif