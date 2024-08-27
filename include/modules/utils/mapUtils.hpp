#ifndef MAP_UTILS
#define MAP_UTILS

#include <string>
#include <unordered_map>

using namespace std;

template<typename T>
inline void overrideValues(unordered_map<string, T>& map, unordered_map<string, T> s) {
    for (const auto& [property, value] : s) {
        map[property] = value;
    }
}

#endif