#ifndef FUNCTION_MANAGER
#define FUNCTION_MANAGER

#include <unordered_map>
#include <string>
#include "SFML/Graphics.hpp"
#include <functional>

using namespace std;

class FunctionManager {
public:
    static function<void()>* get(const string& key);
    static void add(const string& key, function<void()> function);
    static void call(const string& key);

private:
    static unordered_map<string, function<void()>> functions;
};

#endif