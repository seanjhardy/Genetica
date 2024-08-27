#include "modules/graphics/functionManager.hpp"
#include <sstream>
#include <functional>
#include <string>

using namespace std;

unordered_map<string, function<void()>> FunctionManager::functions;

function<void()>* FunctionManager::get(const string& key) {
    auto it = functions.find(key);
    if (it != functions.end()) {
        return &it->second;
    }
    return nullptr;
}

void FunctionManager::add(const string& key, function<void()> function) {
    functions[key] = function;
}