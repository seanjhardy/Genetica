#include "modules/graphics/styleManager.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <string>
#include <iostream>
#include <modules/utils/stringUtils.hpp>
#include <modules/utils/print.hpp>

namespace fs = std::filesystem;

std::unordered_map<std::string, std::string> Styles::styles;

void Styles::init() {
    std::string basePath = "./assets/styles/";

    // Traverse all files in the directory and its subdirectories
    for (const auto &entry: fs::recursive_directory_iterator(basePath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".css") {
            std::ifstream file(entry.path());
            if (!file.is_open()) {
                std::cerr << "Failed to open " << entry.path() << std::endl;
                continue;
            }

            std::string line;
            std::string styleName;
            std::string styleContent;
            bool insideStyle = false;

            while (std::getline(file, line)) {
                line.erase(0, line.find_first_not_of(" \t\r\n"));
                line.erase(line.find_last_not_of(" \t\r\n") + 1);

                if (line.empty()) {
                    continue;  // Skip empty lines
                }

                std::istringstream iss(line);
                std::string token;
                iss >> token;

                if (!insideStyle && line.contains("{")) {
                    styleName = trim(line.substr(0, line.find("{")));
                    styleContent.clear();
                    insideStyle = true;
                } else if (insideStyle) {
                    if (line.contains("}")) {
                        // End of the style block
                        styles[styleName] = styleContent;
                        insideStyle = false;
                    } else {
                        // Append line to style content
                        styleContent += line + "\n";
                    }
                }
            }

            file.close();
        }
    }
}

std::string Styles::get(const std::string& key) {
    auto it = styles.find(key);
    if (it != styles.end()) {
        return it->second;
    }
    return "";
}