#include <string>
#include <sstream>
#include <regex>
#include <fstream>
#include <modules/utils/print.hpp>

using namespace std;

// Function to trim leading and trailing spaces from a string
string trim(const string& str) {
    // Lambda function to check if a character is a whitespace
    auto isSpace = [](char ch) { return isspace(static_cast<unsigned char>(ch)); };

    // Find the first non-whitespace character
    auto start = find_if_not(str.begin(), str.end(), isSpace);

    // If the entire string is whitespace
    if (start == str.end()) {
        return "";
    }

    // Find the last non-whitespace character
    auto end = find_if_not(str.rbegin(), str.rend(), isSpace).base();

    // Create a new string with trimmed spaces
    return {start, end};
}

string stripHTML(const string& str) {
    string newString = str;

    // Remove comment
    std::string startTag = "<!--";
    std::string endTag = "-->";

    size_t startPos = newString.find(startTag);
    while (startPos != std::string::npos) {
        size_t endPos = newString.find(endTag, startPos + startTag.length());
        if (endPos != std::string::npos) {
            newString.erase(startPos, endPos + endTag.length() - startPos);
        } else {
            break; // Malformed comment; stop processing
        }
        startPos = newString.find(startTag);
    }

    // Remove newlines
    newString = regex_replace(newString, regex("\n"), "");

    return newString;
}

string readFile(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Unable to open file: " + filename);
    }

    stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

vector<string> split(const string& str, const string& delimiter) {
    vector<string> tokens;
    size_t start = 0, end = 0;
    while ((end = str.find(delimiter, start)) != string::npos) {
        tokens.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
    }
    tokens.push_back(str.substr(start));
    return tokens;
}