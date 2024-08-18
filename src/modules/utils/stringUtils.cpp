#include <string>
#include <sstream>
#include <regex>

// Function to trim leading and trailing spaces from a string
std::string trimSpaces(const std::string& str) {
    // Lambda function to check if a character is a whitespace
    auto isSpace = [](char ch) { return std::isspace(static_cast<unsigned char>(ch)); };

    // Find the first non-whitespace character
    auto start = std::find_if_not(str.begin(), str.end(), isSpace);

    // If the entire string is whitespace
    if (start == str.end()) {
        return "";
    }

    // Find the last non-whitespace character
    auto end = std::find_if_not(str.rbegin(), str.rend(), isSpace).base();

    // Create a new string with trimmed spaces
    return {start, end};
}