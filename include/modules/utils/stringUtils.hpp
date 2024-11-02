#ifndef STRING_UTILS
#define STRING_UTILS

#include <string>

using namespace std;

// Function to trim leading and trailing spaces from a string
string trim(const string& str);
string stripHTML(const string &str);
string readFile(const string& filename);
vector<string> split(const string& str, const string& delimiter=" ");
string roundToDecimalPlaces(float number, int decimalPlaces = 3);
string formatNumber(float number, int precision=3);

#endif