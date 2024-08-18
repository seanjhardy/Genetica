#ifndef PRINT
#define PRINT

#include <iostream>
#include <vector_types.h>
#include <tuple>
#include <vector>
#include <unordered_map>
#include <string>
#include <modules/verlet/point.hpp>
#include <modules/graphics/UI/utils/size.hpp>
#include <modules/graphics/UI/utils/styleParser.hpp>

inline void printElement(const Size c) {
    std::cout <<
              (c.getMode() == Size::Mode::Pixel ? "Pixel(" :
               (c.getMode() == Size::Mode::Percent ? "Percent(" :
                (c.getMode() == Size::Mode::Flex ? "Flex(" : ""))) << c.getValue() << ")";
}

inline void printElement(const Point p) {
    std::cout << "Point(" << p.pos.x << ", " << p.pos.y << ", " << p.mass << ")";
}

// Template specialization to print a vector
template <typename T>
inline void printElement(const std::vector<T>& vec) {
    std::cout << "[ ";
    for (const auto& item : vec) {
        printElement(item);
        std::cout << " ";
    }
    std::cout << "]";
}


// Template specialization to print a tuple
template <typename Tuple, std::size_t... Is>
inline void printTuple(const Tuple& tup, std::index_sequence<Is...>) {
    (..., (std::cout << (Is == 0 ? "" : " ") << std::get<Is>(tup)));
}

template <typename... Args>
inline void printElement(const std::tuple<Args...>& tup) {
    std::cout << "(";
    printTuple(tup, std::index_sequence_for<Args...>{});
    std::cout << ")";
}

inline void printElement(const float2& f2) {
    std::cout << f2.x << " " << f2.y;
}

template<typename T>
inline void printElement(const sf::Vector2<T>& f2) {
    std::cout << f2.x << " " << f2.y;
}

inline void printElement(const sf::Color& c) {
    std::cout << "Color(" << static_cast<int>(c.r) << ", " << static_cast<int>(c.g) << ", " << static_cast<int>(c.b) << ", " << static_cast<int>(c.a) << ")";
}

inline void printElement(const Alignment& c) {
    std::cout << "Alignment(" <<
              (c == Alignment::Start ? "Start" :
               (c == Alignment::Center ? "Center" :
                (c == Alignment::End ? "End" :
                 (c == Alignment::SpaceBetween ? "SpaceBetween" :
                  (c == Alignment::SpaceAround ? "SpaceAround" : ""))))) << ")";
}

inline void printElement(const Direction& c) {
    std::cout << "Direction(" <<
              (c == Direction::Row ? "Row" :
               (c == Direction::Column ? "Column" : "")) << ")";
}

inline void printElement(const std::string& str) {
    std::cout << "\"" << str.c_str() << "\"";
}

inline void printElement(const int item) {
    std::cout << item;
}

template <typename K, typename V>
inline void printElement(const std::unordered_map<K, V>& map) {
    std::cout << "{ ";
    for (const auto& [key, value] : map) {
        std::cout << "(";
        printElement(key);
        std::cout << ": ";
        printElement(value);
        std::cout << ") ";
    }
    std::cout << "}";
}

// Template for generic types
template <typename T>
inline void printElement(const T& item) {
    std::cout << item;
}

// Variadic template to handle multiple arguments
template <typename T, typename... Args>
inline void print(const T& first, const Args&... args) {
    printElement(first);
    if constexpr (sizeof...(args) > 0) {
        std::cout << " ";
        print(args...);
    } else {
        std::cout << std::endl;
    }
}

#endif