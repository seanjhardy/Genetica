#ifndef PRINT
#define PRINT

#include "iostream"
#include "vector_types.h"
#include "tuple"
#include "vector"

// Template specialization to print a vector
template <typename T>
void printElement(const std::vector<T>& vec) {
    std::cout << "[ ";
    for (const auto& item : vec) {
        std::cout << item << " ";
    }
    std::cout << "]";
}

// Template specialization to print a tuple
template <typename Tuple, std::size_t... Is>
void printTuple(const Tuple& tup, std::index_sequence<Is...>) {
    (..., (std::cout << (Is == 0 ? "" : " ") << std::get<Is>(tup)));
}

template <typename... Args>
void printElement(const std::tuple<Args...>& tup) {
    std::cout << "(";
    printTuple(tup, std::index_sequence_for<Args...>{});
    std::cout << ")";
}

template <typename... Args>
void printElement(const float2& f2) {
    std::cout << f2.x << " " << f2.y;
}

// Template for generic types
template <typename T>
void printElement(const T& item) {
    std::cout << item;
}

// Variadic template to handle multiple arguments
template <typename T, typename... Args>
void print(const T& first, const Args&... args) {
    printElement(first);
    if constexpr (sizeof...(args) > 0) {
        std::cout << " ";
        print(args...);
    } else {
        std::cout << std::endl;
    }
}

#endif