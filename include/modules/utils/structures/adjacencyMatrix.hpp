#ifndef ADJACENCY_MATRIX_HPP
#define ADJACENCY_MATRIX_HPP

#include <unordered_map>
#include <unordered_set>

template <typename T>
class AdjacencyMatrix {
private:
    std::unordered_map<T, std::unordered_set<T>> adj;

public:
    void addEdge(T u, T v);
    bool isConnected(T u, T v) const;
};

#include "../../../../src/modules/utils/structures/adjacencyMatrix.tpp"

#endif // ADJACENCY_MATRIX_HPP