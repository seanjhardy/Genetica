#ifndef ADJACENCY_MATRIX_TPP
#define ADJACENCY_MATRIX_TPP

template <typename T>
void AdjacencyMatrix<T>::addEdge(T u, T v) {
    adj[u].insert(v);
    adj[v].insert(u);
}

template <typename T>
bool AdjacencyMatrix<T>::isConnected(T u, T v) const {
    auto it = adj.find(u);
    return it != adj.end() && it->second.count(v);
}

#endif