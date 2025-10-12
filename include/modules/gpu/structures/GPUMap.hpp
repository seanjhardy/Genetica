#ifndef GPU_MAP_H
#define GPU_MAP_H

#include "GPUVector.hpp"
#include "cuda_runtime.h"

template<typename K, typename V>
class GPUMap {
    struct Entry {
        K key;
        V value;
    };

    GPUVector<Entry> entries;

public:
    __host__ GPUMap() = default;

    __host__ void insert(const K& key, const V& value);
    __host__ __device__ V& operator[](const K& key);
    __host__ __device__ bool contains(const K& key);
    __host__ __device__ void erase(const K& key);
    __host__ __device__ [[nodiscard]] size_t size() const;
    __host__ __device__ [[nodiscard]] bool empty() const;
};


#include "../../../src/modules/gpu/structures/GPUMap.tpp"

#endif // GPU_MAP_H