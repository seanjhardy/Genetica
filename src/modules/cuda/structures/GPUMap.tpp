#pragma once

template<typename K, typename V>
void GPUMap<K, V>::insert(const K& key, const V& value) {
    Entry entry = {key, value};
    entries.push(entry);
}

template<typename K, typename V>
V& GPUMap<K, V>::operator[](const K& key) {
    for (size_t i = 0; i < entries.size(); ++i) {
        if (entries[i].key == key) {
            return entries[i].value;
        }
    }
    Entry entry = {key, V()};
    entries.push(entry);
    return entries[entries.size() - 1].value;
}

template<typename K, typename V>
bool GPUMap<K, V>::contains(const K& key) {
    for (size_t i = 0; i < entries.size(); ++i) {
        if (entries[i].key == key) {
            return true;
        }
    }
    return false;
}

template<typename K, typename V>
void GPUMap<K, V>::erase(const K& key) {
    for (size_t i = 0; i < entries.size(); ++i) {
        if (entries[i].key == key) {
            entries.remove(i);
            return;
        }
    }
}

template<typename K, typename V>
size_t GPUMap<K, V>::size() const {
    return entries.size();
}

template<typename K, typename V>
bool GPUMap<K, V>::empty() const {
    return entries.size() == 0;
}