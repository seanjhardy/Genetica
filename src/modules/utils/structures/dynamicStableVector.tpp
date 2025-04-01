#ifndef DYNAMIC_STABLE_VECTOR_TPP
#define DYNAMIC_STABLE_VECTOR_TPP

#include <modules/utils/structures/dynamicStableVector.hpp>

template <typename T>
size_t dynamicStableVector<T>::push(const T& value) {
    if (!freeList_.empty()) {
        // Reuse a free slot
        size_t index = *freeList_.begin(); // Get first free index
        freeList_.erase(index); // Remove that index from the set
        data_[index] = value;
        return index;
    }
    // Add a new slot at the end
    data_.emplace_back(value);
    return data_.size() - 1;
}

template <typename T>
void dynamicStableVector<T>::remove(size_t index) {
    if (index >= data_.size() || !data_[index]) {
        throw std::out_of_range("Invalid index for removal");
    }
    freeList_.insert(index);  // Mark the slot as free
}

template <typename T>
T& dynamicStableVector<T>::at(size_t index) {
    if (index >= data_.size()) {
        throw std::out_of_range("Invalid index for access");
    }
    return data_[index];
}

template <typename T>
const T& dynamicStableVector<T>::at(size_t index) const {
    if (index >= data_.size() || !data_[index]) {
        throw std::out_of_range("Invalid index for access");
    }
    return *data_[index];
}

template <typename T>
size_t dynamicStableVector<T>::size() const {
    return data_.size() - freeList_.size();
}

template <typename T>
bool dynamicStableVector<T>::isValid(size_t index) const {
    return index < data_.size() && data_[index] != nullptr;
}

template <typename T>
size_t dynamicStableVector<T>::getNextIndex() {
    if (!freeList_.empty()) {
        // Reuse a free slot
        size_t index = *freeList_.begin(); // Get first free index
        freeList_.erase(index); // Remove that index from the set
        return index;
    } else {
        // Add a new slot at the end
        return data_.size();
    }
}

template <typename T>
void dynamicStableVector<T>::clear() {
    data_.clear();
    freeList_.clear();
}

#endif  // DYNAMIC_STABLE_VECTOR_TPP
