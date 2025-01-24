#ifndef DYNAMIC_STABLE_VECTOR_HPP
#define DYNAMIC_STABLE_VECTOR_HPP

#include <vector>
#include <memory>
#include <stdexcept>

// DynamicStableVector: A container that provides stable indices even when items are added or removed.
template <typename T>
class DynamicStableVector {
    std::vector<std::unique_ptr<T>> data_;  // Stores pointers to objects
    std::vector<size_t> freeList_;          // Tracks free indices for reuse

    class iterator {
    	typename std::vector<std::unique_ptr<T>>::iterator current_;
    	typename std::vector<std::unique_ptr<T>>::iterator end_;

   		void skipNullptr() {
        	while (current_ != end_ && !(*current_)) {
            	++current_;
        	}
    	}

	public:
    	iterator(typename std::vector<std::unique_ptr<T>>::iterator current,
             typename std::vector<std::unique_ptr<T>>::iterator end)
        	: current_(current), end_(end) {
        	skipNullptr();
    	}

    	T& operator*() const { return **current_; }
    	T* operator->() const { return current_->get(); }

    	iterator& operator++() {
    	    ++current_;
    	    skipNullptr();
    	    return *this;
    	}

    	bool operator!=(const iterator& other) const { return current_ != other.current_; }
    	bool operator==(const iterator& other) const { return current_ == other.current_; }
	};

public:
    DynamicStableVector() = default;

    // Insert a new object and return its index
    size_t insert(const T& value);

    // Remove an object at a given index
    void remove(size_t index);

    // Access an object at a given index
    T& at(size_t index);
    const T& at(size_t index) const;

    // Get the current number of valid objects
    size_t size() const;

    // Check if an index is valid
    bool isValid(size_t index) const;

   	size_t getNextIndex();

    void clear();

    iterator begin() { return iterator(data_.begin(), data_.end()); }
	iterator end() { return iterator(data_.end(), data_.end()); }

    T& operator[](size_t index) { return at(index); }
    const T& operator[](size_t index) const { return at(index); }

	size_t getSize() { return data_.size(); }
};

#include "../../../../src/modules/utils/structures/DynamicStableVector.tpp"

#endif  // DYNAMIC_STABLE_VECTOR_HPP
