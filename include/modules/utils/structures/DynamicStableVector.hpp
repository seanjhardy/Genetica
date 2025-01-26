#ifndef DYNAMIC_STABLE_VECTOR_HPP
#define DYNAMIC_STABLE_VECTOR_HPP

#include <vector>
#include <unordered_set>

// DynamicStableVector: A container that provides stable indices even when items are added or removed.
template <typename T>
class DynamicStableVector {
public:
    std::vector<T> data_;  // Stores pointers to objects
    std::unordered_set<size_t> freeList_; // Tracks free indices for reuse

	class Iterator {
	public:
		using iterator_category = std::forward_iterator_tag;
		using value_type = T;
		using difference_type = std::ptrdiff_t;
		using pointer = T*;
		using reference = T&;

	private:
		typename std::vector<T>::iterator current_;
		typename std::vector<T>::iterator end_;
		const std::unordered_set<size_t>* freeList_;
		size_t index_;

		void skipFreeSlots() {
			while (current_ != end_ && freeList_->contains(index_)) {
				++current_;
				++index_;
			}
		}

	public:
		Iterator(typename std::vector<T>::iterator current,
				 typename std::vector<T>::iterator end,
				 const std::unordered_set<size_t>* freeList,
				 size_t startIndex)
			: current_(current), end_(end), freeList_(freeList), index_(startIndex) {
			skipFreeSlots();
		}

		reference operator*() { return *current_; }
		pointer operator->() { return &(*current_); }

		Iterator& operator++() {
			++current_;
			++index_;
			skipFreeSlots();
			return *this;
		}

		Iterator operator++(int) {
			Iterator tmp = *this;
			++(*this);
			return tmp;
		}

		bool operator!=(const Iterator& other) const {
			return current_ != other.current_;
		}

		bool operator==(const Iterator& other) const {
			return current_ == other.current_;
		}
	};

    DynamicStableVector() = default;

    // Insert a new object and return its index
    size_t push(const T& value);

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



	Iterator begin() { return Iterator(data_.begin(), data_.end(), &freeList_, 0); }
	Iterator end() { return Iterator(data_.end(), data_.end(), &freeList_, data_.size()); }

    T& operator[](size_t index) { return at(index); }
    const T& operator[](size_t index) const { return at(index); }

	size_t getSize() { return data_.size(); }
};

#include "../../../../src/modules/utils/structures/DynamicStableVector.tpp"

#endif  // DYNAMIC_STABLE_VECTOR_HPP
