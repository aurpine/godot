/**************************************************************************/
/*  st_hash_map.h                                                         */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/os/memory.h"
#include "core/string/print_string.h" // IWYU pragma: keep. `WARN_VERBOSE` macro.
#include "core/templates/hashfuncs.h"
#include "core/templates/pair.h"
#include "core/templates/sort_list.h"

#if defined(__GNUC__) || defined(__clang__)
#include <emmintrin.h>
#include <immintrin.h>
#elif defined(_MSC_VER)
#include <intrin.h>
#endif

#include <initializer_list>

/**
 * A hash map implementation that uses open addressing with Swiss Table hashing.
 * Tries to be a drop-in replacement for HashMap.
 *
 * Keys and values are stored in a double linked list by insertion order. This
 * has a slight performance overhead on lookup, which can be mostly compensated
 * using a paged allocator if required.
 *
 * The assignment operator copy the pairs from one map to the other.
 */

template <typename TKey, typename TValue>
struct STHashMapElement {
	STHashMapElement *next = nullptr;
	STHashMapElement *prev = nullptr;
	KeyValue<TKey, TValue> data;
	STHashMapElement() {}
	STHashMapElement(const TKey &p_key, const TValue &p_value) :
			data(p_key, p_value) {}
};

template <typename TKey, typename TValue,
		typename Hasher = HashMapHasherDefault,
		typename Comparator = HashMapComparatorDefault<TKey>,
		typename Allocator = DefaultTypedAllocator<STHashMapElement<TKey, TValue>>>
class STHashMap : private Allocator {
public:
	static constexpr uint32_t MIN_CAPACITY_INDEX = 2; // Use a prime.
	static constexpr float MAX_OCCUPANCY = 0.90;
	static constexpr uint32_t EMPTY_HASH = 0;
	using KV = KeyValue<TKey, TValue>; // Type alias for easier access to KeyValue.

private:
	alignas(16) uint8_t *_control = nullptr;
	// capacity = groups * 16
	uint32_t _groups = 1;

	STHashMapElement<TKey, TValue> **_elements = nullptr;
	STHashMapElement<TKey, TValue> *_head_element = nullptr;
	STHashMapElement<TKey, TValue> *_tail_element = nullptr;

	uint32_t _size = 0;

	_FORCE_INLINE_ static uint32_t _hash(const TKey &p_key) {
		return Hasher::hash(p_key);
	}

	static constexpr uint32_t H1_MASK = 0xffffff80u;
	static constexpr uint32_t H2_MASK = 0x7fu;
	static constexpr uint8_t ZERO_BYTE = 0x80u;
	static constexpr uint8_t DELETED_BYTE = 0xFFu;

	_FORCE_INLINE_ static uint32_t _h1(uint32_t p_hash) {
		return (p_hash & H1_MASK) >> 7;
	}

	_FORCE_INLINE_ static uint8_t _h2(uint32_t p_hash) {
		// TODO: change this cast
		return (uint8_t)(p_hash & H2_MASK);
	}

	_FORCE_INLINE_ static int32_t _match(const __m128i &p_metadata, uint8_t p_h2) {
		return _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_set1_epi8(p_h2), p_metadata));
	}

	_FORCE_INLINE_ static int32_t _match_zeroes(const __m128i &p_metadata) {
		__m128i zeroes = _mm_set1_epi8(ZERO_BYTE);

		return _mm_movemask_epi8(_mm_cmpeq_epi8(zeroes, p_metadata));
	}

	_FORCE_INLINE_ static int32_t _match_zeroes_or_deleted(const __m128i &p_metadata) {
		__m128i mask = _mm_set1_epi8(ZERO_BYTE);

		return _mm_movemask_epi8(_mm_and_si128(mask, p_metadata));
	}

	bool _lookup_idx(const TKey &p_key, uint32_t &r_idx) const {
		return _elements != nullptr && _size > 0 && _lookup_idx_unchecked(p_key, _hash(p_key), r_idx);
	}

	/// Note: Assumes that _elements != nullptr
	bool _lookup_idx_unchecked(const TKey &p_key, uint32_t p_hash, uint32_t &r_idx) const {
		const uint32_t h1 = _h1(p_hash);
		const uint8_t h2 = _h2(p_hash);

		// uint32_t starting_group = h1 % _groups;
		uint32_t starting_group = h1 & (_groups - 1);
		uint32_t group = starting_group;
		uint32_t idx = group << 4;

		while (true) {
			__m128i *metadata = reinterpret_cast<__m128i *>(_control + idx);
			uint32_t matches = _match(*metadata, h2);

			while (matches != 0) {
#if defined(__GNUC__) || defined(__clang__)
				uint32_t pos = __builtin_ffs(matches) - 1;
#elif defined(_MSC_VER)
				unsigned long pos = 0;
				_BitScanForward(&pos, matches);
#endif
				idx += pos;

				// Check match
				if (Comparator::compare(_elements[idx]->data.key, p_key)) {
					r_idx = idx;
					return true;
				}
				// Otherwise go to the next one
				matches >>= pos + 1;
				idx++;
			}

			// Check zeroes
			if (_match_zeroes(*metadata)) {
				return false;
			}

			// Check next group
			group++;
			if (group >= _groups) {
				group -= _groups;
			}
			if (group == starting_group) {
				ERR_PRINT("SEARCHED THROUGH ALL CONTROL");
				return false;
			}
			idx = group << 4;
		}
	}

	void _insert_element(uint32_t p_hash, STHashMapElement<TKey, TValue> *p_value) {
		const uint32_t h1 = _h1(p_hash);
		const uint8_t h2 = _h2(p_hash);

		// uint32_t starting_group = h1 % _groups;
		uint32_t starting_group = h1 & (_groups - 1);
		uint32_t group = starting_group;

		while (true) {
			__m128i metadata = _mm_load_si128(reinterpret_cast<__m128i *>(_control + (group << 4)));
			uint32_t matches = _match_zeroes_or_deleted(metadata);

			if (matches) {
				// Find first match and update it
#if defined(__GNUC__) || defined(__clang__)
				uint32_t pos = __builtin_ffs(matches) - 1;
#elif defined(_MSC_VER)
				unsigned long pos = 0;
				_BitScanForward(&pos, matches);
#endif
				uint32_t idx = (group << 4) + pos;

				_elements[idx] = p_value;
				_control[idx] = h2;

				_size++;

				return;
			}

			// Check next group
			group++;
			if (group >= _groups) {
				group -= _groups;
			}
			ERR_FAIL_COND_MSG(group == starting_group, "BAD INSERT");
		}
	}

	void _resize_and_rehash(uint32_t p_new_groups) {
		_groups = p_new_groups;
		uint32_t capacity = _groups << 4;

		STHashMapElement<TKey, TValue> **old_elements = _elements;
		uint8_t *old_control = _control;

		_size = 0;

		_elements = reinterpret_cast<STHashMapElement<TKey, TValue> **>(Memory::alloc_static(sizeof(STHashMapElement<TKey, TValue> *) * capacity));
		_control = reinterpret_cast<uint8_t *>(Memory::alloc_aligned_static(sizeof(uint8_t) * capacity, 16));
		memset(_control, ZERO_BYTE, capacity);

		STHashMapElement<TKey, TValue> *elem = _head_element;

		while (elem != nullptr) {
			_insert_element(_hash(elem->data.key), elem);
			elem = elem->next;
		}

		Memory::free_static(old_elements);
		Memory::free_aligned_static(old_control);
	}

	_FORCE_INLINE_ STHashMapElement<TKey, TValue> *_insert(const TKey &p_key, const TValue &p_value, uint32_t p_hash, bool p_front_insert = false) {
		uint32_t capacity = _groups * 16;

		if (unlikely(_elements == nullptr)) {
			// Allocate on demand to save memory.

			static_assert(EMPTY_HASH == 0, "Assuming EMPTY_HASH = 0 for alloc_static_zeroed call");
			_elements = reinterpret_cast<STHashMapElement<TKey, TValue> **>(Memory::alloc_static(sizeof(STHashMapElement<TKey, TValue> *) * capacity));
			_control = reinterpret_cast<uint8_t *>(Memory::alloc_aligned_static(sizeof(uint8_t) * capacity, 16));
			memset(_control, ZERO_BYTE, capacity);
		}

		if (_size + 1 > MAX_OCCUPANCY * capacity) {
			ERR_FAIL_COND_V_MSG(capacity * 2ll >= 2147483648ull, nullptr, "Hash table maximum capacity reached, aborting insertion.");
			_resize_and_rehash(_groups * 2);
		}

		STHashMapElement<TKey, TValue> *elem = Allocator::new_allocation(STHashMapElement<TKey, TValue>(p_key, p_value));

		if (_tail_element == nullptr) {
			_head_element = elem;
			_tail_element = elem;
		} else if (p_front_insert) {
			_head_element->prev = elem;
			elem->next = _head_element;
			_head_element = elem;
		} else {
			_tail_element->next = elem;
			elem->prev = _tail_element;
			_tail_element = elem;
		}

		_insert_element(p_hash, elem);
		return elem;
	}

	void _clear_data() {
		STHashMapElement<TKey, TValue> *current = _tail_element;
		while (current != nullptr) {
			STHashMapElement<TKey, TValue> *prev = current->prev;
			Allocator::delete_allocation(current);
			current = prev;
		}
	}

public:
	_FORCE_INLINE_ uint32_t get_capacity() const { return _groups << 4; }
	_FORCE_INLINE_ uint32_t size() const { return _size; }

	/* Standard Godot Container API */

	bool is_empty() const {
		return _size == 0;
	}

	void clear() {
		if (_elements == nullptr || _size == 0) {
			return;
		}

		_clear_data();
		memset(_control, ZERO_BYTE, get_capacity());

		_tail_element = nullptr;
		_head_element = nullptr;
		_size = 0;
	}

	void sort() {
		sort_custom<KeyValueSort<TKey, TValue>>();
	}

	template <typename C>
	void sort_custom() {
		if (size() < 2) {
			return;
		}

		using E = STHashMapElement<TKey, TValue>;
		SortList<E, KeyValue<TKey, TValue>, &E::data, &E::prev, &E::next, C> sorter;
		sorter.sort(_head_element, _tail_element);
	}

	TValue &get(const TKey &p_key) _LIFETIME_BOUND_ {
		uint32_t idx = 0;
		bool exists = _lookup_idx(p_key, idx);
		CRASH_COND_MSG(!exists, "STHashMap key not found.");
		return _elements[idx]->data.value;
	}

	const TValue &get(const TKey &p_key) const _LIFETIME_BOUND_ {
		uint32_t idx = 0;
		bool exists = _lookup_idx(p_key, idx);
		CRASH_COND_MSG(!exists, "STHashMap key not found.");
		return _elements[idx]->data.value;
	}

	const TValue *getptr(const TKey &p_key) const _LIFETIME_BOUND_ {
		uint32_t idx = 0;
		bool exists = _lookup_idx(p_key, idx);

		if (exists) {
			return &_elements[idx]->data.value;
		}
		return nullptr;
	}

	TValue *getptr(const TKey &p_key) _LIFETIME_BOUND_ {
		uint32_t idx = 0;
		bool exists = _lookup_idx(p_key, idx);

		if (exists) {
			return &_elements[idx]->data.value;
		}
		return nullptr;
	}

	_FORCE_INLINE_ bool has(const TKey &p_key) const {
		uint32_t _idx = 0;
		return _lookup_idx(p_key, _idx);
	}

	bool erase(const TKey &p_key) {
		uint32_t idx = 0;
		bool exists = _lookup_idx(p_key, idx);

		if (!exists) {
			return false;
		}

		uint32_t group = idx / 16;

		__m128i metadata = _mm_load_si128(reinterpret_cast<__m128i *>(_control + (group * 16)));
		if (_match_zeroes(metadata)) {
			_control[idx] = ZERO_BYTE;
		} else {
			_control[idx] = DELETED_BYTE;
		}

		if (_head_element == _elements[idx]) {
			_head_element = _elements[idx]->next;
		}

		if (_tail_element == _elements[idx]) {
			_tail_element = _elements[idx]->prev;
		}

		if (_elements[idx]->prev) {
			_elements[idx]->prev->next = _elements[idx]->next;
		}

		if (_elements[idx]->next) {
			_elements[idx]->next->prev = _elements[idx]->prev;
		}

		Allocator::delete_allocation(_elements[idx]);

		_size--;
		return true;
	}

	// Reserves space for a number of elements, useful to avoid many resizes and rehashes.
	// If adding a known (possibly large) number of elements at once, must be larger than old capacity.
	void reserve(uint32_t p_new_capacity) {
		// TODO: add some warnings here.

		uint32_t new_groups = 1;
		uint32_t new_capacity = new_groups << 4;
		while ((new_groups << 4) * MAX_OCCUPANCY <= p_new_capacity) {
			new_groups <<= 1;
			new_capacity <<= 1;
		}

		if (new_groups <= _groups) {
			if (new_capacity < _size) {
				WARN_VERBOSE("reserve() called with a capacity smaller than the current size. This is likely a mistake.");
			}
			return;
		}

		if (_elements == nullptr) {
			_groups = new_groups;
			return; // Unallocated yet.
		}
		_resize_and_rehash(new_groups);
	}

	/** Iterator API **/

	struct ConstIterator {
		_FORCE_INLINE_ const KeyValue<TKey, TValue> &operator*() const {
			return E->data;
		}
		_FORCE_INLINE_ const KeyValue<TKey, TValue> *operator->() const { return &E->data; }
		_FORCE_INLINE_ ConstIterator &operator++() {
			if (E) {
				E = E->next;
			}
			return *this;
		}
		_FORCE_INLINE_ ConstIterator &operator--() {
			if (E) {
				E = E->prev;
			}
			return *this;
		}

		_FORCE_INLINE_ bool operator==(const ConstIterator &b) const { return E == b.E; }
		_FORCE_INLINE_ bool operator!=(const ConstIterator &b) const { return E != b.E; }

		_FORCE_INLINE_ explicit operator bool() const {
			return E != nullptr;
		}

		_FORCE_INLINE_ ConstIterator(const STHashMapElement<TKey, TValue> *p_E) { E = p_E; }
		_FORCE_INLINE_ ConstIterator() {}
		_FORCE_INLINE_ ConstIterator(const ConstIterator &p_it) { E = p_it.E; }
		_FORCE_INLINE_ void operator=(const ConstIterator &p_it) {
			E = p_it.E;
		}

	private:
		const STHashMapElement<TKey, TValue> *E = nullptr;
	};

	struct Iterator {
		_FORCE_INLINE_ KeyValue<TKey, TValue> &operator*() const {
			return E->data;
		}
		_FORCE_INLINE_ KeyValue<TKey, TValue> *operator->() const { return &E->data; }
		_FORCE_INLINE_ Iterator &operator++() {
			if (E) {
				E = E->next;
			}
			return *this;
		}
		_FORCE_INLINE_ Iterator &operator--() {
			if (E) {
				E = E->prev;
			}
			return *this;
		}

		_FORCE_INLINE_ bool operator==(const Iterator &b) const { return E == b.E; }
		_FORCE_INLINE_ bool operator!=(const Iterator &b) const { return E != b.E; }

		_FORCE_INLINE_ explicit operator bool() const {
			return E != nullptr;
		}

		_FORCE_INLINE_ Iterator(STHashMapElement<TKey, TValue> *p_E) { E = p_E; }
		_FORCE_INLINE_ Iterator() {}
		_FORCE_INLINE_ Iterator(const Iterator &p_it) { E = p_it.E; }
		_FORCE_INLINE_ void operator=(const Iterator &p_it) {
			E = p_it.E;
		}

		operator ConstIterator() const {
			return ConstIterator(E);
		}

	private:
		STHashMapElement<TKey, TValue> *E = nullptr;
	};

	_FORCE_INLINE_ Iterator begin() _LIFETIME_BOUND_ {
		return Iterator(_head_element);
	}
	_FORCE_INLINE_ Iterator end() _LIFETIME_BOUND_ {
		return Iterator(nullptr);
	}
	_FORCE_INLINE_ Iterator last() _LIFETIME_BOUND_ {
		return Iterator(_tail_element);
	}

	_FORCE_INLINE_ Iterator find(const TKey &p_key) _LIFETIME_BOUND_ {
		uint32_t idx = 0;
		bool exists = _lookup_idx(p_key, idx);
		if (!exists) {
			return end();
		}
		return Iterator(_elements[idx]);
	}

	_FORCE_INLINE_ void remove(const Iterator &p_iter) {
		if (p_iter) {
			erase(p_iter->key);
		}
	}

	_FORCE_INLINE_ ConstIterator begin() const _LIFETIME_BOUND_ {
		return ConstIterator(_head_element);
	}
	_FORCE_INLINE_ ConstIterator end() const _LIFETIME_BOUND_ {
		return ConstIterator(nullptr);
	}
	_FORCE_INLINE_ ConstIterator last() const _LIFETIME_BOUND_ {
		return ConstIterator(_tail_element);
	}

	_FORCE_INLINE_ ConstIterator find(const TKey &p_key) const _LIFETIME_BOUND_ {
		uint32_t idx = 0;
		bool exists = _lookup_idx(p_key, idx);
		if (!exists) {
			return end();
		}
		return ConstIterator(_elements[idx]);
	}

	/* Indexing */

	const TValue &operator[](const TKey &p_key) const _LIFETIME_BOUND_ {
		uint32_t idx = 0;
		bool exists = _lookup_idx(p_key, idx);
		CRASH_COND(!exists);
		return _elements[idx]->data.value;
	}

	TValue &operator[](const TKey &p_key) _LIFETIME_BOUND_ {
		const uint32_t hash = _hash(p_key);
		uint32_t idx = 0;
		bool exists = _elements && _size > 0 && _lookup_idx_unchecked(p_key, hash, idx);
		if (!exists) {
			return _insert(p_key, TValue(), hash)->data.value;
		} else {
			return _elements[idx]->data.value;
		}
	}

	/* Insert */

	Iterator insert(const TKey &p_key, const TValue &p_value, bool p_front_insert = false) {
		const uint32_t hash = _hash(p_key);
		uint32_t idx = 0;
		bool exists = _elements && _size > 0 && _lookup_idx_unchecked(p_key, hash, idx);
		if (!exists) {
			return Iterator(_insert(p_key, p_value, hash, p_front_insert));
		} else {
			_elements[idx]->data.value = p_value;
			return Iterator(_elements[idx]);
		}
	}

	/* Constructors */

	explicit STHashMap(const STHashMap &p_other) {
		reserve(p_other.get_capacity());

		if (p_other._size == 0) {
			return;
		}

		for (const KeyValue<TKey, TValue> &E : p_other) {
			insert(E.key, E.value);
		}
	}

	STHashMap(STHashMap &&p_other) {
		_elements = p_other._elements;
		_control = p_other._control;
		_head_element = p_other._head_element;
		_tail_element = p_other._tail_element;
		_groups = p_other._groups;
		_size = p_other._size;

		p_other._elements = nullptr;
		p_other._control = nullptr;
		p_other._head_element = nullptr;
		p_other._tail_element = nullptr;
		p_other._groups = 1;
		p_other._size = 0;
	}

	void operator=(const STHashMap &p_other) {
		if (this == &p_other) {
			return; // Ignore self assignment.
		}
		if (_size != 0) {
			clear();
		}

		reserve(p_other.get_capacity());

		if (p_other._elements == nullptr) {
			return; // Nothing to copy.
		}

		for (const KeyValue<TKey, TValue> &E : p_other) {
			insert(E.key, E.value);
		}
	}

	STHashMap &operator=(STHashMap &&p_other) {
		if (this == &p_other) {
			return *this;
		}

		if (_size != 0) {
			clear();
		}
		if (_elements != nullptr) {
			Memory::free_static(_elements);
			Memory::free_aligned_static(_control);
		}

		_elements = p_other._elements;
		_control = p_other._control;
		_head_element = p_other._head_element;
		_tail_element = p_other._tail_element;
		_groups = p_other._groups;
		_size = p_other._size;

		p_other._elements = nullptr;
		p_other._control = nullptr;
		p_other._head_element = nullptr;
		p_other._tail_element = nullptr;
		p_other._groups = 1;
		p_other._size = 0;

		return *this;
	}

	STHashMap(uint32_t p_initial_capacity) {
		// Capacity can't be 0.
		_groups = 1;
		reserve(p_initial_capacity);
	}
	STHashMap() {
		_groups = 1;
	}

	STHashMap(std::initializer_list<KeyValue<TKey, TValue>> p_init) {
		reserve(p_init.size());
		for (const KeyValue<TKey, TValue> &E : p_init) {
			insert(E.key, E.value);
		}
	}

	~STHashMap() {
		_clear_data();

		if (_elements != nullptr) {
			Memory::free_static(_elements);
			Memory::free_aligned_static(_control);
		}
	}
};
