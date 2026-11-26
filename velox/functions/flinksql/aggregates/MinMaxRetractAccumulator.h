/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <folly/container/F14Map.h>

#include <functional>
#include <utility>

#include "velox/common/memory/HashStringAllocator.h"

namespace facebook::velox::aggregate::flinksql {

/// Variable-width multiset behind the retractable min/max aggregates, the
/// counterpart of Flink's Min|MaxWithRetractAggFunction accumulator. The
/// accumulator itself lives in the fixed region of an aggregate row
/// (placement-constructed by the aggregate's initializeNewGroups), while the
/// entries and their hash table chunks live in the group's
/// HashStringAllocator - the same in-row folly::F14FastMap arrangement
/// exec's SetAccumulator uses. Every key carries a signed count:
///
/// - mapSize() counts the keys with a positive count; the final value
///   depends on nothing else: mapSize() > 0 means extreme() is the min/max
///   over the positive-count keys, mapSize() == 0 means an empty aggregate.
/// - a count of zero is never stored: counts cross zero only at the erase
///   boundary, from either direction.
/// - negative-count keys stay stored (retracting a never-accumulated key,
///   or merging partially retracted shards) but do not count toward
///   mapSize() and never win extreme().
///
/// The aggregate direction (min or max) comes in per call as a 'less'
/// comparator: std::less<T> for min, std::greater<T> for max - a key
/// comparing true against the current extreme replaces it.
template <typename T>
class MinMaxRetractAccumulator {
 public:
  explicit MinMaxRetractAccumulator(HashStringAllocator* allocator)
      : counts_{
            AlignedStlAllocator<std::pair<const T, int64_t>, 16>(allocator)} {}

  /// Adds one occurrence of 'key': inserts {key, 1} on a miss (mapSize grows,
  /// extreme may improve), or increments the stored count - a count coming
  /// back up from -1 reaches zero and the key disappears again.
  template <typename Compare>
  void accumulate(T key, Compare less) {
    auto [it, inserted] = counts_.try_emplace(key, 1);
    if (inserted) {
      ++mapSize_;
      if (mapSize_ == 1 || less(key, extreme_)) {
        extreme_ = key;
      }
      return;
    }
    auto& count = it->second;
    if (count == -1) {
      // -1 + 1 = 0, and zero-count keys are never stored.
      counts_.erase(it);
    } else {
      ++count;
    }
  }

  /// Removes one occurrence of 'key': a count dropping to zero erases the
  /// key and shrinks mapSize (when the key was the extreme, a rescan finds
  /// the next one, an empty rescan leaving mapSize at zero); a drop below
  /// zero stores the negative count as is. A key that was never accumulated
  /// stores -1 - Flink does not guard this either.
  template <typename Compare>
  void retract(T key, Compare less) {
    auto [it, inserted] = counts_.try_emplace(key, -1);
    if (inserted) {
      return;
    }
    if (it->second == 1) {
      counts_.erase(it);
      --mapSize_;
      // The erased key being the extreme - necessarily so when mapSize_
      // reaches zero, the only positive key was the extreme.
      if (key == extreme_) {
        rescanExtreme(less);
      }
    } else {
      --it->second;
    }
  }

  /// Merges one entry of another accumulator's multiset (the decoded
  /// MAP<T, BIGINT> intermediate of a partial state): sums the counts and
  /// keeps mapSize consistent with the sign crossing. Callers finish a
  /// sequence of merges with finishMerge().
  void mergeEntry(T key, int64_t count) {
    auto [it, inserted] = counts_.try_emplace(key, count);
    if (inserted) {
      if (count > 0) {
        ++mapSize_;
      }
      return;
    }
    const bool wasPositive = it->second > 0;
    it->second += count;
    if (it->second > 0) {
      if (!wasPositive) {
        ++mapSize_;
      }
    } else if (it->second == 0) {
      counts_.erase(it);
      if (wasPositive) {
        --mapSize_;
      }
    } else if (wasPositive) {
      --mapSize_;
    }
  }

  /// Completes a merge sequence the way Flink's merge ends with a marked
  /// updateMin: recomputes the extreme over the positive-count keys. Flink
  /// skips the rescan when no merged key touched the extreme; rescanning
  /// unconditionally yields the same result.
  template <typename Compare>
  void finishMerge(Compare less) {
    rescanExtreme(less);
  }

  /// Calls 'func(key, count)' for every stored entry, negative counts
  /// included - the entry set the MAP<T, BIGINT> intermediate carries.
  template <typename F>
  void forEachEntry(F&& func) const {
    for (const auto& [key, count] : counts_) {
      func(key, count);
    }
  }

  /// Number of stored entries, negative-count entries included.
  int64_t numEntries() const {
    return counts_.size();
  }

  /// The stored count of 'key', 0 for an absent key (zero-count keys are
  /// never stored, so 0 means both).
  int64_t countOf(T key) const {
    auto it = counts_.find(key);
    return it == counts_.end() ? 0 : it->second;
  }

  /// Number of keys with a positive count.
  int64_t mapSize() const {
    return mapSize_;
  }

  /// The current extreme value, valid only while mapSize() > 0.
  T extreme() const {
    return extreme_;
  }

  /// Returns the entry storage to the allocator. The aggregate's destroy
  /// path calls this before the row memory goes back (SetAccumulator::free
  /// pattern); the chunks free through the allocator embedded in the table.
  void free() {
    using Map = decltype(counts_);
    counts_.~Map();
  }

 private:
  /// Recomputes mapSize and extreme from the stored entries: the min/max
  /// over the positive-count keys, with an empty result leaving mapSize at
  /// zero - Flink's updateMin with its empty-scan fallback.
  template <typename Compare>
  void rescanExtreme(Compare less) {
    mapSize_ = 0;
    for (const auto& [key, count] : counts_) {
      if (count > 0) {
        ++mapSize_;
        if (mapSize_ == 1 || less(key, extreme_)) {
          extreme_ = key;
        }
      }
    }
  }

  /// The multiset: key -> signed occurrence count.
  folly::F14FastMap<
      T,
      int64_t,
      std::hash<T>,
      std::equal_to<T>,
      AlignedStlAllocator<std::pair<const T, int64_t>, 16>>
      counts_;

  /// Current extreme value, valid only while mapSize_ > 0.
  T extreme_{};

  /// Number of positive-count keys.
  int64_t mapSize_{0};
};

} // namespace facebook::velox::aggregate::flinksql
