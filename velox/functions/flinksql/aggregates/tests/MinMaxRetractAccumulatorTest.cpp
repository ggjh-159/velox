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
#include "velox/functions/flinksql/aggregates/MinMaxRetractAccumulator.h"

#include <gtest/gtest.h>

#include <functional>

#include "velox/common/memory/HashStringAllocator.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::aggregate::flinksql::test {
namespace {

class MinMaxRetractAccumulatorTest
    : public ::testing::Test,
      public facebook::velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance({});
  }

  std::unique_ptr<HashStringAllocator> allocator_{
      std::make_unique<HashStringAllocator>(pool())};
};

TEST_F(MinMaxRetractAccumulatorTest, repeatKeyAccumulateAndRetract) {
  MinMaxRetractAccumulator<int64_t> acc(allocator_.get());
  std::less<int64_t> less;
  for (auto i = 0; i < 3; ++i) {
    acc.accumulate(5, less);
  }
  for (auto i = 0; i < 2; ++i) {
    acc.accumulate(7, less);
  }
  acc.retract(5, less);

  EXPECT_EQ(2, acc.countOf(5));
  EXPECT_EQ(2, acc.countOf(7));
  EXPECT_EQ(2, acc.mapSize());
  EXPECT_EQ(2, acc.numEntries());
  EXPECT_EQ(5, acc.extreme());

  int64_t total = 0;
  acc.forEachEntry([&](int64_t, int64_t count) { total += count; });
  EXPECT_EQ(4, total);
}

TEST_F(MinMaxRetractAccumulatorTest, retractToZeroErasesKey) {
  MinMaxRetractAccumulator<int64_t> acc(allocator_.get());
  std::less<int64_t> less;
  acc.accumulate(5, less);
  acc.accumulate(7, less);
  acc.retract(7, less);

  // Retracting a non-extreme key leaves the extreme alone.
  EXPECT_EQ(0, acc.countOf(7));
  EXPECT_EQ(1, acc.numEntries());
  EXPECT_EQ(1, acc.mapSize());
  EXPECT_EQ(5, acc.extreme());
}

TEST_F(MinMaxRetractAccumulatorTest, retractExtremeRescansAndEmptyFallsBack) {
  MinMaxRetractAccumulator<int64_t> acc(allocator_.get());
  std::less<int64_t> less;
  acc.accumulate(3, less);
  acc.accumulate(5, less);
  acc.accumulate(7, less);
  ASSERT_EQ(3, acc.extreme());

  acc.retract(3, less);
  EXPECT_EQ(5, acc.extreme());
  acc.retract(5, less);
  EXPECT_EQ(7, acc.extreme());

  // The last positive key goes: the empty rescan leaves mapSize at zero,
  // which is the aggregate's null-value signal.
  acc.retract(7, less);
  EXPECT_EQ(0, acc.mapSize());
  EXPECT_EQ(0, acc.numEntries());
}

TEST_F(MinMaxRetractAccumulatorTest, retractNeverAccumulatedStoresNegative) {
  MinMaxRetractAccumulator<int64_t> acc(allocator_.get());
  std::less<int64_t> less;

  // A malformed changelog retracts a key that was never accumulated: the
  // count goes to -1 and stays visible, like Flink's unguarded map.
  acc.retract(9, less);
  EXPECT_EQ(-1, acc.countOf(9));
  EXPECT_EQ(1, acc.numEntries());
  EXPECT_EQ(0, acc.mapSize());

  // Coming back up: -1 + 1 = 0 erases the key again.
  acc.accumulate(9, less);
  EXPECT_EQ(0, acc.countOf(9));
  EXPECT_EQ(0, acc.numEntries());
  EXPECT_EQ(0, acc.mapSize());

  // Deeper negative counts climb back one step at a time.
  acc.retract(9, less);
  acc.retract(9, less);
  ASSERT_EQ(-2, acc.countOf(9));
  acc.accumulate(9, less);
  EXPECT_EQ(-1, acc.countOf(9));
  EXPECT_EQ(0, acc.mapSize());
}

TEST_F(MinMaxRetractAccumulatorTest, negativeEntriesDoNotWinExtreme) {
  MinMaxRetractAccumulator<int64_t> acc(allocator_.get());
  std::less<int64_t> less;
  acc.retract(4, less);
  acc.retract(4, less);
  ASSERT_EQ(-2, acc.countOf(4));
  ASSERT_EQ(0, acc.mapSize());

  acc.accumulate(6, less);
  EXPECT_EQ(6, acc.extreme());
  EXPECT_EQ(1, acc.mapSize());
  EXPECT_EQ(-2, acc.countOf(4));
}

TEST_F(MinMaxRetractAccumulatorTest, maxDirectionViaGreater) {
  MinMaxRetractAccumulator<int64_t> acc(allocator_.get());
  std::greater<int64_t> greater;
  acc.accumulate(3, greater);
  acc.accumulate(5, greater);
  acc.accumulate(7, greater);
  ASSERT_EQ(7, acc.extreme());

  acc.retract(7, greater);
  EXPECT_EQ(5, acc.extreme());
}

TEST_F(MinMaxRetractAccumulatorTest, doubleKeys) {
  MinMaxRetractAccumulator<double> acc(allocator_.get());
  std::less<double> less;
  acc.accumulate(2.5, less);
  acc.accumulate(1.5, less);
  ASSERT_EQ(1.5, acc.extreme());

  acc.retract(1.5, less);
  EXPECT_EQ(2.5, acc.extreme());
}

TEST_F(MinMaxRetractAccumulatorTest, mergeEntrySignMath) {
  MinMaxRetractAccumulator<int64_t> acc(allocator_.get());
  std::less<int64_t> less;
  acc.accumulate(10, less);
  acc.accumulate(10, less);
  acc.accumulate(20, less);
  acc.retract(30, less);
  acc.accumulate(40, less);
  ASSERT_EQ(3, acc.mapSize());
  ASSERT_EQ(10, acc.extreme());

  acc.mergeEntry(10, -1); // 2 - 1 = 1: stays positive, mapSize unchanged.
  acc.mergeEntry(20, -1); // 1 - 1 = 0: erased, mapSize shrinks.
  acc.mergeEntry(30, 1); // -1 + 1 = 0: erased, was not positive.
  acc.mergeEntry(40, -3); // 1 - 3 = -2: stored negative, mapSize shrinks.
  acc.mergeEntry(5, 2); // New positive key, mapSize grows.
  acc.finishMerge(less);

  EXPECT_EQ(5, acc.extreme());
  EXPECT_EQ(2, acc.mapSize());
  EXPECT_EQ(3, acc.numEntries());
  EXPECT_EQ(1, acc.countOf(10));
  EXPECT_EQ(0, acc.countOf(20));
  EXPECT_EQ(0, acc.countOf(30));
  EXPECT_EQ(-2, acc.countOf(40));
  EXPECT_EQ(2, acc.countOf(5));
}

TEST_F(MinMaxRetractAccumulatorTest, finishMergeRescansWhenExtremeKeyDies) {
  MinMaxRetractAccumulator<int64_t> acc(allocator_.get());
  std::less<int64_t> less;
  acc.accumulate(10, less);
  acc.accumulate(20, less);
  ASSERT_EQ(10, acc.extreme());

  acc.mergeEntry(10, -1);
  // The extreme is stale until finishMerge: merge defers its maintenance.
  EXPECT_EQ(10, acc.extreme());
  acc.finishMerge(less);
  EXPECT_EQ(20, acc.extreme());
  EXPECT_EQ(1, acc.mapSize());
}

TEST_F(MinMaxRetractAccumulatorTest, manyKeysGrowAndDrain) {
  MinMaxRetractAccumulator<int64_t> acc(allocator_.get());
  std::less<int64_t> less;
  for (int64_t key = 0; key < 2000; ++key) {
    acc.accumulate(key, less);
  }
  ASSERT_EQ(2000, acc.mapSize());
  ASSERT_EQ(0, acc.extreme());

  // Retracting the current extreme forces a rescan over the full table.
  acc.retract(0, less);
  EXPECT_EQ(1, acc.extreme());
  for (int64_t key = 1; key < 2000; ++key) {
    acc.retract(key, less);
  }
  EXPECT_EQ(0, acc.mapSize());
  EXPECT_EQ(0, acc.numEntries());
}

TEST_F(MinMaxRetractAccumulatorTest, freeReturnsAllMemory) {
  // The production lifecycle: placement-constructed, freed explicitly, no
  // destructor ever runs (aggregate rows go back to their container).
  using Acc = MinMaxRetractAccumulator<int64_t>;
  alignas(Acc) std::byte storage[sizeof(Acc)];
  auto* acc = new (storage) Acc(allocator_.get());
  std::less<int64_t> less;
  for (int64_t key = 0; key < 100; ++key) {
    acc->accumulate(key, less);
  }
  acc->retract(50, less);
  ASSERT_GT(allocator_->currentBytes(), 0);

  acc->free();
  EXPECT_EQ(0, allocator_->currentBytes());
}

} // namespace
} // namespace facebook::velox::aggregate::flinksql::test
