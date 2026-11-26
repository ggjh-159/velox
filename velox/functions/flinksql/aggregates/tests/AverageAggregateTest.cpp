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
#include "velox/functions/flinksql/aggregates/AverageAggregate.h"

#include <gtest/gtest.h>

#include <optional>
#include <vector>

#include "velox/common/base/BitUtil.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/RowContainer.h"
#include "velox/experimental/stateful/agg/RetractableAggregate.h"
#include "velox/type/DecimalUtil.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/SelectivityVector.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::aggregate::flinksql::test {
namespace {

class AverageAggregateTest : public ::testing::Test,
                             public facebook::velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance({});
    registerAverageAggregate("", true, true);
  }

  // Drives one single-group accumulator directly - the way the group
  // aggregation operator drives aggregate functions outside the plan path -
  // so retraction can be exercised without a retract-aware plan.
  class Driver {
   public:
    Driver(
        std::vector<TypePtr> argTypes,
        const TypePtr& resultType,
        memory::MemoryPool* pool)
        : fn_(exec::Aggregate::create(
              "avg",
              core::AggregationNode::Step::kSingle,
              argTypes,
              resultType,
              queryConfig())),
          allocator_(pool),
          argTypes_(std::move(argTypes)) {
      fn_->setAllocator(&allocator_);
      const int32_t rowSizeOffset = bits::nbytes(1);
      int32_t offset = rowSizeOffset + sizeof(uint32_t);
      offset = bits::roundUp(offset, fn_->accumulatorAlignmentSize());
      fn_->setOffsets(
          offset,
          exec::RowContainer::nullByte(0),
          exec::RowContainer::nullMask(0),
          exec::RowContainer::initializedByte(0),
          exec::RowContainer::initializedMask(0),
          rowSizeOffset);
      row_.resize(offset + fn_->accumulatorFixedWidthSize());
      std::vector<char*> groups{row_.data()};
      const std::vector<vector_size_t> indices{0};
      fn_->initializeNewGroups(groups.data(), indices);
    }

    void add(const VectorPtr& arg) {
      run(arg->size(), arg, /*retract=*/false);
    }

    void retract(const VectorPtr& arg) {
      run(arg->size(), arg, /*retract=*/true);
    }

    template <typename TResult>
    std::optional<TResult> value() {
      auto result = BaseVector::create(fn_->resultType(), 1, allocator_.pool());
      std::vector<char*> groups{row_.data()};
      fn_->extractValues(groups.data(), 1, &result);
      if (result->isNullAt(0)) {
        return std::nullopt;
      }
      return result->as<FlatVector<TResult>>()->valueAt(0);
    }

    VectorPtr extractIntermediate() {
      auto result = BaseVector::create(
          exec::Aggregate::intermediateType("avg", argTypes_),
          1,
          allocator_.pool());
      std::vector<char*> groups{row_.data()};
      fn_->extractAccumulators(groups.data(), 1, &result);
      return result;
    }

    void addIntermediate(const VectorPtr& intermediate) {
      std::vector<char*> groups{row_.data()};
      fn_->addIntermediateResults(
          groups.data(), SelectivityVector(1), {intermediate}, false);
    }

   private:
    static core::QueryConfig queryConfig() {
      return core::QueryConfig({});
    }

    void run(int32_t numRows, const VectorPtr& arg, bool retract) {
      std::vector<VectorPtr> args{arg};
      const SelectivityVector rows(numRows);
      std::vector<char*> groups(numRows, row_.data());
      if (retract) {
        auto* retractable =
            dynamic_cast<stateful::RetractableAggregate*>(fn_.get());
        VELOX_CHECK_NOT_NULL(retractable);
        retractable->retractRawInput(groups.data(), rows, args, false);
      } else {
        fn_->addRawInput(groups.data(), rows, args, false);
      }
    }

    // The aggregate must outlive the allocator feeding it.
    std::unique_ptr<exec::Aggregate> fn_;
    HashStringAllocator allocator_;
    std::vector<char> row_;
    std::vector<TypePtr> argTypes_;
  };
};

TEST_F(AverageAggregateTest, integralTruncatingDivisionAndRetract) {
  Driver driver({BIGINT()}, BIGINT(), pool());
  driver.add(makeNullableFlatVector<int64_t>({10, 7, 2}));
  EXPECT_EQ(6, driver.value<int64_t>()); // 19 / 3 truncates.
  driver.retract(makeNullableFlatVector<int64_t>({7}));
  EXPECT_EQ(6, driver.value<int64_t>()); // 12 / 2.
  driver.retract(makeNullableFlatVector<int64_t>({2}));
  EXPECT_EQ(10, driver.value<int64_t>()); // 10 / 1.
}

TEST_F(AverageAggregateTest, retractToZeroReadsNull) {
  Driver driver({BIGINT()}, BIGINT(), pool());
  driver.add(makeNullableFlatVector<int64_t>({10}));
  ASSERT_EQ(10, driver.value<int64_t>());
  driver.retract(makeNullableFlatVector<int64_t>({10}));
  // The value follows the count alone: zero count is an empty aggregate.
  EXPECT_EQ(std::nullopt, driver.value<int64_t>());
  // The group comes back on the next input.
  driver.add(makeNullableFlatVector<int64_t>({5}));
  EXPECT_EQ(5, driver.value<int64_t>());
}

TEST_F(AverageAggregateTest, integralNullsSkipBothDirections) {
  Driver driver({BIGINT()}, BIGINT(), pool());
  driver.add(makeNullableFlatVector<int64_t>({1, std::nullopt, 2}));
  EXPECT_EQ(1, driver.value<int64_t>()); // 3 / 2 truncates.
  driver.retract(makeNullableFlatVector<int64_t>({std::nullopt, 2}));
  EXPECT_EQ(1, driver.value<int64_t>()); // 1 / 1.
  driver.retract(makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt}));
  EXPECT_EQ(1, driver.value<int64_t>());
}

TEST_F(AverageAggregateTest, integralSumWrapsAroundBothDirections) {
  Driver driver({BIGINT()}, BIGINT(), pool());
  const int64_t kMax = std::numeric_limits<int64_t>::max();
  driver.add(makeNullableFlatVector<int64_t>({kMax}));
  driver.add(makeNullableFlatVector<int64_t>({1}));
  // The sum wraps like Java's long addition: kMax + 1 = -2^63.
  EXPECT_EQ(-4611686018427387904, driver.value<int64_t>()); // -2^63 / 2.
  // Retraction wraps back: subtracting 1 restores kMax.
  driver.retract(makeNullableFlatVector<int64_t>({1}));
  EXPECT_EQ(kMax, driver.value<int64_t>());
}

TEST_F(AverageAggregateTest, overRetractKeepsNegativeCount) {
  Driver driver({BIGINT()}, BIGINT(), pool());
  driver.add(makeNullableFlatVector<int64_t>({10}));
  driver.retract(makeNullableFlatVector<int64_t>({10}));
  ASSERT_EQ(std::nullopt, driver.value<int64_t>());
  // Retraction does not stop at zero: the group keeps sum -3, count -1, and
  // the division of the negatives reads back the input, matching Flink,
  // which does not guard the running values.
  driver.retract(makeNullableFlatVector<int64_t>({3}));
  EXPECT_EQ(3, driver.value<int64_t>()); // -3 / -1.
  // Climbing back: -1 + 1 = 0 reads null, another input reads live again.
  driver.add(makeNullableFlatVector<int64_t>({3}));
  EXPECT_EQ(std::nullopt, driver.value<int64_t>());
  driver.add(makeNullableFlatVector<int64_t>({3}));
  EXPECT_EQ(3, driver.value<int64_t>());
}

TEST_F(AverageAggregateTest, intermediateRoundTripCarriesNegativeState) {
  Driver source({BIGINT()}, BIGINT(), pool());
  source.add(makeNullableFlatVector<int64_t>({10}));
  source.retract(makeNullableFlatVector<int64_t>({10}));
  source.retract(makeNullableFlatVector<int64_t>({3}));
  ASSERT_EQ(3, source.value<int64_t>()); // sum -3, count -1.

  // The intermediate carries the negative state, and re-merging it into a
  // fresh accumulator reproduces it - the arithmetic a state restore runs.
  Driver restored({BIGINT()}, BIGINT(), pool());
  restored.addIntermediate(source.extractIntermediate());
  EXPECT_EQ(3, restored.value<int64_t>());
  restored.add(makeNullableFlatVector<int64_t>({3}));
  EXPECT_EQ(std::nullopt, restored.value<int64_t>());
  restored.add(makeNullableFlatVector<int64_t>({3}));
  EXPECT_EQ(3, restored.value<int64_t>());
}

TEST_F(AverageAggregateTest, doubleAverageRetract) {
  Driver driver({DOUBLE()}, DOUBLE(), pool());
  driver.add(makeNullableFlatVector<double>({1.5, 2.5}));
  EXPECT_EQ(2.0, driver.value<double>());
  driver.retract(makeNullableFlatVector<double>({2.5}));
  EXPECT_EQ(1.5, driver.value<double>());
  driver.retract(makeNullableFlatVector<double>({1.5}));
  EXPECT_EQ(std::nullopt, driver.value<double>());
  driver.add(makeNullableFlatVector<double>({4.0}));
  EXPECT_EQ(4.0, driver.value<double>());
}

TEST_F(AverageAggregateTest, realAverageRoundsDoubleQuotient) {
  Driver driver({REAL()}, REAL(), pool());
  driver.add(makeNullableFlatVector<float>({1.0, 2.0}));
  EXPECT_EQ(1.5, driver.value<float>());
  driver.retract(makeNullableFlatVector<float>({2.0}));
  EXPECT_EQ(1.0, driver.value<float>());
}

TEST_F(AverageAggregateTest, shortDecimalRetract) {
  Driver driver({DECIMAL(10, 3)}, DECIMAL(38, 6), pool());
  driver.add(makeNullableFlatVector<int64_t>({12500, 7250}, DECIMAL(10, 3)));
  // 19.750 / 2 = 9.875: unscaled 9875 at the input scale, rescaled by 10^3
  // to the result scale of 6.
  EXPECT_EQ((int128_t)9875000, driver.value<int128_t>());
  driver.retract(makeNullableFlatVector<int64_t>({7250}, DECIMAL(10, 3)));
  EXPECT_EQ((int128_t)12500000, driver.value<int128_t>());
}

TEST_F(AverageAggregateTest, longDecimalOverflowCarriesThroughRetract) {
  Driver driver({DECIMAL(38, 8)}, DECIMAL(38, 8), pool());
  const int128_t kMax = DecimalUtil::kLongDecimalMax;
  driver.add(makeNullableFlatVector<int128_t>({kMax}, DECIMAL(38, 8)));
  driver.add(makeNullableFlatVector<int128_t>({kMax}, DECIMAL(38, 8)));
  // The doubled max overflows the 127-bit magnitude the accumulator keeps:
  // the overflow word holds the carry and the final value divides it back
  // in.
  EXPECT_EQ(kMax, driver.value<int128_t>());

  // The serialized intermediate carries the overflow word: restoring it
  // into a fresh accumulator reproduces the same value.
  Driver restored({DECIMAL(38, 8)}, DECIMAL(38, 8), pool());
  restored.addIntermediate(driver.extractIntermediate());
  EXPECT_EQ(kMax, restored.value<int128_t>());

  // Retraction runs the representation backwards: the first subtract
  // borrows nothing (mixed signs), the second takes the carry back out of
  // the overflow word, and the state returns to zero.
  driver.retract(makeNullableFlatVector<int128_t>({kMax}, DECIMAL(38, 8)));
  EXPECT_EQ(kMax, driver.value<int128_t>());
  driver.retract(makeNullableFlatVector<int128_t>({kMax}, DECIMAL(38, 8)));
  EXPECT_EQ(std::nullopt, driver.value<int128_t>());
  driver.add(makeNullableFlatVector<int128_t>({kMax}, DECIMAL(38, 8)));
  EXPECT_EQ(kMax, driver.value<int128_t>());
}

TEST_F(AverageAggregateTest, decimalOverRetractKeepsNegativeCount) {
  Driver driver({DECIMAL(10, 3)}, DECIMAL(38, 6), pool());
  driver.add(makeNullableFlatVector<int64_t>({10000}, DECIMAL(10, 3)));
  driver.retract(makeNullableFlatVector<int64_t>({10000}, DECIMAL(10, 3)));
  ASSERT_EQ(std::nullopt, driver.value<int128_t>());
  // sum -3000, count -1: the division of the negatives reads back the
  // input, rescaled to the result scale like any other value.
  driver.retract(makeNullableFlatVector<int64_t>({3000}, DECIMAL(10, 3)));
  EXPECT_EQ((int128_t)3000000, driver.value<int128_t>());
  driver.add(makeNullableFlatVector<int64_t>({3000}, DECIMAL(10, 3)));
  EXPECT_EQ(std::nullopt, driver.value<int128_t>());
  driver.add(makeNullableFlatVector<int64_t>({3000}, DECIMAL(10, 3)));
  EXPECT_EQ((int128_t)3000000, driver.value<int128_t>());
}

} // namespace
} // namespace facebook::velox::aggregate::flinksql::test
